/// OpenCL miner implementation.
///
/// @file
/// @copyright GNU General Public License

// Modified by Evgeniy Sukhomlinov 2018

#include "AOCLMiner.h"
#include "Utils/PathUtils.h"
#include <fstream>
#include "Hash/sha256_mod.h"
#include <boost/algorithm/string.hpp>
#include "Utils/Utils.h"
#include "AOCLUtils/aocl_utils.h"

extern "C" FILE* __cdecl __iob_func()
{
	struct _iobuf_VS2012 { // ...\Microsoft Visual Studio 11.0\VC\include\stdio.h #56
		char *_ptr;
		int   _cnt;
		char *_base;
		int   _flag;
		int   _file;
		int   _charbuf;
		int   _bufsiz;
		char *_tmpfname; };
	// VS2015 has only FILE = struct {void*}

	int const count = sizeof(_iobuf_VS2012) / sizeof(FILE);

	//// stdout
	//return (FILE*)(&(__acrt_iob_func(1)->_Placeholder) - count);

	// stderr
	return (FILE*)(&(__acrt_iob_func(2)->_Placeholder) - 2 * count);
}
extern "C" void __imp__set_output_format(void) {};

using namespace XDag;
using namespace aocl_utils;

#define OUTPUT_SIZE 15  //15 positions in output buffer + 1 position for flag
#define KERNEL_ARG_NONCE 0
#define KERNEL_ARG_STATE 1
#define KERNEL_ARG_PRECALC_STATE 2
#define KERNEL_ARG_DATA 3
#define KERNEL_ARG_TARGET_H 4
#define KERNEL_ARG_TARGET_G 5
#define KERNEL_ARG_OUTPUT 6
#define KERNEL_ITERATIONS 16

#define MAX_GPU_ERROR_COUNT 3

uint32_t AOCLMiner::_sWorkgroupSize = AOCLMiner::_defaultLocalWorkSize;
uint32_t AOCLMiner::_sInitialGlobalWorkSize = AOCLMiner::_defaultGlobalWorkSizeMultiplier * AOCLMiner::_defaultLocalWorkSize;
#ifdef __linux__
std::string AOCLMiner::_clKernelName = "CL/AOCLMiner_kernel.aocx";
#else
std::string AOCLMiner::_clKernelName = "AOCLMiner_kernel.aocx";
#endif
bool AOCLMiner::_useOpenClCpu = false;
bool AOCLMiner::_useNvidiaFix = false;
double AOCLMiner::_nvidiaSpinDamp = AOCLMiner::_defaultNvidiaSpinDamp / 100.0;
bool AOCLMiner::_useVectors = false;

struct AOCLChannel : public LogChannel
{
    static const char* name() { return EthOrange " aocl"; }
    static const int verbosity = 2;
    static const bool debug = false;
};
#define cllog clog(AOCLChannel)
#define XCL_LOG(_contents) cllog << _contents

/**
 * Returns the name of a numerical cl_int error
 * Takes constants from CL/cl.h and returns them in a readable format
 */
static const char *strClError(cl_int err)
{
    switch(err)
    {
    case CL_SUCCESS:
        return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND:
        return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE:
        return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE:
        return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES:
        return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY:
        return "CL_OUT_OF_HOST_MEMORY";
    case CL_PROFILING_INFO_NOT_AVAILABLE:
        return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_MEM_COPY_OVERLAP:
        return "CL_MEM_COPY_OVERLAP";
    case CL_IMAGE_FORMAT_MISMATCH:
        return "CL_IMAGE_FORMAT_MISMATCH";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
        return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_BUILD_PROGRAM_FAILURE:
        return "CL_BUILD_PROGRAM_FAILURE";
    case CL_MAP_FAILURE:
        return "CL_MAP_FAILURE";
    case CL_MISALIGNED_SUB_BUFFER_OFFSET:
        return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
        return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";

#ifdef CL_VERSION_1_2
    case CL_COMPILE_PROGRAM_FAILURE:
        return "CL_COMPILE_PROGRAM_FAILURE";
    case CL_LINKER_NOT_AVAILABLE:
        return "CL_LINKER_NOT_AVAILABLE";
    case CL_LINK_PROGRAM_FAILURE:
        return "CL_LINK_PROGRAM_FAILURE";
    case CL_DEVICE_PARTITION_FAILED:
        return "CL_DEVICE_PARTITION_FAILED";
    case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
        return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
#endif // CL_VERSION_1_2

    case CL_INVALID_VALUE:
        return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE:
        return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM:
        return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE:
        return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT:
        return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES:
        return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_COMMAND_QUEUE:
        return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_HOST_PTR:
        return "CL_INVALID_HOST_PTR";
    case CL_INVALID_MEM_OBJECT:
        return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
        return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_IMAGE_SIZE:
        return "CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_SAMPLER:
        return "CL_INVALID_SAMPLER";
    case CL_INVALID_BINARY:
        return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS:
        return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM:
        return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE:
        return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME:
        return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_DEFINITION:
        return "CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL:
        return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX:
        return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE:
        return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE:
        return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_ARGS:
        return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION:
        return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE:
        return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE:
        return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_OFFSET:
        return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_EVENT_WAIT_LIST:
        return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_EVENT:
        return "CL_INVALID_EVENT";
    case CL_INVALID_OPERATION:
        return "CL_INVALID_OPERATION";
    case CL_INVALID_GL_OBJECT:
        return "CL_INVALID_GL_OBJECT";
    case CL_INVALID_BUFFER_SIZE:
        return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_MIP_LEVEL:
        return "CL_INVALID_MIP_LEVEL";
    case CL_INVALID_GLOBAL_WORK_SIZE:
        return "CL_INVALID_GLOBAL_WORK_SIZE";
    case CL_INVALID_PROPERTY:
        return "CL_INVALID_PROPERTY";

#ifdef CL_VERSION_1_2
    case CL_INVALID_IMAGE_DESCRIPTOR:
        return "CL_INVALID_IMAGE_DESCRIPTOR";
    case CL_INVALID_COMPILER_OPTIONS:
        return "CL_INVALID_COMPILER_OPTIONS";
    case CL_INVALID_LINKER_OPTIONS:
        return "CL_INVALID_LINKER_OPTIONS";
    case CL_INVALID_DEVICE_PARTITION_COUNT:
        return "CL_INVALID_DEVICE_PARTITION_COUNT";
#endif // CL_VERSION_1_2

#ifdef CL_VERSION_2_0
    case CL_INVALID_PIPE_SIZE:
        return "CL_INVALID_PIPE_SIZE";
    case CL_INVALID_DEVICE_QUEUE:
        return "CL_INVALID_DEVICE_QUEUE";
#endif // CL_VERSION_2_0

#ifdef CL_VERSION_2_2
    case CL_INVALID_SPEC_ID:
        return "CL_INVALID_SPEC_ID";
    case CL_MAX_SIZE_RESTRICTION_EXCEEDED:
        return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
#endif // CL_VERSION_2_2
    }

    return "Unknown CL error encountered";
}

/**
 * Prints cl::Errors in a uniform way
 * @param msg text prepending the error message
 * @param clerr cl:Error object
 *
 * Prints errors in the format:
 *      msg: what(), string err() (numeric err())
 */
static std::string XDagAOCLErrorHelper(const char *msg, cl::Error const &clerr)
{
    std::ostringstream osstream;
    osstream << msg << ": " << clerr.what() << ": " << strClError(clerr.err())
        << " (" << clerr.err() << ")";
    return osstream.str();
}

namespace {
	void AddDefinition(std::string& source, char const* id, unsigned value)
	{
		char buf[256] = { 0 };
		snprintf(buf, sizeof(buf), "#define %s %uu\n", id, value);
		source.insert(source.begin(), buf, buf + strlen(buf));
	}

	std::vector<cl::Platform> GetPlatforms()
	{
		std::vector<cl::Platform> platforms;
//		try
//		{
//			cl::Platform::get(&platforms);
//		}
//		catch (cl::Error const& err)
//		{
//#if defined(CL_PLATFORM_NOT_FOUND_KHR)
//			if (err.err() == CL_PLATFORM_NOT_FOUND_KHR)
//			{
//				cwarn << "No OpenCL platforms found";
//			}
//			else
//#endif
//				throw err;
//		}
		cl_platform_id platform = NULL;
		platform = findPlatform("Intel(R) FPGA");
		if (platform == NULL) {
			cwarn << "ERROR: Unable to find Intel FPGA OpenCL platform";
			throw cl::Error(CL_INVALID_PLATFORM, "No FPGA platforms found");
		}
		platforms.push_back(cl::Platform(platform));

		return platforms;
	}

	std::vector<cl::Device> GetDevices(std::vector<cl::Platform> const& platforms, unsigned platformId, bool useAllOpenCLCompatibleDevices)
	{
		std::vector<cl::Device> devices;
		size_t platform_num = std::min<size_t>(platformId, platforms.size() - 1);
		try
		{
			cl_device_type type = CL_DEVICE_TYPE_ALL; // useAllOpenCLCompatibleDevices
				//? CL_DEVICE_TYPE_ALL
				//: CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR;
			platforms[platform_num].getDevices(type, &devices);

#if defined (__APPLE) || defined (__MACOS__)
			auto iter = devices.begin();
			for (; iter != devices.end(); iter++) {
				size_t maxParam = 0;
				iter->getInfo(CL_DEVICE_MAX_PARAMETER_SIZE, &maxParam);
				if (maxParam <= 1024) { //Tricky fix, that Intel integrate graphic card is not supported on Mac OS due to parameter size.
					iter = devices.erase(iter);
					iter--;
				}
			}
#endif
		}
		catch (cl::Error const& err)
		{
			// if simply no devices found return empty vector
			if (err.err() != CL_DEVICE_NOT_FOUND)
			{
				throw err;
			}
		}
		return devices;
	}
}

unsigned AOCLMiner::_selectedPlatformId = 0;
uint32_t AOCLMiner::_numInstances = 0;
int AOCLMiner::_devices[MAX_CL_DEVICES] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };

AOCLMiner::AOCLMiner(uint32_t index, XTaskProcessor* taskProcessor)
    :Miner("cl-", index, taskProcessor)
{
    _kernelExecutionMcs = 0;
}

AOCLMiner::~AOCLMiner()
{
}

bool AOCLMiner::ConfigureGPU(
    uint32_t localWorkSize,
    uint32_t globalWorkSizeMultiplier,
    uint32_t platformId,
    bool useOpenClCpu
)
{
    //TODO: do I need automatically detemine path to executable folder?
    std::string path = PathUtils::GetModuleFolder();
    path += _clKernelName;
    if(!PathUtils::FileExists(path))
    {
        XCL_LOG("OpenCL kernel file is not found: " << path);
        return false;
    }

    _selectedPlatformId = platformId;
    _useOpenClCpu = useOpenClCpu;

    localWorkSize = ((localWorkSize + 7) / 8) * 8;
    _sWorkgroupSize = localWorkSize;
    _sInitialGlobalWorkSize = globalWorkSizeMultiplier * localWorkSize;

    std::vector<cl::Platform> platforms = GetPlatforms();
    if(platforms.empty())
    {
        XCL_LOG("No OpenCL platforms found.");
        return false;
    }
    if(_selectedPlatformId >= platforms.size())
    {
        return false;
    }
	cnote << "Found platform:" << platforms[0].getInfo<CL_PLATFORM_NAME>();
    std::vector<cl::Device> devices = GetDevices(platforms, _selectedPlatformId, _useOpenClCpu);
    if(devices.empty())
    {
        XCL_LOG("No OpenCL devices found...");
        return false;
    }
    cnote << "Found OpenCL devices:";
    for(auto const& device : devices)
    {
        cl_ulong result = 0;
        device.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &result);
        cl::string name = device.getInfo<CL_DEVICE_NAME>();
        boost::trim_right(name);
        cnote << name << " with " << result << " bytes of memory";
    }

    return true;
}

bool AOCLMiner::Initialize()
{
    // get all platforms
    try
    {
        if(!LoadKernelCode())
        {
            XCL_LOG("Cannot load OpenCL kernel file");
            return false;
        }

        std::vector<cl::Platform> platforms = GetPlatforms();
        if(platforms.empty())
        {
            XCL_LOG("No OpenCL platforms found.");
            return false;
        }

        // use selected platform
        unsigned platformIdx = std::min<unsigned>(_selectedPlatformId, (uint32_t)platforms.size() - 1);

        std::string platformName = platforms[platformIdx].getInfo<CL_PLATFORM_NAME>();
        XCL_LOG("Platform: " << platformName);

        _platformId = OPENCL_PLATFORM_UNKNOWN;
        {
            // this mutex prevents race conditions when calling the adl wrapper since it is apparently not thread safe
            static std::mutex mtx;
            std::lock_guard<std::mutex> lock(mtx);

            if(platformName == "NVIDIA CUDA")
            {
                _platformId = OPENCL_PLATFORM_NVIDIA;
                //nvmlh = wrap_nvml_create();
            }
            else if(platformName == "AMD Accelerated Parallel Processing")
            {
                _platformId = OPENCL_PLATFORM_AMD;
                //adlh = wrap_adl_create();
            }
            else if(platformName == "Clover")
            {
                _platformId = OPENCL_PLATFORM_CLOVER;
            }
            else if(platformName == "Apple")
            {
                _platformId = OPENCL_PLATFORM_APPLE;
            }
			else if (platformName == "Intel(R) FPGA SDK for OpenCL(TM)")
			{
				_platformId = OPENCL_PLATFORM_INTELFPGA;
			}
        }

        // get GPU device of the default platform
        std::vector<cl::Device> devices = GetDevices(platforms, platformIdx, _useOpenClCpu);
        if(devices.empty())
        {
            XCL_LOG("No OpenCL devices found.");
            return false;
        }

        // use selected device
        uint32_t deviceId = _devices[_index] > -1 ? _devices[_index] : _index;
        cl::Device& device = devices[std::min<uint32_t>(deviceId, (uint32_t)devices.size() - 1)];
        if(_useOpenClCpu)
        {
            device = devices[std::min<uint32_t>(deviceId, 0)];
        }
        std::string device_version = device.getInfo<CL_DEVICE_VERSION>();
        cl::string name = device.getInfo<CL_DEVICE_NAME>();
        boost::trim_right(name);
        XCL_LOG("Device: " << name << " / " << device_version);

        char options[256];
        int computeCapability = 0;
        if(_platformId == OPENCL_PLATFORM_NVIDIA)
        {
            cl_uint computeCapabilityMajor;
            cl_uint computeCapabilityMinor;
            clGetDeviceInfo(device(), CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV, sizeof(cl_uint), &computeCapabilityMajor, NULL);
            clGetDeviceInfo(device(), CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV, sizeof(cl_uint), &computeCapabilityMinor, NULL);

            computeCapability = computeCapabilityMajor * 10 + computeCapabilityMinor;
            int maxregs = computeCapability >= 35 ? 72 : 63;
            sprintf(options, "-cl-nv-maxrregcount=%d", maxregs);
        }
        else
        {
            sprintf(options, "%s", "");
        }

        char extensions[1024];
        clGetDeviceInfo(device(), CL_DEVICE_EXTENSIONS, 1024, extensions, NULL);
        bool hasBitAlign = strstr(extensions, "cl_amd_media_ops") != NULL;
        bool useBfiInt = false;
        if(hasBitAlign && !strstr(device_version.c_str(), "OpenCL 1.0") && !strstr(device_version.c_str(), "OpenCL 1.1"))
        {
            if(strstr(name.c_str(), "Cedar") ||
                strstr(name.c_str(), "Redwood") ||
                strstr(name.c_str(), "Juniper") ||
                strstr(name.c_str(), "Cypress") ||
                strstr(name.c_str(), "Hemlock") ||
                strstr(name.c_str(), "Caicos") ||
                strstr(name.c_str(), "Turks") ||
                strstr(name.c_str(), "Barts") ||
                strstr(name.c_str(), "Cayman") ||
                strstr(name.c_str(), "Antilles") ||
                strstr(name.c_str(), "Wrestler") ||
                strstr(name.c_str(), "Zacate") ||
                strstr(name.c_str(), "WinterPark"))
            {
                useBfiInt = true;
            }
        }

        // create context
        _context = cl::Context(std::vector<cl::Device>(&device, &device + 1));
        _queue = cl::CommandQueue(_context, device);
		_queue2 = cl::CommandQueue(_context, device);
		_queue3 = cl::CommandQueue(_context, device);

        //AddDefinition(_kernelCode, "PLATFORM", platformId);
        /*AddDefinition(_kernelCode, "OUTPUT_SIZE", OUTPUT_SIZE);
        AddDefinition(_kernelCode, "ITERATIONS_COUNT", KERNEL_ITERATIONS);
        if(hasBitAlign)
        {
            AddDefinition(_kernelCode, "BITALIGN", 1);
        }
        if(useBfiInt)
        {
            AddDefinition(_kernelCode, "BFI_INT", 1);
        }
        if(_useVectors)
        {
            AddDefinition(_kernelCode, "VECTORS", 1);
        }*/

        // create miner OpenCL program
        /*cl::Program::Sources sources { { _kernelCode.data(), _kernelCode.size() } };
        cl::Program program(_context, sources);*/
		cl::Program::Binaries binaries;
		binaries.push_back(_kernelBinary);
		cl::Program program(_context, std::vector<cl::Device>(&device, &device + 1), binaries);

        try
        {
            program.build({ device }, options);
            cllog << "Build info:" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        }
        catch(cl::Error const&err)
        {
            cwarn << "Build info:" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << err.err();
            return false;
        }

        _searchKernel = cl::Kernel(program, "search_nonce");
		_searchKernel2 = cl::Kernel(program, "sha_hash");
		_searchKernel3 = cl::Kernel(program, "nonce_output");
        
#if defined (__APPLE__) || defined (__MACOS__)
        size_t local;
        
        int err = clGetKernelWorkGroupInfo(_searchKernel.get(), device.get(), CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
        if (err != CL_SUCCESS) 
        {
            fprintf(stdout, "Error: Failed to retrieve kernel work group info! err: %d\n", err);
            return false;
        }
        
        _workgroupSize = std::min(_sWorkgroupSize,(uint32_t)local);
        _globalWorkSize = _sInitialGlobalWorkSize;
#else
        _workgroupSize = _sWorkgroupSize;
        _globalWorkSize = _sInitialGlobalWorkSize;
#endif
        // make sure that global work size is evenly divisible by the local workgroup size
        if(_globalWorkSize % _workgroupSize != 0)
        {
            _globalWorkSize = ((_globalWorkSize / _workgroupSize) + 1) * _workgroupSize;
        }

        // create buffer for initial hashing state
        XCL_LOG("Creating buffer for initial hashing state.");
        _stateBuffer = cl::Buffer(_context, CL_MEM_READ_ONLY, 32);

        // create buffer for precalculated hashing state
        XCL_LOG("Creating buffer for precalculated hashing state.");
        _precalcStateBuffer = cl::Buffer(_context, CL_MEM_READ_ONLY, 32);

        // create buffer for initial data
        XCL_LOG("Creating buffer for initial data.");
        _dataBuffer = cl::Buffer(_context, CL_MEM_READ_ONLY, 56);

        // create mining buffers
        XCL_LOG("Creating output buffer");
        _searchBuffer = cl::Buffer(_context, CL_MEM_WRITE_ONLY, (OUTPUT_SIZE + 1) * sizeof(uint64_t));
    }
    catch(cl::Error const& err)
    {
        cwarn << XDagAOCLErrorHelper("FPGA OpenCL init failed", err);
        return false;
    }
    return true;
}

bool AOCLMiner::Reset()
{
    cwarn << "GPU will be restarted";

    // pause for 0.5 sec
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    try
    {
        _stateBuffer = cl::Buffer();
        _precalcStateBuffer = cl::Buffer();
        _dataBuffer = cl::Buffer();
        _searchBuffer = cl::Buffer();
        _searchKernel = cl::Kernel();
        _queue = cl::CommandQueue();

		_searchKernel2 = cl::Kernel();
		_queue2 = cl::CommandQueue();

		_searchKernel3 = cl::Kernel();
		_queue3 = cl::CommandQueue();

        _context = cl::Context();

        return Initialize();
    }
    catch(...)
    {
    }
    return false;
}

void AOCLMiner::WorkLoop()
{
    int errorCount = 0;
    while(errorCount < MAX_GPU_ERROR_COUNT)
    {
        try
        {
            InternalWorkLook(errorCount);
            break;
        }
        catch(cl::Error const& _e)
        {
            cwarn << XDagAOCLErrorHelper("FPGA OpenCL Error", _e);
            if(++errorCount < MAX_GPU_ERROR_COUNT)
            {                
                if(!Reset())
                {
                    break;
                }               
            }  
        }
    }
}

void AOCLMiner::InternalWorkLook(int& errorCount)
{
    xdag_field last;
    uint64_t prevTaskIndex = 0;
    uint64_t nonce = 0;
    uint32_t loopCounter = 0;

    uint64_t results[OUTPUT_SIZE + 1];
    uint64_t zeroBuffer[OUTPUT_SIZE + 1];
    memset(zeroBuffer, 0, (OUTPUT_SIZE + 1) * sizeof(uint64_t));

    while(!ShouldStop())
    {
        XTaskWrapper* taskWrapper = GetTask();
        if(taskWrapper == NULL)
        {
            clog(LogChannel) << "No work. Pause for 3 s.";
            std::this_thread::sleep_for(std::chrono::seconds(3));
            continue;
        }

        if(taskWrapper->GetIndex() != prevTaskIndex)
        {
            //new task came, we have to finish current task and reload all data
            if(prevTaskIndex > 0)
            {
                _queue.finish();
				_queue2.finish();			
				_queue3.finish();
			}

            prevTaskIndex = taskWrapper->GetIndex();
            loopCounter = 0;
            memcpy(last.data, taskWrapper->GetTask()->nonce.data, sizeof(xdag_hash_t));
            nonce = last.amount + _index * 1000000000000;//TODO: think of nonce increment

            WriteKernelArgs(taskWrapper, zeroBuffer);
        }

        bool hasSolution = false;
        if(loopCounter > 0)
        {
            // Read results.
            ReadData(results);
            errorCount = 0;

            //miner return an array with 16 64-bit values. If nonce for hash lower than target hash is found - it is written to array. 
            //the first value in array contains count of found solutions
            hasSolution = results[0] > 0;
            if(hasSolution)
            {
                // Reset search buffer if any solution found.
                _queue3.enqueueWriteBuffer(_searchBuffer, CL_FALSE, 0, sizeof(zeroBuffer), zeroBuffer);
            }
        }

        // Run the kernel.
        _searchKernel.setArg(KERNEL_ARG_NONCE, nonce);
		_searchKernel3.setArg(KERNEL_ARG_NONCE, nonce);
        // _queue.enqueueNDRangeKernel(_searchKernel, cl::NullRange, _globalWorkSize, _workgroupSize);
		// _queue.enqueueNDRangeKernel(_searchKernel, cl::NullRange, 8192*8, 8);
		_queue.enqueueTask(_searchKernel);
		_queue2.enqueueTask(_searchKernel2);
		_queue3.enqueueTask(_searchKernel3);
        // Report results while the kernel is running.
        // It takes some time because hashes must be re-evaluated on CPU.
        if(hasSolution)
        {
            //we need to recalculate hashes for all founded nonces and choose the minimal one
            SetMinShare(taskWrapper, results, last);
#ifdef _DEBUG
            std::cout << HashToHexString(taskWrapper->GetTask()->minhash.data) << std::endl;
#endif
            //new minimal hash is written as target hash for GPU
            _searchKernel3.setArg(1, ((uint32_t*)taskWrapper->GetTask()->minhash.data)[7]);
            _searchKernel3.setArg(2, ((uint32_t*)taskWrapper->GetTask()->minhash.data)[6]);
        }

		// uint32_t hashesProcessed = _globalWorkSize * KERNEL_ITERATIONS;
		/* if(_useVectors)
        {
            hashesProcessed <<= 1;
        }*/
		uint32_t hashesProcessed = _globalWorkSize*4;
        // Increase start nonce for following kernel execution.
        nonce += hashesProcessed;
        // Report hash count
        AddHashCount(hashesProcessed);
        ++loopCounter;
    }

    // Make sure the last buffer write has finished --
    // it reads local variable.
    _queue.finish();
	_queue2.finish();
	_queue3.finish();
}

uint32_t AOCLMiner::GetNumDevices()
{
    std::vector<cl::Platform> platforms = GetPlatforms();
    if(platforms.empty())
    {
        return 0;
    }

    std::vector<cl::Device> devices = GetDevices(platforms, _selectedPlatformId, _useOpenClCpu);
    if(devices.empty())
    {
        cwarn << "No OpenCL devices found.";
        return 0;
    }
    return (uint32_t)devices.size();
}

void AOCLMiner::ListDevices(bool useOpenClCpu)
{
    std::string outString = "\nListing OpenCL devices.\nFORMAT: [platformID] [deviceID] deviceName\n";
    uint32_t i = 0;

    std::vector<cl::Platform> platforms = GetPlatforms();
    if(platforms.empty())
    {
        return;
    }
    for(uint32_t j = 0; j < platforms.size(); ++j)
    {
        i = 0;
        std::vector<cl::Device> devices = GetDevices(platforms, j, useOpenClCpu);
        for(auto const& device : devices)
        {
            outString += "[" + std::to_string(j) + "] [" + std::to_string(i) + "] " + device.getInfo<CL_DEVICE_NAME>() + "\n";
            outString += "\tCL_DEVICE_TYPE: ";
            switch(device.getInfo<CL_DEVICE_TYPE>())
            {
            case CL_DEVICE_TYPE_CPU:
                outString += "CPU\n";
                break;
            case CL_DEVICE_TYPE_GPU:
                outString += "GPU\n";
                break;
            case CL_DEVICE_TYPE_ACCELERATOR:
                outString += "ACCELERATOR\n";
                break;
            default:
                outString += "DEFAULT\n";
                break;
            }
            outString += "\tCL_DEVICE_GLOBAL_MEM_SIZE: " + std::to_string(device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()) + "\n";
            outString += "\tCL_DEVICE_MAX_MEM_ALLOC_SIZE: " + std::to_string(device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>()) + "\n";
            outString += "\tCL_DEVICE_MAX_WORK_GROUP_SIZE: " + std::to_string(device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()) + "\n";
            ++i;
        }
    }
    std::cout << outString;
}

HwMonitor AOCLMiner::Hwmon()
{
    HwMonitor hw;
    unsigned int tempC = 0, fanpcnt = 0;
    hw.tempC = tempC;
    hw.fanP = fanpcnt;
    return hw;
}

/* loads the kernel file into a string */
bool AOCLMiner::LoadKernelCode()
{
    std::string path = PathUtils::GetModuleFolder();
    path += _clKernelName;
    if(!PathUtils::FileExists(path))
    {
        return false;
    }

	try {
		std::ifstream inputFile(path, std::ios::binary);
		cl::vector<unsigned char> fileData((std::istreambuf_iterator<char>(inputFile)),
			std::istreambuf_iterator<char>());
		_kernelBinary = fileData;
		cllog << "OpenCL kernel binary: Custom '" + _clKernelName + "'";
		return true;

	} catch(...)
	{
		cwarn << "FPGA OpenCL binary read failed";
		return false;
	}

    return false;
}

void AOCLMiner::SetMinShare(XTaskWrapper* taskWrapper, uint64_t* searchBuffer, xdag_field& last)
{
    xdag_hash_t minHash;
    xdag_hash_t currentHash;
    uint64_t minNonce = 0;

    uint32_t size = searchBuffer[0] < OUTPUT_SIZE ? (uint32_t)searchBuffer[0] : OUTPUT_SIZE;
    for(uint32_t i = 1; i <= size; ++i)
    {
        uint64_t nonce = searchBuffer[i];
        if(nonce == 0)
        {
            continue;
        }
        shamod::shasha(taskWrapper->GetTask()->ctx.state, taskWrapper->GetTask()->ctx.data, nonce, (uint8_t*)currentHash);
        if(!minNonce || XHash::CompareHashes(currentHash, minHash) < 0)
        {
            memcpy(minHash, currentHash, sizeof(xdag_hash_t));
            minNonce = nonce;
        }
    }

#ifdef _DEBUG
    assert(minNonce > 0);
#endif
    if(minNonce > 0)
    {
        last.amount = minNonce;
        taskWrapper->SetShare(last.data, minHash);
    }
}

void AOCLMiner::WriteKernelArgs(XTaskWrapper* taskWrapper, uint64_t* zeroBuffer)
{
    // Update constant buffers.
    _queue2.enqueueWriteBuffer(_stateBuffer, CL_FALSE, 0, 32, taskWrapper->GetTask()->ctx.state);
    _queue.enqueueWriteBuffer(_precalcStateBuffer, CL_FALSE, 0, 32, taskWrapper->GetPrecalcState());
    _queue.enqueueWriteBuffer(_dataBuffer, CL_FALSE, 0, 56, taskWrapper->GetReversedData());
    _queue3.enqueueWriteBuffer(_searchBuffer, CL_FALSE, 0, sizeof(zeroBuffer), zeroBuffer);

    _searchKernel2.setArg(0, _stateBuffer);
    _searchKernel.setArg(1, _precalcStateBuffer);
    _searchKernel.setArg(2, _dataBuffer);

    //it makes no sense to write all 32 bytes of target hash to GPU memory 
    //we can pass only the first 8 bytes
    _searchKernel3.setArg(1, ((uint32_t*)taskWrapper->GetTask()->minhash.data)[7]);
    _searchKernel3.setArg(2, ((uint32_t*)taskWrapper->GetTask()->minhash.data)[6]);
    _searchKernel3.setArg(3, _searchBuffer); // Supply output buffer to kernel
}

void AOCLMiner::ReadData(uint64_t* results)
{
    if(_platformId != OPENCL_PLATFORM_NVIDIA || !_useNvidiaFix)
    {
        _queue3.enqueueReadBuffer(_searchBuffer, CL_TRUE, 0, (OUTPUT_SIZE + 1) * sizeof(uint64_t), results);
    }
    else
    {
        _queue.flush();
		_queue2.flush();
		_queue3.flush();

        //during executing the opencl program nvidia opencl library enters loop which checks if the execution of opencl program has ended
        //so, current thread just spins in this loop, eating CPU for nothing.
        //workaround for the problem: add sleep for some calculated time after the kernel was queued and flushed
        if(_kernelExecutionMcs > 0)
        {
            std::this_thread::sleep_for(std::chrono::microseconds(_kernelExecutionMcs));
        }
        auto startTime = std::chrono::high_resolution_clock::now();
        _queue3.enqueueReadBuffer(_searchBuffer, CL_TRUE, 0, (OUTPUT_SIZE + 1) * sizeof(uint64_t), results);
        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        _kernelExecutionMcs = (uint32_t)((_kernelExecutionMcs + duration.count()) * _nvidiaSpinDamp);
    }
}
