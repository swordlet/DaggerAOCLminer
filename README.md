# DaggerAOCLMiner
Altera/Intel OPENCL(FPGA Cyclone V) PCI-E Card miner for XDAG (Dagger coin)

**Hardware:**

Terasic C5P(OpenVINO Starter Kit)

https://www.terasic.com.tw/cgi-bin/page/archive.pl?Language=English&CategoryNo=167&No=1159&PartNo=1


This miner does not require files wallet.dat and dnet_key.dat. Storage folder is still necessary.

## Hashrate

140Mh

about 15 coins per day

:(

**Launch parameters:**

	1) FPGA benchmark: DaggerAOCLMinerDaggerAOCLMiner.exe -G -M  
	
	2) FPGA mining: DaggerAOCLMiner.exe -G -a <WALLET_ADDRESS> -p <POOL_ADDRESS>  
	
	3) CPU mining: DaggerAOCLMiner.exe -cpu -a <WALLET_ADDRESS> -p <POOL_ADDRESS> -t 8  
	

	
**The project ONLY supports Windows now.** 

## How to build

## FPGA: 

Intel FPGA Quartus Prime Standard Edition 17.1

Intel(R) FPGA SDK for OpenCL 17.1

C5P(OpenVINO Starter Kit) BSP(Board Support Package) for Intel SDK OpenCL 17.1

https://www.terasic.com.tw/cgi-bin/page/archive.pl?Language=English&CategoryNo=167&No=1159&PartNo=4

OPENCL source code in GpuMiner/CL/AOCLMminer_kernel.cl

Put the compiled AOCLMminer_kernel.aocx file into x64 folder with the executable host application.

## Windows:  

The project has 3 dependencies: FPGA SDK for OpenCL, Boost and OpenSSL 

Boost and OpenSSL libraries are included by Nuget Manager and should be downloaded automatically.

Intel(R) FPGA SDK for OpenCL 17.1


