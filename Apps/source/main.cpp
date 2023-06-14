/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2016 Andrey Alekseenko, Kentaro Wada
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 * 
 * */

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cuda_runtime_api.h>
#include <sstream>
#include "cuda_runtime.h"
#include "cuda.h"
#include <iostream>
#define CUDA_CALL(function, ...)  { \
    cudaError_t status = function(__VA_ARGS__); \
    anyCheck(status == cudaSuccess, cudaGetErrorString(status), #function, __FILE__, __LINE__); \
}

void anyCheck(bool is_ok, const char *description, const char *function, const char *file, int line) {
    if (!is_ok) {
        fprintf(stderr,"Error: %s in %s at %s:%d\n", description, function, file, line);
        exit(EXIT_FAILURE);
    }
}

void getMemoryUsageCUDA(int deviceId, size_t &memUsed, size_t &memTotal) {
    size_t memFree;
    CUDA_CALL(cudaSetDevice, deviceId);
    CUDA_CALL(cudaMemGetInfo, &memFree, &memTotal);
    memUsed = memTotal - memFree;
    memUsed = memUsed / 1024 / 1024;
    memTotal = memTotal / 1024 / 1024;
}

int main() {
    int cudaDeviceCount;
    struct cudaDeviceProp deviceProp;
    size_t memUsed, memTotal;
    CUDA_CALL(cudaGetDeviceCount, &cudaDeviceCount);

    for (int deviceId = 0; deviceId < cudaDeviceCount; ++deviceId) {
        CUDA_CALL(cudaGetDeviceProperties, &deviceProp, deviceId);
        printf("Device %2d", deviceId);
        printf(" [PCIe %04x:%02x:%02x.0]", deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);
        printf(": %20s (CC %d.%d)", deviceProp.name, deviceProp.major, deviceProp.minor);

        getMemoryUsageCUDA(deviceId, memUsed, memTotal);
        printf(": %5zu of %5zu MiB Used", memUsed, memTotal);

        // 获取 GPU 温度
        int temperature;
        char buff[512];
        // 独立显卡
        cudaDeviceGetPCIBusId(buff, 100, deviceId);
        std::string bus_id(buff);
        std::string command =
            "nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader -i " +
            std::to_string(deviceId);
        std::ostringstream output_stream;
        FILE*              pipe = _popen(command.c_str(), "r");
        while (!std::feof(pipe)) {
            if (std::fgets(buff, sizeof(buff), pipe) != NULL) {
                output_stream << buff;
            }
        }
        std::string        output = output_stream.str();
        std::istringstream iss(output);
        iss >> temperature;
        std::cout << "GPU Temperature: " << temperature << " degrees Celsius" << std::endl;
        printf("\n");
    }
    return 0;
}

