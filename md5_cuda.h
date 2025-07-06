#ifndef MD5_CUDA_H
#define MD5_CUDA_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <string>
#include <vector>
#include "md5.h"

// CUDA MD5 函数声明
void MD5Hash_CUDA(const std::vector<std::string>& inputs, std::vector<std::vector<bit32>>& results);

// CUDA设备函数声明
__device__ void cuda_md5_process_block(bit32* state, const bit32* block);
__device__ void cuda_string_process(const char* input, int length, bit32* padded_blocks, int* num_blocks);

// CUDA常量
#define CUDA_BLOCK_SIZE 256
#define MAX_PASSWORD_LENGTH 64
#define MAX_PADDED_LENGTH 128

#endif