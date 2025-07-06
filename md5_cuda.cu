#include "md5_cuda.h"
#include <iostream>
#include <cstring>
#include <iomanip>

// CUDA设备常量
__constant__ bit32 d_md5_constants[64] = {
    // Round 1
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
    0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
    0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    // Round 2
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
    0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
    0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    // Round 3
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
    0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
    0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    // Round 4
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
    0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
    0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391
};

// CUDA设备函数：MD5基本运算
__device__ __forceinline__ bit32 cuda_F(bit32 x, bit32 y, bit32 z) {
    return (x & y) | (~x & z);
}

__device__ __forceinline__ bit32 cuda_G(bit32 x, bit32 y, bit32 z) {
    return (x & z) | (y & ~z);
}

__device__ __forceinline__ bit32 cuda_H(bit32 x, bit32 y, bit32 z) {
    return x ^ y ^ z;
}

__device__ __forceinline__ bit32 cuda_I(bit32 x, bit32 y, bit32 z) {
    return y ^ (x | ~z);
}

__device__ __forceinline__ bit32 cuda_rotleft(bit32 value, int shift) {
    return (value << shift) | (value >> (32 - shift));
}

// CUDA设备函数：MD5轮函数
__device__ __forceinline__ void cuda_FF(bit32& a, bit32 b, bit32 c, bit32 d, bit32 x, int s, bit32 ac) {
    a += cuda_F(b, c, d) + x + ac;
    a = cuda_rotleft(a, s);
    a += b;
}

__device__ __forceinline__ void cuda_GG(bit32& a, bit32 b, bit32 c, bit32 d, bit32 x, int s, bit32 ac) {
    a += cuda_G(b, c, d) + x + ac;
    a = cuda_rotleft(a, s);
    a += b;
}

__device__ __forceinline__ void cuda_HH(bit32& a, bit32 b, bit32 c, bit32 d, bit32 x, int s, bit32 ac) {
    a += cuda_H(b, c, d) + x + ac;
    a = cuda_rotleft(a, s);
    a += b;
}

__device__ __forceinline__ void cuda_II(bit32& a, bit32 b, bit32 c, bit32 d, bit32 x, int s, bit32 ac) {
    a += cuda_I(b, c, d) + x + ac;
    a = cuda_rotleft(a, s);
    a += b;
}

// CUDA设备函数：字符串预处理
__device__ void cuda_string_process(const char* input, int length, bit32* padded_blocks, int* num_blocks) {
    int bit_length = length * 8;
    int padding_bits = bit_length % 512;
    
    if (padding_bits > 448) {
        padding_bits = 512 - (padding_bits - 448);
    } else if (padding_bits < 448) {
        padding_bits = 448 - padding_bits;
    } else {
        padding_bits = 512;
    }
    
    int padding_bytes = padding_bits / 8;
    int padded_length = length + padding_bytes + 8;
    *num_blocks = padded_length / 64;
    
    // 清零填充块
    for (int i = 0; i < *num_blocks * 16; i++) {
        padded_blocks[i] = 0;
    }
    
    // 复制原始数据
    for (int i = 0; i < length; i++) {
        int block_idx = i / 4;
        int byte_idx = i % 4;
        padded_blocks[block_idx] |= ((bit32)(unsigned char)input[i]) << (byte_idx * 8);
    }
    
    // 添加填充位
    int pad_start = length;
    int block_idx = pad_start / 4;
    int byte_idx = pad_start % 4;
    padded_blocks[block_idx] |= 0x80 << (byte_idx * 8);
    
    // 添加长度信息
    int length_pos = *num_blocks * 16 - 2;
    padded_blocks[length_pos] = bit_length;
    padded_blocks[length_pos + 1] = bit_length >> 32;
}

// CUDA设备函数：MD5块处理
__device__ void cuda_md5_process_block(bit32* state, const bit32* block) {
    bit32 a = state[0], b = state[1], c = state[2], d = state[3];
    
    // Round 1
    cuda_FF(a, b, c, d, block[0], 7, d_md5_constants[0]);
    cuda_FF(d, a, b, c, block[1], 12, d_md5_constants[1]);
    cuda_FF(c, d, a, b, block[2], 17, d_md5_constants[2]);
    cuda_FF(b, c, d, a, block[3], 22, d_md5_constants[3]);
    cuda_FF(a, b, c, d, block[4], 7, d_md5_constants[4]);
    cuda_FF(d, a, b, c, block[5], 12, d_md5_constants[5]);
    cuda_FF(c, d, a, b, block[6], 17, d_md5_constants[6]);
    cuda_FF(b, c, d, a, block[7], 22, d_md5_constants[7]);
    cuda_FF(a, b, c, d, block[8], 7, d_md5_constants[8]);
    cuda_FF(d, a, b, c, block[9], 12, d_md5_constants[9]);
    cuda_FF(c, d, a, b, block[10], 17, d_md5_constants[10]);
    cuda_FF(b, c, d, a, block[11], 22, d_md5_constants[11]);
    cuda_FF(a, b, c, d, block[12], 7, d_md5_constants[12]);
    cuda_FF(d, a, b, c, block[13], 12, d_md5_constants[13]);
    cuda_FF(c, d, a, b, block[14], 17, d_md5_constants[14]);
    cuda_FF(b, c, d, a, block[15], 22, d_md5_constants[15]);
    
    // Round 2
    cuda_GG(a, b, c, d, block[1], 5, d_md5_constants[16]);
    cuda_GG(d, a, b, c, block[6], 9, d_md5_constants[17]);
    cuda_GG(c, d, a, b, block[11], 14, d_md5_constants[18]);
    cuda_GG(b, c, d, a, block[0], 20, d_md5_constants[19]);
    cuda_GG(a, b, c, d, block[5], 5, d_md5_constants[20]);
    cuda_GG(d, a, b, c, block[10], 9, d_md5_constants[21]);
    cuda_GG(c, d, a, b, block[15], 14, d_md5_constants[22]);
    cuda_GG(b, c, d, a, block[4], 20, d_md5_constants[23]);
    cuda_GG(a, b, c, d, block[9], 5, d_md5_constants[24]);
    cuda_GG(d, a, b, c, block[14], 9, d_md5_constants[25]);
    cuda_GG(c, d, a, b, block[3], 14, d_md5_constants[26]);
    cuda_GG(b, c, d, a, block[8], 20, d_md5_constants[27]);
    cuda_GG(a, b, c, d, block[13], 5, d_md5_constants[28]);
    cuda_GG(d, a, b, c, block[2], 9, d_md5_constants[29]);
    cuda_GG(c, d, a, b, block[7], 14, d_md5_constants[30]);
    cuda_GG(b, c, d, a, block[12], 20, d_md5_constants[31]);
    
    // Round 3
    cuda_HH(a, b, c, d, block[5], 4, d_md5_constants[32]);
    cuda_HH(d, a, b, c, block[8], 11, d_md5_constants[33]);
    cuda_HH(c, d, a, b, block[11], 16, d_md5_constants[34]);
    cuda_HH(b, c, d, a, block[14], 23, d_md5_constants[35]);
    cuda_HH(a, b, c, d, block[1], 4, d_md5_constants[36]);
    cuda_HH(d, a, b, c, block[4], 11, d_md5_constants[37]);
    cuda_HH(c, d, a, b, block[7], 16, d_md5_constants[38]);
    cuda_HH(b, c, d, a, block[10], 23, d_md5_constants[39]);
    cuda_HH(a, b, c, d, block[13], 4, d_md5_constants[40]);
    cuda_HH(d, a, b, c, block[0], 11, d_md5_constants[41]);
    cuda_HH(c, d, a, b, block[3], 16, d_md5_constants[42]);
    cuda_HH(b, c, d, a, block[6], 23, d_md5_constants[43]);
    cuda_HH(a, b, c, d, block[9], 4, d_md5_constants[44]);
    cuda_HH(d, a, b, c, block[12], 11, d_md5_constants[45]);
    cuda_HH(c, d, a, b, block[15], 16, d_md5_constants[46]);
    cuda_HH(b, c, d, a, block[2], 23, d_md5_constants[47]);
    
    // Round 4
    cuda_II(a, b, c, d, block[0], 6, d_md5_constants[48]);
    cuda_II(d, a, b, c, block[7], 10, d_md5_constants[49]);
    cuda_II(c, d, a, b, block[14], 15, d_md5_constants[50]);
    cuda_II(b, c, d, a, block[5], 21, d_md5_constants[51]);
    cuda_II(a, b, c, d, block[12], 6, d_md5_constants[52]);
    cuda_II(d, a, b, c, block[3], 10, d_md5_constants[53]);
    cuda_II(c, d, a, b, block[10], 15, d_md5_constants[54]);
    cuda_II(b, c, d, a, block[1], 21, d_md5_constants[55]);
    cuda_II(a, b, c, d, block[8], 6, d_md5_constants[56]);
    cuda_II(d, a, b, c, block[15], 10, d_md5_constants[57]);
    cuda_II(c, d, a, b, block[6], 15, d_md5_constants[58]);
    cuda_II(b, c, d, a, block[13], 21, d_md5_constants[59]);
    cuda_II(a, b, c, d, block[4], 6, d_md5_constants[60]);
    cuda_II(d, a, b, c, block[11], 10, d_md5_constants[61]);
    cuda_II(c, d, a, b, block[2], 15, d_md5_constants[62]);
    cuda_II(b, c, d, a, block[9], 21, d_md5_constants[63]);
    
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
}

// CUDA核函数：并行MD5计算
__global__ void cuda_md5_kernel(char* d_inputs, int* d_lengths, bit32* d_results, int num_inputs, int max_length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_inputs) return;
    
    // 获取当前线程处理的字符串
    char* current_input = d_inputs + idx * max_length;
    int current_length = d_lengths[idx];
    
    // 初始化MD5状态
    bit32 state[4] = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476};
    
    // 预处理字符串
    bit32 padded_blocks[32]; // 假设最大2个块
    int num_blocks;
    cuda_string_process(current_input, current_length, padded_blocks, &num_blocks);
    
    // 处理每个块
    for (int i = 0; i < num_blocks; i++) {
        cuda_md5_process_block(state, padded_blocks + i * 16);
    }
    
    // 字节序转换并存储结果
    bit32* result = d_results + idx * 4;
    for (int i = 0; i < 4; i++) {
        bit32 value = state[i];
        result[i] = ((value & 0xff) << 24) |
                   ((value & 0xff00) << 8) |
                   ((value & 0xff0000) >> 8) |
                   ((value & 0xff000000) >> 24);
    }
}

// 主机函数：CUDA MD5哈希
void MD5Hash_CUDA(const std::vector<std::string>& inputs, std::vector<std::vector<bit32>>& results) {
    int num_inputs = inputs.size();
    if (num_inputs == 0) return;
    
    // 找到最大字符串长度
    int max_length = 0;
    for (const auto& input : inputs) {
        max_length = std::max(max_length, (int)input.length());
    }
    max_length = ((max_length + 3) / 4) * 4; // 4字节对齐
    
    // 分配主机内存
    char* h_inputs = new char[num_inputs * max_length];
    int* h_lengths = new int[num_inputs];
    bit32* h_results = new bit32[num_inputs * 4];
    
    // 准备输入数据
    memset(h_inputs, 0, num_inputs * max_length);
    for (int i = 0; i < num_inputs; i++) {
        strcpy(h_inputs + i * max_length, inputs[i].c_str());
        h_lengths[i] = inputs[i].length();
    }
    
    // 分配设备内存
    char* d_inputs;
    int* d_lengths;
    bit32* d_results;
    
    cudaMalloc(&d_inputs, num_inputs * max_length);
    cudaMalloc(&d_lengths, num_inputs * sizeof(int));
    cudaMalloc(&d_results, num_inputs * 4 * sizeof(bit32));
    
    // 复制数据到设备
    cudaMemcpy(d_inputs, h_inputs, num_inputs * max_length, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths, h_lengths, num_inputs * sizeof(int), cudaMemcpyHostToDevice);
    
    // 配置CUDA执行参数
    int block_size = CUDA_BLOCK_SIZE;
    int grid_size = (num_inputs + block_size - 1) / block_size;
    
    // 启动CUDA核函数
    cuda_md5_kernel<<<grid_size, block_size>>>(d_inputs, d_lengths, d_results, num_inputs, max_length);
    
    // 等待GPU完成
    cudaDeviceSynchronize();
    
    // 复制结果回主机
    cudaMemcpy(h_results, d_results, num_inputs * 4 * sizeof(bit32), cudaMemcpyDeviceToHost);
    
    // 整理结果
    results.resize(num_inputs);
    for (int i = 0; i < num_inputs; i++) {
        results[i].resize(4);
        for (int j = 0; j < 4; j++) {
            results[i][j] = h_results[i * 4 + j];
        }
    }
    
    // 释放内存
    delete[] h_inputs;
    delete[] h_lengths;
    delete[] h_results;
    cudaFree(d_inputs);
    cudaFree(d_lengths);
    cudaFree(d_results);
}