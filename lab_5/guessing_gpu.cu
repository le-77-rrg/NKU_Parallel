#include "PCFG.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <string>
#include <iostream>
#include <cstring>

using namespace std;

// ==================== 常量定义区域 ====================
// GPU加速处理的数据量阈值
#define GPU_ACCELERATION_THRESHOLD 1000

// CUDA运行时错误检测宏定义
#define CHECK_CUDA_ERROR(operation) do { \
    cudaError_t cuda_result = operation; \
    if (cuda_result != cudaSuccess) { \
        fprintf(stderr, "CUDA运行错误 %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(cuda_result)); \
        exit(1); \
    } \
} while(0)

// ==================== 前置声明区域 ====================
class CudaComputeManager;
static CudaComputeManager* global_cuda_manager = nullptr;

// ==================== 核函数定义区域 ====================
// CUDA核心函数：字符串拼接处理
__global__ void string_concatenation_kernel(const char* input_data, const int* data_offsets, const int* data_lengths,
                                          const char* prefix_data, int prefix_length,
                                          char* output_buffer, int max_string_length, int total_items) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= total_items) return;

    char* current_output = output_buffer + thread_id * max_string_length;
    int current_position = 0;

    // 拷贝前缀字符串
    if (prefix_data && prefix_length > 0) {
        for (int i = 0; i < prefix_length && current_position < max_string_length - 1; i++) {
            current_output[current_position++] = prefix_data[i];
        }
    }

    // 拷贝主要数据
    const char* source_data = input_data + data_offsets[thread_id];
    int source_length = data_lengths[thread_id];
    for (int i = 0; i < source_length && current_position < max_string_length - 1; i++) {
        current_output[current_position++] = source_data[i];
    }

    current_output[current_position] = '\0';
}

// ==================== 工具函数区域 ====================
// GPU计算环境初始化
void initialize_cuda_environment();
void release_cuda_resources();
bool processWithGPU(const vector<string>& values, const string& prefix, vector<string>& results);

// ==================== 核心类定义区域 ====================
// GPU计算资源管理器
class CudaComputeManager {
private:
    // 内存容量限制常量
    static const size_t MAXIMUM_ITEM_COUNT = 120000;
    static const size_t MAXIMUM_INPUT_CAPACITY = 60 * 1024 * 1024;  // 50MB
    static const size_t MAXIMUM_OUTPUT_CAPACITY = 120 * 1024 * 1024; // 100MB

    // CUDA设备内存指针
    char* cuda_input_buffer = nullptr;
    char* cuda_prefix_buffer = nullptr;
    char* cuda_result_buffer = nullptr;
    int* cuda_offset_array = nullptr;
    int* cuda_length_array = nullptr;
    cudaStream_t cuda_stream;

public:
    CudaComputeManager() {
        // 分配GPU内存空间
        CHECK_CUDA_ERROR(cudaMalloc(&cuda_input_buffer, MAXIMUM_INPUT_CAPACITY));
        CHECK_CUDA_ERROR(cudaMalloc(&cuda_prefix_buffer, 1024)); // 前缀缓冲区1KB
        CHECK_CUDA_ERROR(cudaMalloc(&cuda_result_buffer, MAXIMUM_OUTPUT_CAPACITY));
        CHECK_CUDA_ERROR(cudaMalloc(&cuda_offset_array, MAXIMUM_ITEM_COUNT * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&cuda_length_array, MAXIMUM_ITEM_COUNT * sizeof(int)));
        CHECK_CUDA_ERROR(cudaStreamCreate(&cuda_stream));
    }

    ~CudaComputeManager() {
        // 释放GPU内存资源
        cudaFree(cuda_input_buffer);
        cudaFree(cuda_prefix_buffer);
        cudaFree(cuda_result_buffer);
        cudaFree(cuda_offset_array);
        cudaFree(cuda_length_array);
        cudaStreamDestroy(cuda_stream);
    }

    // 批量处理字符串拼接任务
    bool process_string_batch(const vector<string>& input_strings, const string& prefix_string, vector<string>& output_results) {
        size_t string_count = input_strings.size();
        if (string_count == 0 || string_count > MAXIMUM_ITEM_COUNT) return false;

        // 计算输入数据总大小
        size_t total_input_bytes = 0;
        for(const auto& str : input_strings) {
            total_input_bytes += str.length();
        }
        if (total_input_bytes > MAXIMUM_INPUT_CAPACITY) return false;

        char* host_input_data = (char*)malloc(total_input_bytes);
        vector<int> host_offset_data(string_count);
        vector<int> host_length_data(string_count);
        if (!host_input_data) return false;

        size_t byte_position = 0;
        for(size_t i = 0; i < string_count; ++i) {
            host_offset_data[i] = byte_position;
            host_length_data[i] = input_strings[i].length();
            memcpy(host_input_data + byte_position, input_strings[i].c_str(), host_length_data[i]);
            byte_position += host_length_data[i];
        }

        const int MAXIMUM_STRING_LENGTH = 64;
        size_t result_buffer_size = string_count * MAXIMUM_STRING_LENGTH;
        if (result_buffer_size > MAXIMUM_OUTPUT_CAPACITY) {
            free(host_input_data);
            return false;
        }

        char* host_result_data = (char*)malloc(result_buffer_size);
        if (!host_result_data) {
            free(host_input_data);
            return false;
        }

        // 异步数据传输：主机到设备
        CHECK_CUDA_ERROR(cudaMemcpyAsync(cuda_input_buffer, host_input_data, total_input_bytes, cudaMemcpyHostToDevice, cuda_stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(cuda_offset_array, host_offset_data.data(), string_count * sizeof(int), cudaMemcpyHostToDevice, cuda_stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(cuda_length_array, host_length_data.data(), string_count * sizeof(int), cudaMemcpyHostToDevice, cuda_stream));

        char* device_prefix_pointer = nullptr;
        if (!prefix_string.empty()) {
            CHECK_CUDA_ERROR(cudaMemcpyAsync(cuda_prefix_buffer, prefix_string.c_str(), prefix_string.length(), cudaMemcpyHostToDevice, cuda_stream));
            device_prefix_pointer = cuda_prefix_buffer;
        }

        // 启动CUDA核函数
        const int threads_per_block = 512;
        int blocks_per_grid = (string_count + threads_per_block - 1) / threads_per_block;
        string_concatenation_kernel<<<blocks_per_grid, threads_per_block, 0, cuda_stream>>>(
            cuda_input_buffer, cuda_offset_array, cuda_length_array, device_prefix_pointer, prefix_string.length(),
            cuda_result_buffer, MAXIMUM_STRING_LENGTH, string_count);

        // 异步数据传输：设备到主机
        CHECK_CUDA_ERROR(cudaMemcpyAsync(host_result_data, cuda_result_buffer, result_buffer_size, cudaMemcpyDeviceToHost, cuda_stream));

        // 同步等待计算完成
        CHECK_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));

        output_results.reserve(output_results.size() + string_count);
        for (size_t i = 0; i < string_count; i++) {
            output_results.emplace_back(host_result_data + i * MAXIMUM_STRING_LENGTH);
        }

        // 清理主机内存
        free(host_input_data);
        free(host_result_data);

        return true;
    }
};

// ==================== 接口函数实现区域 ====================
void initialize_cuda_environment() {
    if (!global_cuda_manager) {
        global_cuda_manager = new CudaComputeManager();
    }
}

void cleanup_gpu_resources() {
    if (global_cuda_manager) {
        delete global_cuda_manager;
        global_cuda_manager = nullptr;
    }
}

// 对外提供的GPU处理接口
bool processWithGPU(const vector<string>& values, const string& prefix, vector<string>& results) {
    if (!global_cuda_manager) {
        initialize_cuda_environment();
    }
    return global_cuda_manager->process_string_batch(values, prefix, results);
}

// ==================== PCFG算法实现区域 ====================
// 以下代码保持不变（L189-360）
void PriorityQueue::CalProb(PT &pt)
{
    pt.prob = pt.preterm_prob;
    int index = 0;

    for (int idx : pt.curr_indices)
    {
        if (pt.content[index].type == 1)
        {
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
        }
        index += 1;
    }
}

void PriorityQueue::init()
{
    initialize_cuda_environment();
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            if (seg.type == 2)
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            if (seg.type == 3)
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        CalProb(pt);
        priority.emplace_back(pt);
    }
}

void PriorityQueue::PopNext()
{
    const int BATCH_SIZE = 64;// 批处理大小
    vector<PT> batch_pts;

    for (int i = 0; i < BATCH_SIZE && !priority.empty(); i++) {
        batch_pts.push_back(priority.front());
        priority.erase(priority.begin());
    }

    if (!batch_pts.empty()) {
        BatchGenerate(batch_pts);
    }

    for (PT& processed_pt : batch_pts) {
        vector<PT> new_pts = processed_pt.NewPTs();
        for (PT pt : new_pts) {
            CalProb(pt);
            bool inserted = false;
            for (auto iter = priority.begin(); iter != priority.end(); iter++) {
                if (pt.prob > iter->prob) {
                    priority.emplace(iter, pt);
                    inserted = true;
                    break;
                }
            }
            if (!inserted) {
                priority.emplace_back(pt);
            }
        }
    }
}

vector<PT> PT::NewPTs()
{
    vector<PT> res;
    if (content.size() == 1) {
        return res;
    }
    else {
        int init_pivot = pivot;
        for (int i = pivot; i < curr_indices.size() - 1; i += 1) {
            curr_indices[i] += 1;
            if (curr_indices[i] < max_indices[i]) {
                pivot = i;
                res.emplace_back(*this);
            }
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }
}

void PriorityQueue::Generate(PT pt)
{
    CalProb(pt);

    if (pt.content.size() == 1)
    {
        segment *a;
        if (pt.content[0].type == 1) a = &m.letters[m.FindLetter(pt.content[0])];
        if (pt.content[0].type == 2) a = &m.digits[m.FindDigit(pt.content[0])];
        if (pt.content[0].type == 3) a = &m.symbols[m.FindSymbol(pt.content[0])];

        int num_values = pt.max_indices[0];

        if (num_values >= GPU_ACCELERATION_THRESHOLD) {
            vector<string> temp_results;
            if (processWithGPU(a->ordered_values, "", temp_results)) {
                guesses.insert(guesses.end(), temp_results.begin(), temp_results.end());
                total_guesses += temp_results.size();
                return;
            }
        }

        // GPU处理失败或数据量不足，使用CPU处理
        for (int i = 0; i < num_values; i++)
        {
            guesses.push_back(a->ordered_values[i]);
            total_guesses++;
        }
    }
    else
    {
        string prefix;
        int seg_idx = 0;
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1) prefix += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            if (pt.content[seg_idx].type == 2) prefix += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            if (pt.content[seg_idx].type == 3) prefix += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1) break;
        }

        segment *a;
        if (pt.content.back().type == 1) a = &m.letters[m.FindLetter(pt.content.back())];
        if (pt.content.back().type == 2) a = &m.digits[m.FindDigit(pt.content.back())];
        if (pt.content.back().type == 3) a = &m.symbols[m.FindSymbol(pt.content.back())];

        int num_values = pt.max_indices.back();

        if (num_values >= GPU_ACCELERATION_THRESHOLD) {
            vector<string> temp_results;
            if (processWithGPU(a->ordered_values, prefix, temp_results)) {
                guesses.insert(guesses.end(), temp_results.begin(), temp_results.end());
                total_guesses += temp_results.size();
                return;
            }
        }

        // GPU处理失败或数据量不足，使用CPU处理
        for (int i = 0; i < num_values; i++)
        {
            guesses.push_back(prefix + a->ordered_values[i]);
            total_guesses++;
        }
    }
}

void PriorityQueue::BatchGenerate(const vector<PT>& batch_pts) {
    for (const PT& pt : batch_pts) {
        Generate(const_cast<PT&>(pt));
    }
}