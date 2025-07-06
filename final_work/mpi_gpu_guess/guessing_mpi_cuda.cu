#include "PCFG.h"
#include <chrono>
#include <queue>
#include <condition_variable>
#include <cstring>
#include <unistd.h>
#include <sstream>
#include <mpi.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <string>
#include <iostream>

using namespace std;

// ==================== CUDA相关常量和宏定义 ====================
#define GPU_ACCELERATION_THRESHOLD 2000
#define CHECK_CUDA_ERROR(operation) do { \
    cudaError_t cuda_result = operation; \
    if (cuda_result != cudaSuccess) { \
        fprintf(stderr, "CUDA运行错误 %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(cuda_result)); \
        exit(1); \
    } \
} while(0)

// ==================== CUDA核函数 ====================
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

// ==================== CUDA管理类 ====================
class CudaComputeManager {
private:
    static const size_t MAXIMUM_ITEM_COUNT = 120000;
    static const size_t MAXIMUM_INPUT_CAPACITY = 60 * 1024 * 1024;
    static const size_t MAXIMUM_OUTPUT_CAPACITY = 120 * 1024 * 1024;

    char* cuda_input_buffer = nullptr;
    char* cuda_prefix_buffer = nullptr;
    char* cuda_result_buffer = nullptr;
    int* cuda_offset_array = nullptr;
    int* cuda_length_array = nullptr;
    cudaStream_t cuda_stream;
    bool initialized = false;

public:
    CudaComputeManager() {
        // 检查CUDA设备可用性
        int device_count;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess || device_count == 0) {
            initialized = false;
            return;
        }

        // 分配GPU内存空间
        if (cudaMalloc(&cuda_input_buffer, MAXIMUM_INPUT_CAPACITY) != cudaSuccess ||
            cudaMalloc(&cuda_prefix_buffer, 1024) != cudaSuccess ||
            cudaMalloc(&cuda_result_buffer, MAXIMUM_OUTPUT_CAPACITY) != cudaSuccess ||
            cudaMalloc(&cuda_offset_array, MAXIMUM_ITEM_COUNT * sizeof(int)) != cudaSuccess ||
            cudaMalloc(&cuda_length_array, MAXIMUM_ITEM_COUNT * sizeof(int)) != cudaSuccess ||
            cudaStreamCreate(&cuda_stream) != cudaSuccess) {
            initialized = false;
            cleanup();
            return;
        }
        initialized = true;
    }

    ~CudaComputeManager() {
        cleanup();
    }

    void cleanup() {
        if (cuda_input_buffer) cudaFree(cuda_input_buffer);
        if (cuda_prefix_buffer) cudaFree(cuda_prefix_buffer);
        if (cuda_result_buffer) cudaFree(cuda_result_buffer);
        if (cuda_offset_array) cudaFree(cuda_offset_array);
        if (cuda_length_array) cudaFree(cuda_length_array);
        if (initialized) cudaStreamDestroy(cuda_stream);
    }

    bool isInitialized() const { return initialized; }

    // 批量处理字符串拼接任务（支持MPI分片）
    bool process_string_batch_mpi(const vector<string>& input_strings, const string& prefix_string, 
                                  vector<string>& output_results, int start_idx, int end_idx) {
        if (!initialized || start_idx >= end_idx || start_idx >= input_strings.size()) {
            return false;
        }

        // 调整范围
        end_idx = min(end_idx, (int)input_strings.size());
        int actual_count = end_idx - start_idx;
        
        if (actual_count == 0 || actual_count > MAXIMUM_ITEM_COUNT) return false;

        // 计算输入数据总大小
        size_t total_input_bytes = 0;
        for(int i = start_idx; i < end_idx; i++) {
            total_input_bytes += input_strings[i].length();
        }
        if (total_input_bytes > MAXIMUM_INPUT_CAPACITY) return false;

        char* host_input_data = (char*)malloc(total_input_bytes);
        vector<int> host_offset_data(actual_count);
        vector<int> host_length_data(actual_count);
        if (!host_input_data) return false;

        size_t byte_position = 0;
        for(int i = 0; i < actual_count; ++i) {
            int actual_idx = start_idx + i;
            host_offset_data[i] = byte_position;
            host_length_data[i] = input_strings[actual_idx].length();
            memcpy(host_input_data + byte_position, input_strings[actual_idx].c_str(), host_length_data[i]);
            byte_position += host_length_data[i];
        }

        const int MAXIMUM_STRING_LENGTH = 64;
        size_t result_buffer_size = actual_count * MAXIMUM_STRING_LENGTH;
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
        CHECK_CUDA_ERROR(cudaMemcpyAsync(cuda_offset_array, host_offset_data.data(), actual_count * sizeof(int), cudaMemcpyHostToDevice, cuda_stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(cuda_length_array, host_length_data.data(), actual_count * sizeof(int), cudaMemcpyHostToDevice, cuda_stream));

        char* device_prefix_pointer = nullptr;
        if (!prefix_string.empty()) {
            CHECK_CUDA_ERROR(cudaMemcpyAsync(cuda_prefix_buffer, prefix_string.c_str(), prefix_string.length(), cudaMemcpyHostToDevice, cuda_stream));
            device_prefix_pointer = cuda_prefix_buffer;
        }

        // 启动CUDA核函数
        const int threads_per_block = 512;
        int blocks_per_grid = (actual_count + threads_per_block - 1) / threads_per_block;
        string_concatenation_kernel<<<blocks_per_grid, threads_per_block, 0, cuda_stream>>>(
            cuda_input_buffer, cuda_offset_array, cuda_length_array, device_prefix_pointer, prefix_string.length(),
            cuda_result_buffer, MAXIMUM_STRING_LENGTH, actual_count);

        // 异步数据传输：设备到主机
        CHECK_CUDA_ERROR(cudaMemcpyAsync(host_result_data, cuda_result_buffer, result_buffer_size, cudaMemcpyDeviceToHost, cuda_stream));

        // 同步等待计算完成
        CHECK_CUDA_ERROR(cudaStreamSynchronize(cuda_stream));

        output_results.reserve(output_results.size() + actual_count);
        for (int i = 0; i < actual_count; i++) {
            output_results.emplace_back(host_result_data + i * MAXIMUM_STRING_LENGTH);
        }

        // 清理主机内存
        free(host_input_data);
        free(host_result_data);

        return true;
    }
};

// 全局CUDA管理器
static CudaComputeManager* global_cuda_manager = nullptr;

// ==================== 原有PCFG函数实现 ====================
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
    // 初始化CUDA管理器
    if (!global_cuda_manager) {
        global_cuda_manager = new CudaComputeManager();
    }
    
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
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        int init_pivot = pivot;
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            curr_indices[i] += 1;
            if (curr_indices[i] < max_indices[i])
            {
                pivot = i;
                res.emplace_back(*this);
            }
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }
}

// ==================== MPI+CUDA混合并行实现 ====================
void PriorityQueue::Generate_mpi_cuda(PT pt)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    CalProb(pt);
    
    if (pt.content.size() == 1) {
        segment *a;
        if (pt.content[0].type == 1) {
            a = &m.letters[m.FindLetter(pt.content[0])];
        }
        else if (pt.content[0].type == 2) {
            a = &m.digits[m.FindDigit(pt.content[0])];
        }
        else if (pt.content[0].type == 3) {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }
        
        int total = pt.max_indices[0];
        
        // MPI负载均衡：动态分配
        int base_chunk = total / size;
        int remainder = total % size;
        int start_idx, end_idx;
        
        if (rank < remainder) {
            start_idx = rank * (base_chunk + 1);
            end_idx = start_idx + base_chunk + 1;
        } else {
            start_idx = remainder * (base_chunk + 1) + (rank - remainder) * base_chunk;
            end_idx = start_idx + base_chunk;
        }
        
        int local_count = end_idx - start_idx;
        
        // 尝试使用GPU加速（如果数据量足够大且GPU可用）
        if (local_count >= GPU_ACCELERATION_THRESHOLD && global_cuda_manager && global_cuda_manager->isInitialized()) {
            vector<string> temp_results;
            if (global_cuda_manager->process_string_batch_mpi(a->ordered_values, "", temp_results, start_idx, end_idx)) {
                guesses.insert(guesses.end(), temp_results.begin(), temp_results.end());
            } else {
                // GPU处理失败，回退到CPU
                for (int i = start_idx; i < end_idx; i++) {
                    guesses.push_back(a->ordered_values[i]);
                }
            }
        } else {
            // 数据量不足或GPU不可用，使用CPU处理
            for (int i = start_idx; i < end_idx; i++) {
                guesses.push_back(a->ordered_values[i]);
            }
        }
        
        // MPI合并：收集所有进程的计数
        int global_total = 0;
        MPI_Allreduce(&local_count, &global_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        total_guesses = global_total;
    }
    else {
        string prefix;
        int seg_idx = 0;
        for (int idx : pt.curr_indices) {
            if (pt.content[seg_idx].type == 1) {
                prefix += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }
            else if (pt.content[seg_idx].type == 2) {
                prefix += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }
            else if (pt.content[seg_idx].type == 3) {
                prefix += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx++;
            if (seg_idx == pt.content.size() - 1) {
                break;
            }
        }
        
        segment *a;
        int last_seg_idx = pt.content.size() - 1;
        if (pt.content[last_seg_idx].type == 1) {
            a = &m.letters[m.FindLetter(pt.content[last_seg_idx])];
        }
        else if (pt.content[last_seg_idx].type == 2) {
            a = &m.digits[m.FindDigit(pt.content[last_seg_idx])];
        }
        else if (pt.content[last_seg_idx].type == 3) {
            a = &m.symbols[m.FindSymbol(pt.content[last_seg_idx])];
        }
        
        int total = pt.max_indices[pt.content.size() - 1];
        
        // MPI负载均衡：动态分配
        int base_chunk = total / size;
        int remainder = total % size;
        int start_idx, end_idx;
        
        if (rank < remainder) {
            start_idx = rank * (base_chunk + 1);
            end_idx = start_idx + base_chunk + 1;
        } else {
            start_idx = remainder * (base_chunk + 1) + (rank - remainder) * base_chunk;
            end_idx = start_idx + base_chunk;
        }
        
        int local_count = end_idx - start_idx;
        
        // 尝试使用GPU加速（如果数据量足够大且GPU可用）
        if (local_count >= GPU_ACCELERATION_THRESHOLD && global_cuda_manager && global_cuda_manager->isInitialized()) {
            vector<string> temp_results;
            if (global_cuda_manager->process_string_batch_mpi(a->ordered_values, prefix, temp_results, start_idx, end_idx)) {
                guesses.insert(guesses.end(), temp_results.begin(), temp_results.end());
            } else {
                // GPU处理失败，回退到CPU
                for (int i = start_idx; i < end_idx; i++) {
                    guesses.push_back(prefix + a->ordered_values[i]);
                }
            }
        } else {
            // 数据量不足或GPU不可用，使用CPU处理
            for (int i = start_idx; i < end_idx; i++) {
                guesses.push_back(prefix + a->ordered_values[i]);
            }
        }
        
        // MPI合并：收集所有进程的计数
        int global_total = 0;
        MPI_Allreduce(&local_count, &global_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        total_guesses = global_total;
    }
}

// ==================== 资源清理函数 ====================
void cleanup_mpi_cuda_resources() {
    if (global_cuda_manager) {
        delete global_cuda_manager;
        global_cuda_manager = nullptr;
    }
}

void PriorityQueue::BatchGenerate(const vector<PT>& batch_pts) {
    for (const PT& pt : batch_pts) {
        Generate_mpi_cuda(const_cast<PT&>(pt));
    }
}
