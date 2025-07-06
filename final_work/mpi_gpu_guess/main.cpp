// #include "PCFG.h"
// #include <chrono>
// #include <fstream>
// #include "md5.h"
// #include <iomanip>
// using namespace std;
// using namespace chrono;

// // 编译指令如下
// // g++ main.cpp train.cpp guessing.cpp md5.cpp -o test.exe
// // g++ main.cpp train.cpp guessing.cpp md5.cpp -o test.exe -O1
// // g++ main.cpp train.cpp guessing.cpp md5.cpp -o test.exe -O2

// int main()
// {
//     double time_hash = 0;  // 用于MD5哈希的时间
//     double time_guess = 0; // 哈希和猜测的总时长
//     double time_train = 0; // 模型训练的总时长
//     PriorityQueue q;
//     auto start_train = system_clock::now();
//     q.m.train("./input/Rockyou-singleLined-full.txt");
//     q.m.order();
//     auto end_train = system_clock::now();
//     auto duration_train = duration_cast<microseconds>(end_train - start_train);
//     time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

//     q.init();
//     cout << "here" << endl;
//     int curr_num = 0;
//     auto start = system_clock::now();
//     // 由于需要定期清空内存，我们在这里记录已生成的猜测总数
//     int history = 0;
//     // std::ofstream a("./output/results.txt");
//     while (!q.priority.empty())
//     {
//         q.PopNext();
//         q.total_guesses = q.guesses.size();
//         if (q.total_guesses - curr_num >= 100000)
//         {
//             cout << "Guesses generated: " <<history + q.total_guesses << endl;
//             curr_num = q.total_guesses;

//             // 在此处更改实验生成的猜测上限
//             int generate_n=10000000;
//             if (history + q.total_guesses > 10000000)
//             {
//                 auto end = system_clock::now();
//                 auto duration = duration_cast<microseconds>(end - start);
//                 time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
//                 cout << "Guess time:" << time_guess - time_hash << "seconds"<< endl;
//                 cout << "Hash time:" << time_hash << "seconds"<<endl;
//                 cout << "Train time:" << time_train <<"seconds"<<endl;
//                 break;
//             }
//         }
//         // 为了避免内存超限，我们在q.guesses中口令达到一定数目时，将其中的所有口令取出并且进行哈希
//         // 然后，q.guesses将会被清空。为了有效记录已经生成的口令总数，维护一个history变量来进行记录
//         if (curr_num > 1000000)
//         {
//             auto start_hash = system_clock::now();
//             bit32 state[4];
//             for (string pw : q.guesses)
//             {
//                 // TODO：对于SIMD实验，将这里替换成你的SIMD MD5函数
//                 MD5Hash(pw, state);

//                 // 以下注释部分用于输出猜测和哈希，但是由于自动测试系统不太能写文件，所以这里你可以改成cout
//                 // a<<pw<<"\t";
//                 // for (int i1 = 0; i1 < 4; i1 += 1)
//                 // {
//                 //     a << std::setw(8) << std::setfill('0') << hex << state[i1];
//                 // }
//                 // a << endl;
//             }

//             // 在这里对哈希所需的总时长进行计算
//             auto end_hash = system_clock::now();
//             auto duration = duration_cast<microseconds>(end_hash - start_hash);
//             time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;

//             // 记录已经生成的口令总数
//             history += curr_num;
//             curr_num = 0;
//             q.guesses.clear();
//         }
//     }
// }


#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <mpi.h>
#include <cuda_runtime.h>  // 新增：CUDA运行时
using namespace std;
using namespace chrono;
//nvcc -ccbin mpic++ main.cpp train.cpp guessing_mpi_cuda.cu md5.cpp -o test_mpi_cuda -std=c++11 -lcuda -lcudart

// 声明外部清理函数
extern void cleanup_mpi_cuda_resources();

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // 检查CUDA设备可用性（仅在rank 0进程中报告）
    if (rank == 0) {
        int device_count;
        cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
        if (cuda_status == cudaSuccess && device_count > 0) {
            cout << "CUDA devices available: " << device_count << endl;
            for (int i = 0; i < device_count; i++) {
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, i);
                cout << "Device " << i << ": " << prop.name << " (Compute " << prop.major << "." << prop.minor << ")" << endl;
            }
        } else {
            cout << "No CUDA devices available, will use CPU-only processing" << endl;
        }
    }
    
    // 使用MPI计时工具
    double mpi_time_start_total = MPI_Wtime();  // 总体开始时间
    double time_hash = 0;  // 用于MD5哈希的时间
    double time_guess = 0; // 哈希和猜测的总时长
    double time_train = 0; // 模型训练的总时长
    
    PriorityQueue q;
    
    // MPI计时：训练阶段
    double mpi_time_train_start = MPI_Wtime();
    if (rank == 0) {
        cout << "Starting model training..." << endl;
    }
    
    q.m.train("input/Rockyou-singleLined-full.txt");
    q.m.order();
    
    double mpi_time_train_end = MPI_Wtime();
    time_train = mpi_time_train_end - mpi_time_train_start;
    
    if (rank == 0) {
        cout << "Model training completed in " << time_train << " seconds" << endl;
    }

    // 等待所有进程完成训练
    MPI_Barrier(MPI_COMM_WORLD);

    q.init();
    
    if (rank == 0) {
        cout << "Starting password generation with " << size << " MPI processes (with GPU acceleration)..." << endl;
    }
    
    int curr_num = 0;
    
    // MPI计时：猜测生成阶段
    double mpi_time_guess_start = MPI_Wtime();
    
    // 由于需要定期清空内存，我们在这里记录已生成的猜测总数
    int history = 0;
    
    while (!q.priority.empty())
    {
        q.PopNext();  // 现在使用Generate_mpi_cuda进行MPI+CUDA混合并行化
        
        // 收集所有进程的猜测总数
        int local_guesses = q.guesses.size();
        int global_guesses = 0;
        MPI_Allreduce(&local_guesses, &global_guesses, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        q.total_guesses = global_guesses;
        
        if (q.total_guesses - curr_num >= 100000)
        {
            if (rank == 0) {
                cout << "Process " << rank << " - Global guesses generated: " << history + q.total_guesses << endl;
            }
            curr_num = q.total_guesses;

            // 在此处更改实验生成的猜测上限
            int generate_n = 10000000;
            if (history + q.total_guesses > generate_n)
            {
                double mpi_time_guess_end = MPI_Wtime();
                time_guess = mpi_time_guess_end - mpi_time_guess_start;
                
                if (rank == 0) {
                    cout << "=== MPI+CUDA Timing Results (Process " << rank << ") ===" << endl;
                    cout << "Guess time: " << time_guess - time_hash << " seconds" << endl;
                    cout << "Hash time: " << time_hash << " seconds" << endl;
                    cout << "Train time: " << time_train << " seconds" << endl;
                    
                    double total_time = MPI_Wtime() - mpi_time_start_total;
                    cout << "Total execution time: " << total_time << " seconds" << endl;
                }
                break;
            }
        }
        
        // 为了避免内存超限，我们在q.guesses中口令达到一定数目时，将其中的所有口令取出并且进行哈希
        if (curr_num > 1000000)
        {
            // MPI计时：哈希阶段
            double mpi_time_hash_start = MPI_Wtime();
            bit32 state[4];
            for (string pw : q.guesses)
            {
                // TODO：对于SIMD实验，将这里替换成你的SIMD MD5函数
                MD5Hash(pw, state);

                // 以下注释部分用于输出猜测和哈希，但是由于自动测试系统不太能写文件，所以这里你可以改成cout
                // a<<pw<<"\t";
                // for (int i1 = 0; i1 < 4; i1 += 1)
                // {
                //     a << std::setw(8) << std::setfill('0') << hex << state[i1];
                // }
                // a << endl;
            }
            // MPI计时：哈希结束
            double mpi_time_hash_end = MPI_Wtime();
            double hash_duration = mpi_time_hash_end - mpi_time_hash_start;
            time_hash += hash_duration;
            
            if (rank == 0 && curr_num % 5000000 == 0) {
                cout << "Process " << rank << " - Hash batch completed in " << hash_duration << " seconds" << endl;
            }

            // 记录已经生成的口令总数
            history += curr_num;
            curr_num = 0;
            q.guesses.clear();
        }
    }
    
    // 清理CUDA资源（在MPI_Finalize之前）
    cleanup_mpi_cuda_resources();
    
    // 最终的MPI计时统计
    double mpi_time_end_total = MPI_Wtime();
    double total_execution_time = mpi_time_end_total - mpi_time_start_total;
    
    // 收集所有进程的计时信息
    double max_total_time, min_total_time, avg_total_time;
    MPI_Reduce(&total_execution_time, &max_total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_execution_time, &min_total_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_execution_time, &avg_total_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        avg_total_time /= size;
        cout << "\n=== Final MPI+CUDA Timing Summary ===" << endl;
        cout << "Number of MPI processes: " << size << endl;
        cout << "Max execution time across all processes: " << max_total_time << " seconds" << endl;
        cout << "Min execution time across all processes: " << min_total_time << " seconds" << endl;
        cout << "Average execution time across all processes: " << avg_total_time << " seconds" << endl;
        cout << "Load balance efficiency: " << (min_total_time / max_total_time) * 100 << "%" << endl;
    }
    
    MPI_Finalize();
    return 0;
}