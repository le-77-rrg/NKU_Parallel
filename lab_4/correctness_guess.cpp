#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <unordered_set>
using namespace std;
using namespace chrono;
#include <mpi.h>

//mpic++ correctness_guess.cpp train.cpp guessing.cpp md5.cpp -o test_mpi_optimized -O2 -std=c++11

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double time_hash = 0;  // 用于MD5哈希的时间
    double time_guess = 0; // 哈希和猜测的总时长
    double time_train = 0; // 模型训练的总时长
    PriorityQueue q;
    
    // 只有rank 0进行训练和输出
    if (rank == 0) {
        cout << "Starting model training with " << size << " MPI processes..." << endl;
    }
    
    auto start_train = system_clock::now();
    q.m.train("input/Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    if (rank == 0) {
        cout << "Model training completed in " << time_train << " seconds" << endl;
    }
    
    // 等待所有进程完成训练
    MPI_Barrier(MPI_COMM_WORLD);

    // 加载测试数据（只在rank 0加载，然后广播）
    unordered_set<std::string> test_set;
    vector<string> test_passwords;
    
    if (rank == 0) {
        ifstream test_data("input/Rockyou-singleLined-full.txt");
        int test_count = 0;
        string pw;
        while(test_data >> pw) {   
            test_count += 1;
            test_set.insert(pw);
            test_passwords.push_back(pw);
            if (test_count >= 1000000) {
                break;
            }
        }
        cout << "Loaded " << test_count << " test passwords" << endl;
    }
    
    // 广播测试密码数量
    int test_size = test_passwords.size();
    MPI_Bcast(&test_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // 广播测试密码到所有进程
    if (rank != 0) {
        test_passwords.resize(test_size);
    }
    
    for (int i = 0; i < test_size; i++) {
        int pw_len = 0;
        if (rank == 0) {
            pw_len = test_passwords[i].length();
        }
        MPI_Bcast(&pw_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank != 0) {
            test_passwords[i].resize(pw_len);
        }
        MPI_Bcast(&test_passwords[i][0], pw_len, MPI_CHAR, 0, MPI_COMM_WORLD);
        
        if (rank != 0) {
            test_set.insert(test_passwords[i]);
        }
    }
    
    int local_cracked = 0;
    int global_cracked = 0;

    q.init();
    
    if (rank == 0) {
        cout << "Starting password generation and cracking..." << endl;
    }
    
    int curr_num = 0;
    auto start = system_clock::now();
    int history = 0;
    
    while (!q.priority.empty())
    {
        q.PopNext();  // 这里会调用Generate_mpi进行MPI并行化
        
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

            int generate_n = 10000000;
            if (history + q.total_guesses > generate_n)
            {
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                
                // 收集所有进程的破解数量
                MPI_Allreduce(&local_cracked, &global_cracked, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
                
                if (rank == 0) {
                    cout << "=== Final Results ===" << endl;
                    cout << "Guess time: " << time_guess - time_hash << " seconds" << endl;
                    cout << "Hash time: " << time_hash << " seconds" << endl;
                    cout << "Train time: " << time_train << " seconds" << endl;
                    cout << "Total cracked: " << global_cracked << endl;
                    cout << "Total guesses: " << history + q.total_guesses << endl;
                }
                break;
            }
        }
        
        if (curr_num > 1000000)
        {
            auto start_hash = system_clock::now();
            
            // 每个进程处理自己的密码
            for (const string& pw : q.guesses)
            {
                if (test_set.find(pw) != test_set.end()) {
                    local_cracked++;
                }
                
                bit32 state[4];
                MD5Hash(pw, state);
            }

            auto end_hash = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);
            time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;

            // 收集当前破解数量并输出进度
            int current_global_cracked = 0;
            MPI_Allreduce(&local_cracked, &current_global_cracked, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            
            if (rank == 0) {
                cout << "Hash batch completed. Current cracked: " << current_global_cracked << endl;
            }

            history += curr_num;
            curr_num = 0;
            q.guesses.clear();
        }
    }
    
    // 最终统计
    MPI_Allreduce(&local_cracked, &global_cracked, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    if (rank == 0) {
        cout << "\n=== MPI Final Summary ===" << endl;
        cout << "Number of MPI processes: " << size << endl;
        cout << "Total passwords cracked: " << global_cracked << endl;
        cout << "Cracking efficiency: " << (double)global_cracked / 1000000 * 100 << "%" << endl;
    }

    MPI_Finalize();
    return 0;
}
