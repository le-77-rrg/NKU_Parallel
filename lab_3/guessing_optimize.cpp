#include "PCFG.h"
#include <chrono>
#include <queue>
#include <condition_variable>
#include <cstring>  // 添加这一行，支持 memset 函数
#ifdef _OPENMP
#include <omp.h>
#endif
#include<unistd.h>

using namespace std;
using namespace std::chrono;

// 修改线程池任务结构，添加线程本地结果数组
struct Task {
    int start, end;
    string prefix;
    vector<string>* values;
    string* guesses_out;        // 可以为空，表示不使用共享结果数组
    int* guesses_count;
    int thread_id;              // 新增：线程ID，用于标识写入哪个线程本地结果数组
    vector<string>* local_results; // 新增：线程本地结果数组
};

// 线程池类
// 修改线程池类，添加任务完成通知机制
class ThreadPool {
private:
    vector<pthread_t> workers;
    queue<Task> tasks;
    pthread_mutex_t queue_mutex;
    pthread_cond_t condition;
    pthread_cond_t completion_cond;  // 新增完成条件变量
    int active_tasks;                // 新增活跃任务计数器
    bool stop;
    
public:
    // 初始化线程池
    ThreadPool(size_t threads) : stop(false), active_tasks(0) {
        pthread_mutex_init(&queue_mutex, NULL);
        pthread_cond_init(&condition, NULL);
        pthread_cond_init(&completion_cond, NULL);  // 初始化完成条件变量
        
        workers.resize(threads);
        for (size_t i = 0; i < threads; ++i) {
            pthread_create(&workers[i], NULL, &ThreadPool::WorkerThread, this);
        }
    }
    
    // 析构函数，清理线程池
    ~ThreadPool() {
        {
            pthread_mutex_lock(&queue_mutex);
            stop = true;
            pthread_mutex_unlock(&queue_mutex);
        }
        pthread_cond_broadcast(&condition);
        
        for (size_t i = 0; i < workers.size(); ++i) {
            pthread_join(workers[i], NULL);
        }
        
        pthread_mutex_destroy(&queue_mutex);
        pthread_cond_destroy(&condition);
        pthread_cond_destroy(&completion_cond);  // 销毁完成条件变量
    }
    
    // 添加任务到线程池
    void Enqueue(Task task) {
        pthread_mutex_lock(&queue_mutex);
        tasks.push(task);
        active_tasks++;  // 增加活跃任务计数
        pthread_mutex_unlock(&queue_mutex);
        pthread_cond_signal(&condition);
    }
    
    // 等待所有任务完成
    void WaitAll() {
        pthread_mutex_lock(&queue_mutex);
        while (active_tasks > 0 || !tasks.empty()) {
            pthread_cond_wait(&completion_cond, &queue_mutex);
        }
        pthread_mutex_unlock(&queue_mutex);
    }
    
private:
    // 工作线程函数
    static void* WorkerThread(void* arg) {
        ThreadPool* pool = static_cast<ThreadPool*>(arg);
        
        while (true) {
            Task task;
            bool got_task = false;
            
            {
                pthread_mutex_lock(&pool->queue_mutex);
                
                // 如果队列为空且线程池停止，则退出
                if (pool->stop && pool->tasks.empty()) {
                    pthread_mutex_unlock(&pool->queue_mutex);
                    return NULL;
                }
                
                // 如果队列不为空，取出一个任务
                if (!pool->tasks.empty()) {
                    task = pool->tasks.front();
                    pool->tasks.pop();
                    got_task = true;
                } else if (!pool->stop) {
                    // 如果队列为空且线程池未停止，等待条件变量
                    pthread_cond_wait(&pool->condition, &pool->queue_mutex);
                    pthread_mutex_unlock(&pool->queue_mutex);
                    continue;
                } else {
                    pthread_mutex_unlock(&pool->queue_mutex);
                    continue;
                }
                
                pthread_mutex_unlock(&pool->queue_mutex);
            }
            
            if (got_task) {
                // 执行任务
                int local_count = 0;
                
                // 根据任务类型选择写入目标
                if (task.local_results != nullptr) {
                    // 写入线程本地结果数组（无锁）
                    for (int i = task.start; i < task.end; ++i) {
                        string guess = task.prefix + (*(task.values))[i];
                        task.local_results->push_back(std::move(guess));
                        local_count++;
                    }
                } else if (task.guesses_out != nullptr) {
                    // 写入共享结果数组（传统方式）
                    for (int i = task.start; i < task.end; ++i) {
                        string guess = task.prefix + (*(task.values))[i];
                        task.guesses_out[i] = std::move(guess);
                        local_count++;
                    }
                }
                
                // 更新线程局部计数并减少活跃任务计数
                pthread_mutex_lock(&pool->queue_mutex);
                *(task.guesses_count) = local_count;
                pool->active_tasks--;
                
                // 如果没有活跃任务，发送完成信号
                if (pool->active_tasks == 0 && pool->tasks.empty()) {
                    pthread_cond_signal(&pool->completion_cond);
                }
                pthread_mutex_unlock(&pool->queue_mutex);
            }
        }
        
        return NULL;
    }
};

// 全局线程池，在程序启动时创建
ThreadPool* global_thread_pool = nullptr;

// 初始化全局线程池
void InitThreadPool(int num_threads) {
    if (global_thread_pool == nullptr) {
        global_thread_pool = new ThreadPool(num_threads);
        // 删除打印信息
        // cout << "线程池已初始化，线程数: " << num_threads << endl;
    }
}

// 清理全局线程池
void CleanupThreadPool() {
    if (global_thread_pool != nullptr) {
        delete global_thread_pool;
        global_thread_pool = nullptr;
        // 删除打印信息
        // cout << "线程池已清理" << endl;
    }
}

void PriorityQueue::CalProb(PT &pt)
{
    // 计算PriorityQueue里面一个PT的流程如下：
    // 1. 首先需要计算一个PT本身的概率。例如，L6S1的概率为0.15
    // 2. 需要注意的是，Queue里面的PT不是“纯粹的”PT，而是除了最后一个segment以外，全部被value实例化的PT
    // 3. 所以，对于L6S1而言，其在Queue里面的实际PT可能是123456S1，其中“123456”为L6的一个具体value。
    // 4. 这个时候就需要计算123456在L6中出现的概率了。假设123456在所有L6 segment中的概率为0.1，那么123456S1的概率就是0.1*0.15

    // 计算一个PT本身的概率。后续所有具体segment value的概率，直接累乘在这个初始概率值上
    pt.prob = pt.preterm_prob;

    // index: 标注当前segment在PT中的位置
    int index = 0;


    for (int idx : pt.curr_indices)
    {
        // pt.content[index].PrintSeg();
        if (pt.content[index].type == 1)
        {
            // 下面这行代码的意义：
            // pt.content[index]：目前需要计算概率的segment
            // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
            // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
            // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
            // cout << m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.letters[m.FindLetter(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
            // cout << m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.digits[m.FindDigit(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].total_freq << endl;
        }
        index += 1;
    }
    // cout << pt.prob << endl;
}

void PriorityQueue::init()
{
    // cout << m.ordered_pts.size() << endl;
    // 用所有可能的PT，按概率降序填满整个优先队列
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                // 下面这行代码的意义：
                // max_indices用来表示PT中各个segment的可能数目。例如，L6S1中，假设模型统计到了100个L6，那么L6对应的最大下标就是99
                // （但由于后面采用了"<"的比较关系，所以其实max_indices[0]=100）
                // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
                // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
                // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
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
        // pt.PrintPT();
        // cout << " " << m.preterm_freq[m.FindPT(pt)] << " " << m.total_preterm << " " << pt.preterm_prob << endl;

        // 计算当前pt的概率
        CalProb(pt);
        // 将PT放入优先队列
        priority.emplace_back(pt);
    }
    // cout << "priority size:" << priority.size() << endl;
}

void PriorityQueue::PopNext()
{

    // 对优先队列最前面的PT，首先利用这个PT生成一系列猜测
    //Generate(priority.front()); //调用不同函数
    Generate_pthread_pool(priority.front()); //调用不同函数
    //Generate_pthread(priority.front()); //调用不同函数
    //Generate_openmp(priority.front()); // 使用OpenMP版本
    // 然后需要根据即将出队的PT，生成一系列新的PT
    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts)
    {
        // 计算概率
        CalProb(pt);
        // 接下来的这个循环，作用是根据概率，将新的PT插入到优先队列中
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            // 对于非队首和队尾的特殊情况
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
                // 判定概率
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                {
                    priority.emplace(iter + 1, pt);
                    break;
                }
            }
            if (iter == priority.end() - 1)
            {
                priority.emplace_back(pt);
                break;
            }
            if (iter == priority.begin() && iter->prob < pt.prob)
            {
                priority.emplace(iter, pt);
                break;
            }
        }
    }

    // 现在队首的PT善后工作已经结束，将其出队（删除）
    priority.erase(priority.begin());
}

// 这个函数你就算看不懂，对并行算法的实现影响也不大
// 当然如果你想做一个基于多优先队列的并行算法，可能得稍微看一看了
vector<PT> PT::NewPTs()
{
    // 存储生成的新PT
    vector<PT> res;

    // 假如这个PT只有一个segment
    // 那么这个segment的所有value在出队前就已经被遍历完毕，并作为猜测输出
    // 因此，所有这个PT可能对应的口令猜测已经遍历完成，无需生成新的PT
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        // 最初的pivot值。我们将更改位置下标大于等于这个pivot值的segment的值（最后一个segment除外），并且一次只更改一个segment
        // 上面这句话里是不是有没看懂的地方？接着往下看你应该会更明白
        int init_pivot = pivot;

        // 开始遍历所有位置值大于等于init_pivot值的segment
        // 注意i < curr_indices.size() - 1，也就是除去了最后一个segment（这个segment的赋值预留给并行环节）
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            // curr_indices: 标记各segment目前的value在模型里对应的下标
            curr_indices[i] += 1;

            // max_indices：标记各segment在模型中一共有多少个value
            if (curr_indices[i] < max_indices[i])
            {
                // 更新pivot值
                pivot = i;
                res.emplace_back(*this);
            }

            // 这个步骤对于你理解pivot的作用、新PT生成的过程而言，至关重要
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }

    return res;
}


// 这个函数是PCFG并行化算法的主要载体
// 尽量看懂，然后进行并行实现
void PriorityQueue::Generate(PT pt)
{
    // 计算PT的概率，这里主要是给PT的概率进行初始化
    CalProb(pt);

    // 对于只有一个segment的PT，直接遍历生成其中的所有value即可
    if (pt.content.size() == 1)
    {
        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        // 在模型中定位到这个segment
        if (pt.content[0].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[0])];
        }
        if (pt.content[0].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[0])];
        }
        if (pt.content[0].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }
        
        // Multi-thread TODO：
        // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
        // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
        // 这个过程是可以高度并行化的
        for (int i = 0; i < pt.max_indices[0]; i += 1)
        {
            string guess = a->ordered_values[i];
            // cout << guess << endl;
            guesses.emplace_back(guess);
            total_guesses += 1;
        }
    }
    else
    {
        string guess;
        int seg_idx = 0;
        // 这个for循环的作用：给当前PT的所有segment赋予实际的值（最后一个segment除外）
        // segment值根据curr_indices中对应的值加以确定
        // 这个for循环你看不懂也没太大问题，并行算法不涉及这里的加速
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
            {
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 2)
            {
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 3)
            {
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
            {
                break;
            }
        }

        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        if (pt.content[pt.content.size() - 1].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];
        }
        
        // Multi-thread TODO：
        // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
        // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
        // 这个过程是可以高度并行化的
        for (int i = 0; i < pt.max_indices[pt.content.size() - 1]; i += 1)
        {
            string temp = guess + a->ordered_values[i];
            // cout << temp << endl;
            guesses.emplace_back(temp);
            total_guesses += 1;
        }
    }
}

// 线程参数结构
struct ThreadArgs {
    int start;
    int end;
    string prefix;
    vector<string>* values;
    string* results;
    int* count;
    int* total_count;
};

// 线程函数优化
void* GenerateGuesses(void* arg) {
    ThreadArgs* args = static_cast<ThreadArgs*>(arg);
    
    // 使用局部变量减少指针解引用
    int start = args->start;
    int end = args->end;
    string prefix = args->prefix;
    vector<string>* values = args->values;
    string* results = args->results;
    
    // 直接写入预分配的结果数组
    for (int i = start; i < end; ++i) {
        results[i] = prefix + (*values)[i];
    }
    
    // 一次性更新计数，减少原子操作
    *(args->count) = end - start;
    
    return NULL;
}

void PriorityQueue::Generate_pthread(PT pt)
{
    // 计算PT的概率
    CalProb(pt);

    // 对于只有一个segment的PT，直接遍历生成其中的所有value即可
    if (pt.content.size() == 1)
    {
        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        // 在模型中定位到这个segment
        if (pt.content[0].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[0])];
        }
        if (pt.content[0].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[0])];
        }
        if (pt.content[0].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }
        
        int total = pt.max_indices[0];
        
        // 提高任务粒度阈值，减少小任务的并行开销
        if (total < 5000) {
            // 直接使用串行处理
            for (int i = 0; i < total; i++) {
                guesses.push_back(a->ordered_values[i]);
            }
            total_guesses += total;
            return;
        }
        
        // 使用固定的线程数量
        int num_threads = 4;
        int chunk = (total + num_threads - 1) / num_threads;
        
        // 预分配结果数组
        string* results = new string[total];
        int thread_counts[num_threads];
        memset(thread_counts, 0, sizeof(thread_counts));
        
        pthread_t threads[num_threads];
        ThreadArgs args[num_threads];
        
        // 创建线程并分配任务
        for (int t = 0; t < num_threads; ++t) {
            args[t] = {
                t * chunk,
                min((t + 1) * chunk, total),
                "",
                &a->ordered_values,
                results,
                &thread_counts[t],
                &total_guesses
            };
            pthread_create(&threads[t], NULL, GenerateGuesses, &args[t]);
        }
        
        // 等待所有线程完成
        for (int t = 0; t < num_threads; ++t) {
            pthread_join(threads[t], NULL);
        }
        
        // 批量添加结果 - 优化内存操作
        guesses.reserve(guesses.size() + total);
        for (int i = 0; i < total; ++i) {
            guesses.push_back(std::move(results[i]));
        }
        
        // 更新总计数
        total_guesses += total;
        
        // 释放内存
        delete[] results;
    }
    else
    {
        string guess;
        int seg_idx = 0;
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
            {
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 2)
            {
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 3)
            {
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
            {
                break;
            }
        }

        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        if (pt.content[pt.content.size() - 1].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];
        }
        
        int total = pt.max_indices[pt.content.size() - 1];
        
        // 提高任务粒度阈值
        if (total < 5000) {
            // 直接使用串行处理
            for (int i = 0; i < total; i++) {
                guesses.push_back(guess + a->ordered_values[i]);
            }
            total_guesses += total;
            return;
        }
        
        // 使用固定的线程数量
        int num_threads = 4;
        int chunk = (total + num_threads - 1) / num_threads;
        
        // 预分配结果数组
        string* results = new string[total];
        int thread_counts[num_threads];
        memset(thread_counts, 0, sizeof(thread_counts));
        
        pthread_t threads[num_threads];
        ThreadArgs args[num_threads];
        
        // 创建线程并分配任务
        for (int t = 0; t < num_threads; ++t) {
            args[t] = {
                t * chunk,
                min((t + 1) * chunk, total),
                guess,
                &a->ordered_values,
                results,
                &thread_counts[t],
                &total_guesses
            };
            pthread_create(&threads[t], NULL, GenerateGuesses, &args[t]);
        }
        
        // 等待所有线程完成
        for (int t = 0; t < num_threads; ++t) {
            pthread_join(threads[t], NULL);
        }
        
        // 批量添加结果 - 优化内存操作
        guesses.reserve(guesses.size() + total);
        for (int i = 0; i < total; ++i) {
            guesses.push_back(std::move(results[i]));
        }
        
        // 更新总计数
        total_guesses += total;
        
        // 释放内存
        delete[] results;
    }
}

void PriorityQueue::Generate_openmp(PT pt)
{
    CalProb(pt);

    if (pt.content.size() == 1)
    {
        segment *a;
        if (pt.content[0].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[0])];
        }
        if (pt.content[0].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[0])];
        }
        if (pt.content[0].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }

        // OpenMP 并行化
        #pragma omp parallel
        {
            std::vector<std::string> local_guesses;

            #pragma omp for nowait
            for (int i = 0; i < pt.max_indices[0]; i++)
            {
                string guess = a->ordered_values[i];
                local_guesses.emplace_back(guess);
            }

            // 合并局部结果
            #pragma omp critical
            {
                guesses.insert(guesses.end(), local_guesses.begin(), local_guesses.end());
                total_guesses += local_guesses.size();
            }
        }
    }
    else
    {
        string guess;
        int seg_idx = 0;
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
            {
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 2)
            {
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 3)
            {
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
            {
                break;
            }
        }

        segment *a;
        if (pt.content.back().type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content.back())];
        }
        if (pt.content.back().type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content.back())];
        }
        if (pt.content.back().type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content.back())];
        }

        // OpenMP 并行化
        #pragma omp parallel
        {
            std::vector<std::string> local_guesses;

            #pragma omp for nowait
            for (int i = 0; i < pt.max_indices[pt.content.size() - 1]; i++)
            {
                string temp = guess + a->ordered_values[i];
                local_guesses.emplace_back(temp);
            }

            // 合并局部结果
            #pragma omp critical
            {
                guesses.insert(guesses.end(), local_guesses.begin(), local_guesses.end());
                total_guesses += local_guesses.size();
            }
        }
    }
}

void PriorityQueue::Generate_pthread_pool(PT pt)
{
    // 计算PT的概率
    CalProb(pt);

    // 对于只有一个segment的PT，直接遍历生成其中的所有value
    if (pt.content.size() == 1)
    {
        // 指向segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        // 在模型中定位到这个segment
        if (pt.content[0].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[0])];
        }
        if (pt.content[0].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[0])];
        }
        if (pt.content[0].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }
        
        int total = pt.max_indices[0];
        
        // 提高任务粒度阈值，减少小任务的并行开销
        if (total < 100000) {
            // 直接使用串行处理
            for (int i = 0; i < total; i++) {
                guesses.push_back(a->ordered_values[i]);
            }
            total_guesses += total;
            return;
        }
        
        // 使用更多的线程和更小的任务块
        int num_threads = 8; // 自动获取CPU核心数
        int optimal_chunk_size = 1000; // 每个任务处理1000个元素
        int num_chunks = (total + optimal_chunk_size - 1) / optimal_chunk_size;
        
        // 预分配结果数组
        guesses.reserve(guesses.size() + total); // 预先分配空间避免频繁扩容
        string* results = new string[total];
        int* thread_counts = new int[num_chunks];
        memset(thread_counts, 0, sizeof(int) * num_chunks);
        
        // 创建任务并提交到线程池
        for (int t = 0; t < num_chunks; ++t) {
            int start = t * optimal_chunk_size;
            int end = std::min((t + 1) * optimal_chunk_size, total);
            
            Task task = {
                start,
                end,
                "",
                &a->ordered_values,
                results,
                &thread_counts[t]
            };
            
            global_thread_pool->Enqueue(task);
        }
        
        // 使用条件变量等待所有任务完成，而不是轮询
        global_thread_pool->WaitAll();
        
        // 批量添加结果 - 优化内存操作
        for (int i = 0; i < total; ++i) {
            guesses.push_back(std::move(results[i]));
        }
        
        // 更新总计数
        total_guesses += total;
        
        // 释放内存
        delete[] results;
        delete[] thread_counts;
    }
    else
    {
        string guess;
        int seg_idx = 0;
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
            {
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 2)
            {
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 3)
            {
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
            {
                break;
            }
        }

        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        if (pt.content[pt.content.size() - 1].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];
        }
        
        int total = pt.max_indices[pt.content.size() - 1];
        
        // 提高任务粒度阈值
        if (total < 100000) {
            // 直接使用串行处理
            for (int i = 0; i < total; i++) {
                guesses.push_back(guess + a->ordered_values[i]);
            }
            total_guesses += total;
            return;
        }
        
        // 使用更多的线程和更小的任务块
        int num_threads = 8; // 自动获取CPU核心数
        int optimal_chunk_size = 1000; // 每个任务处理1000个元素
        int num_chunks = (total + optimal_chunk_size - 1) / optimal_chunk_size;
        
        // 预分配结果数组
        guesses.reserve(guesses.size() + total); // 预先分配空间避免频繁扩容
        string* results = new string[total];
        int* thread_counts = new int[num_chunks];
        memset(thread_counts, 0, sizeof(int) * num_chunks);
        
        // 创建任务并提交到线程池
        for (int t = 0; t < num_chunks; ++t) {
            int start = t * optimal_chunk_size;
            int end = std::min((t + 1) * optimal_chunk_size, total);
            
            Task task = {
                start,
                end,
                guess,
                &a->ordered_values,
                results,
                &thread_counts[t]
            };
            
            global_thread_pool->Enqueue(task);
        }
        
        // 使用条件变量等待所有任务完成，而不是轮询
        global_thread_pool->WaitAll();
        
        // 批量添加结果 - 优化内存操作
        for (int i = 0; i < total; ++i) {
            guesses.push_back(std::move(results[i]));
        }
        
        // 更新总计数
        total_guesses += total;
        
        // 释放内存
        delete[] results;
        delete[] thread_counts;
    }
}
