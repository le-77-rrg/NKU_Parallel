#include <iostream>
using namespace std;
#include <windows.h>

int n=10000;
int a[10001];
int b[10001][10001];
int sum[10001];

void initialize() {
    for (int i = 0;i < n;i++) {
        a[i] = i;
    }
    for (int i = 0;i < n;i++) {
        for (int j = 0;j < n;j++) {
            b[i][j] = i + j;
        }
    }

}

void normal_test() {
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER *) & freq);
    QueryPerformanceCounter((LARGE_INTEGER *) & head);
    for (int j = 0;j < n;j++) {
        sum[j] = 0;
        for (int i = 0;i < n;i++) {
            sum[j] += b[i][j] * a[i];
        }
    }
    QueryPerformanceCounter((LARGE_INTEGER *) & tail);
    cout << "normal:" << (tail - head) * 1000.0 / freq << "ms" << endl;

}

void optimize_test() {
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for (int i = 0;i < n;i++) {
        sum[i] = 0;
    }
    for (int i = 0;i < n;i++) {
        for (int j = 0;j < n;j++) {
            sum[j] += b[i][j] * a[i];
        }
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "optimize:" << (tail - head) * 1000.0 / freq << "ms" << endl;

}

int main()
{ 
        initialize();
        normal_test();
        optimize_test();
    return 0;
}


