#include<iostream>
using namespace std;
#include <windows.h>
#include<chrono>

long long int N;
unsigned long long int a[135000000];


void initialize() {
	for (int i = 0;i < N;i++) {
		a[i] = i;
	}
}

void normal_test() {
	long long head, tail, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	unsigned long long int sum = 0;
	for (int i = 0;i < N;i++) {
		sum += a[i];
	}

	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "normal:" << (tail - head) * 1000.0 / freq << "ms" << endl;
}

void optimize_test1() {
	long long head, tail, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	unsigned long long int sum = 0;
	unsigned long long int sum1 = 0;
	unsigned long long int sum2 = 0;
	for (int i = 0;i < N-1;i += 2) {
		sum1 += a[i];
		//if (i + 1 < N) {
		sum2 += a[i + 1];
		//}
	}
	sum = sum1 + sum2;
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "optimize_1:" << (tail - head) * 1000.0 / freq << "ms" << endl;

}


void unroll() {
	long long head, tail, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	unsigned long long int sum = 0;
	unsigned long long int sum1 = 0;
	unsigned long long int sum2 = 0;
	unsigned long long int sum3 = 0;
	unsigned long long int sum4 = 0;
	for (int i = 0;i < N-3;i += 4) {
		sum1 += a[i];
		//if (i + 1 < N) {
		sum2 += a[i + 1];
		//}
		sum3 += a[i + 2];
		sum4 += a[i + 3];
	}
	sum = sum1 + sum2+sum3+sum4;
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "unroll:" << (tail - head) * 1000.0 / freq << "ms" << endl;

}

void recursion(int n)
{
	if (n == 1)
		return;
	else
	{
		for (int i = 0; i < n / 2; i++)
			a[i] += a[n - i - 1];
		n = n / 2;
		recursion(n);
	}
}

void optimize_test2() {
	long long head, tail, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	unsigned long long int sum = 0;
	recursion(N);
	sum = a[0];
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "optimize_2:" << (tail - head) * 1000.0 / freq << "ms" << endl;
}

void optimize_test3() {
	long long head, tail, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	unsigned long long int sum = 0;
	for (int m = N;m > 1;m /= 2) {
		for (int i = 0;i < m / 2;i++) {
			a[i] = a[2 * i] + a[2 * i + 1];
		}
	}
	sum = a[0];
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "optimize_3:" << (tail - head) * 1000.0 / freq << "ms" << endl;


}



int main() {
	while (cin >> N) {
		N = pow(2, N);
		initialize();
		normal_test();
		optimize_test1();
		//optimize_test2();
		//initialize();
		//optimize_test3();
		unroll();
	}
	return 0;
}



