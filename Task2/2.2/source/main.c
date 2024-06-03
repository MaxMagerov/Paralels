#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <stdlib.h>

const double PI = 3.14159265358979323846;
const double a = -4.0;
const double b = 4.0;
const int nsteps = 40000000;

// Функция для получения текущего времени в секундах
double cpuSecond() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double) ts.tv_sec + (double) ts.tv_nsec * 1.e-9);
}

// Интегрируемая функция
double func(double x) {
    return exp(-x * x);
}

// Последовательная функция интегрирования
double integrate(double (*func)(double), double a, double b, int n) {
    double h = (b - a) / n;
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += func(a + h * (i + 0.5));
    }
    sum *= h;
    return sum;
}

// Параллельная функция интегрирования с использованием OpenMP
double integrate_omp(double (*func)(double), double a, double b, int n, int threads) {
    double h = (b - a) / n;
    double sum = 0.0;

#pragma omp parallel num_threads(threads)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = n / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);
        double sumlock = 0.0;
        for (int i = lb; i <= ub; i++) {
            sumlock += func(a + h * (i + 0.5));
        }
#pragma omp atomic
        sum += sumlock;
    }
    sum *= h;
    return sum;
}

// Функция для запуска последовательного интегрирования
double run_serial() {
    double t = cpuSecond();
    double res = integrate(func, a, b, nsteps);
    t = cpuSecond() - t;
    printf("Result (serial): %.12f; error %.12f\n", res, fabs(res - sqrt(PI)));
    return t;
}

// Функция для запуска параллельного интегрирования
double run_parallel(int threads) {
    double t = cpuSecond();
    double res = integrate_omp(func, a, b, nsteps, threads);
    t = cpuSecond() - t;
    printf("Result (parallel): %.12f; error %.12f\n", res, fabs(res - sqrt(PI)));
    return t;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s <number_of_threads>\n", argv[0]);
        return 1;
    }

    int threads = atoi(argv[1]);
    printf("Integration f(x) on [%.12f, %.12f], nsteps = %d\n", a, b, nsteps);

    double tserial = run_serial();
    double tparallel = run_parallel(threads);

    printf("Execution time (serial): %.6f\n", tserial);
    printf("Execution time (parallel): %.6f\n", tparallel);
    printf("Speedup: %.2f\n", tserial / tparallel);

    return 0;
}