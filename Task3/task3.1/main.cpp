#include <iostream>
#include <vector>
#include <thread>
#include <future>
#include <chrono>
#include <queue>
#include <functional>
#include <atomic>
#include <condition_variable>

class ThreadPool {
public:
    ThreadPool(size_t threads);
    ~ThreadPool();

    template <class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>;

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;

    std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop;
};

ThreadPool::ThreadPool(size_t threads) : stop(false) {
    for (size_t i = 0; i < threads; ++i)
        workers.emplace_back([this] {
            for (;;) {
                std::function<void()> task;

                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    this->condition.wait(lock, [this] { return this->stop.load() || !this->tasks.empty(); });
                    if (this->stop.load() && this->tasks.empty())
                        return;
                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                }

                task();
            }
        });
}

ThreadPool::~ThreadPool() {
    stop.store(true);
    condition.notify_all();
    for (std::thread& worker : workers)
        worker.join();
}

template <class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        if (stop.load())
            throw std::runtime_error("enqueue on stopped ThreadPool");
        tasks.emplace([task]() { (*task)(); });
    }
    condition.notify_one();
    return res;
}

std::chrono::duration<double> Multiplication(int N, int threadAmount) {
    ThreadPool pool(threadAmount);
    std::vector<double> matrix(N * N);
    std::vector<double> vector(N);

    int itemsPerThread = N / threadAmount;
    std::vector<std::future<void>> results;

    for (int i = 0; i < threadAmount; ++i) { // инициализация
        int lb = i * itemsPerThread;
        int ub = (i == threadAmount - 1) ? (N - 1) : (lb + itemsPerThread - 1);
        results.emplace_back(
            pool.enqueue([lb, ub, &matrix, &vector, N]() {
                for (int i = lb; i <= ub; ++i) {
                    for (int j = 0; j < N; ++j)
                        matrix[i * N + j] = i + j;
                    vector[i] = i;
                }
            })
        );
    }

    for (auto&& result : results)
        result.get();

    std::vector<double> resultVector(N, 0);
    results.clear();
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < threadAmount; ++i) {
        int lb = i * itemsPerThread;
        int ub = (i == threadAmount - 1) ? (N - 1) : (lb + itemsPerThread - 1);
        results.emplace_back(
            pool.enqueue([lb, ub, &matrix, &vector, &resultVector, N]() {
                for (int i = lb; i <= ub; ++i) {
                    for (int j = 0; j < N; ++j) {
                        resultVector[i] += matrix[i * N + j] * vector[j];
                    }
                }
            })
        );
    }

    for (auto&& result : results)
        result.get();
    const auto end = std::chrono::steady_clock::now();

    return (end - start);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size> <num_threads>" << std::endl;
        return 1;
    }

    int N = atoi(argv[1]);
    int threadAmount = atoi(argv[2]);
    auto duration = Multiplication(N, threadAmount);
    std::cout << std::fixed << duration.count() << " seconds" << std::endl;

    return 0;
}
