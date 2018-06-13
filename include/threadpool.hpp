#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>
#ifdef _WIN32
#include "mingw/mingw.condition_variable.h"
#include "mingw/mingw.mutex.h"
#include "mingw/mingw.thread.h"
#endif

/**
 * @brief Simple threadpool, largely inspired by github.com/progschj/ThreadPool
 */

class ThreadPool {
 private:
	size_t nthreads = 0;
	std::vector<std::thread> workers;
	std::queue<std::function<void()>> tasks;
	std::mutex queue_mutex;
	std::condition_variable condition, waitingLast;
	bool stop = false;
	std::atomic<int> currentTasks;

 public:
	size_t getNbThreads() { return nthreads; }
	ThreadPool(size_t nt) : stop(false), currentTasks(0) { setNbThreads(nt); }

	void setNbThreads(size_t n) {
		halt();
		workers.clear();
		nthreads = n;
		for (size_t i = 0; i < nthreads; ++i) {
			workers.emplace_back([this] {
				for (;;) {
					std::unique_lock<std::mutex> qlock(this->queue_mutex);
					this->condition.wait(qlock,
					                     [this] { return this->stop || !this->tasks.empty(); });
					if (this->stop && this->tasks.empty()) return;
					auto task(std::move(this->tasks.front()));
					this->tasks.pop();
					qlock.unlock();
					task();
					qlock.lock();
					--currentTasks;
					if (currentTasks == 0) waitingLast.notify_all();
				}
			});
		}
	}

	void waitUntilLast() {
		std::unique_lock<std::mutex> wlock(this->queue_mutex);
		while (currentTasks > 0)
			this->waitingLast.wait(wlock, [this]() { return currentTasks == 0; });
	}

	template <class F> auto enqueue(F&& f) {
		if (nthreads > 0) {
			std::unique_lock<std::mutex> lock(queue_mutex);
			tasks.emplace([func = std::forward<F>(f)]() { std::move(func)(); });
			++currentTasks;
			condition.notify_one();
		} else
			std::forward<F>(f)();
	}

	template <class F> auto enqueueWithFuture(F&& f) {
		using R = decltype(f());
		if (nthreads > 0) {
			++currentTasks;
			if (stop) throw std::runtime_error("enqueue on stopped ThreadPool");
			auto taskPtr = new std::packaged_task<R()>(
			    [func = std::forward<F>(f)]() -> R { return std::move(func)(); });
			std::unique_lock<std::mutex> lock(queue_mutex);
			tasks.emplace([taskPtr]() {
				(*taskPtr)();
				delete taskPtr;
			});
			condition.notify_one();
			return taskPtr->get_future();
		} else {
			std::packaged_task<R()> task(
			    [func = std::forward<F>(f)]() -> R { return std::move(func)(); });
			task();
			return task.get_future();
		}
	}
	template <typename Container, typename F>
	void autoChunks(Container& v, size_t minChunkSize, double avgTasksPerThread, F f) {
		if (nthreads > 0 && v.size() > 2 * minChunkSize) {
			size_t chunkSize = std::max(
			    minChunkSize, static_cast<size_t>(static_cast<double>(v.size()) /
			                                      (static_cast<double>(nthreads) *
			                                       static_cast<double>(avgTasksPerThread))));
			auto prevIt = v.begin();
			auto nextIt = v.begin();
			size_t prevId = 0;
			size_t nextId = 0;
			do {
				nextId = std::min(prevId + chunkSize, v.size());
				nextIt = std::next(prevIt, static_cast<long>(nextId) - static_cast<long>(prevId));
				enqueue([prevIt, nextIt, f, &v]() {
					for (auto i = prevIt; i != nextIt; ++i) f(*i);
				});
				prevId = nextId;
				prevIt = nextIt;
			} while (nextId < v.size());
		} else {
			for (auto& e : v) f(e);
		}
	}

	void halt() {
		{
			std::unique_lock<std::mutex> lock(queue_mutex);
			stop = true;
		}
		condition.notify_all();
		for (std::thread& worker : workers) worker.join();
	}
	~ThreadPool() { halt(); }
};

#endif
