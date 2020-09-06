#ifndef TINYPOOL_HPP
#define TINYPOOL_HPP

#include <cassert>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <queue>
#include <shared_mutex>
#include <thread>
#include <utility>
#include <vector>

// simple thread pool
// mostly from https://www.youtube.com/watch?v=zULU6Hhp42w

namespace TinyPool {
using lock_t = std::unique_lock<std::mutex>;
using namespace std::chrono_literals;
struct notifQueue {
	std::deque<std::function<void(size_t)>> q;
	std::mutex mut;
	std::condition_variable ready;
	bool doneFlag{false};

	notifQueue() {}

	void done() {
		{
			lock_t lock{mut};
			doneFlag = true;
		}
		ready.notify_all();
	}

	bool tryPop(std::function<void(size_t)>& x) {
		lock_t lock{mut, std::try_to_lock};
		if (!lock || q.empty()) return false;
		x = std::move(q.front());
		q.pop_front();
		return true;
	}

	bool pop(std::function<void(size_t)>& x) {
		lock_t lock{mut};
		while (q.empty() && !doneFlag) ready.wait(lock);
		if (q.empty()) return false;
		x = std::move(q.front());
		q.pop_front();
		return true;
	}

	template <typename F> bool tryPush(F&& f) {
		{
			lock_t lock{mut, std::try_to_lock};
			if (!lock) return false;
			q.emplace_back(std::forward<F>(f));
		}
		ready.notify_one();
		return true;
	}

	template <typename F> void push(F&& f) {
		{
			lock_t lock{mut};
			q.emplace_back(std::forward<F>(f));
		}
		ready.notify_one();
	}
};

struct ThreadPool {
	size_t nThreads{1};
	size_t K = 10;
	std::vector<std::thread> threads;
	std::vector<notifQueue> queues;
	std::atomic<size_t> index{0};
	std::atomic<size_t> runningTasks{0};
	std::atomic<bool> shouldJoin{true}, waiterAvailable{false};
	std::condition_variable zeroTasks;
	std::mutex mut;
	std::unique_ptr<std::thread> waiter{nullptr};

	void loop(size_t i) {
		while (true) {
			std::function<void(size_t)> f;
			for (size_t n = 0; n < nThreads; ++n) {
				if (queues[(i + n) % nThreads].tryPop(f)) break;
			}
			if (!f && !queues[i].pop(f)) break;
			f(i);
			std::lock_guard<std::mutex> lck(mut);
			--runningTasks;
			zeroTasks.notify_all();
		}
	}

	void waitAll() {
		lock_t lock{mut};
		while (runningTasks > 0) zeroTasks.wait(lock);
	}

	ThreadPool(size_t n = 1) { reset(n); }

	void reset(size_t n) {
		for (auto& q : queues) q.done();
		for (auto& t : threads) t.join();
		nThreads = n;
		queues.clear();
		threads.clear();
		queues = std::vector<notifQueue>(n);
		for (size_t i = 0; i < n; ++i) threads.emplace_back([&, i] { loop(i); });
	}

	~ThreadPool() {
		for (auto& q : queues) q.done();
		for (auto& t : threads) t.join();
	}

	template <typename F>
	void autoChunksId_work(size_t l, size_t u, F&& f,
	                       double targetNbChunksPerThread = 2.0) {
		const size_t MIN_CHUNK_SIZE = 1;
		assert(l <= u);
		size_t vsize = u - l;
		double inc = static_cast<double>(vsize) /
		             (static_cast<double>(nThreads) * targetNbChunksPerThread);
		double acc = 0.0;
		size_t lbound = 0, ubound;
		do {
			ubound = std::min(
			    vsize, std::max(lbound + MIN_CHUNK_SIZE, static_cast<size_t>(acc + inc)));
			// fcapture = perfect capturing of f through a tuple:
			// allows to use std::move if f is an rvalue, or a reference if it's an lvalue.
			push_work([lbound, ubound,
			           fcapture = std::tuple<F>(std::forward<F>(f))](size_t threadId) {
				for (size_t i = lbound; i < ubound; ++i) std::get<0>(fcapture)(i, threadId);
			});
			lbound = ubound;
			acc += inc;
		} while (ubound < vsize);
	}

	template <typename F> void push_work(F&& f) {
		auto i = index++;
		runningTasks++;
		for (size_t n = 0; n < nThreads * K; ++n) {
			if (queues[(i + n) % nThreads].tryPush(std::forward<F>(f))) {
				return;
			}
		}
		queues[i % nThreads].push(std::forward<F>(f));
	}
};
}  // namespace TinyPool
#endif
