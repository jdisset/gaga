#ifndef COMMON_H
#define COMMON_H
#include <chrono>
#include <iostream>
#include <random>
#include "json/json.hpp"

#define INIT_CONCENTRATION 0.5

static std::default_random_engine grnRand = std::default_random_engine(
    static_cast<unsigned int>(
        std::chrono::system_clock::now().time_since_epoch().count()) +
    std::random_device()());
enum class ProteinType { input = 0u, regul = 1u, output = 2u };
template <typename T> T mix(const T& a, const T& b, const double v) {
	double r = v > 1.0 ? 1.0 : (v < 0.0 ? 0.0 : v);
	return (a * (1.0 - r)) + (r * b);
}

template <typename E> inline static constexpr unsigned int to_underlying(E e) {
	return static_cast<unsigned int>(e);
}

template <typename P, typename C>
inline void eachP(const std::initializer_list<ProteinType>& l, C& container,
                  const std::function<void(P&)>& f) {
	for (const auto& ptype : l) {
		const auto t = to_underlying(ptype);
		for (auto& p : container[t]) {
			f(p.second);
		}
	}
}

template <typename P, typename C>
inline void eachP(const std::initializer_list<ProteinType>& l, C& container,
                  const std::function<void(P&, size_t pt)>& f) {
	for (const auto& ptype : l) {
		const auto t = to_underlying(ptype);
		for (auto& p : container[t]) {
			f(p.second, t);
		}
	}
}

template <typename I, typename T, unsigned int maxn> struct stackUmap {
	std::array<I, maxn> indices;
	std::array<T, maxn> values;
	size_t size;
};

template <typename T> const T& cref(T* t) { return *t; }  // turn ptr into ref
template <typename T> const T& cref(T& t) { return t; }   // cref, charogne !
#endif
