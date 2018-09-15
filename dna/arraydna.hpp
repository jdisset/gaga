#pragma once
#include <array>
#include <random>
#include "../include/json.hpp"
namespace GAGA {
template <typename T, size_t N> struct ArrayDNA {
	using json = nlohmann::json;
	std::array<T, N> values;
	ArrayDNA() {}

	ArrayDNA(const std::string& s) {
		auto o = json::parse(s);
		values = o.at("values");
	}

	std::string serialize() const {
		json o;
		o["values"] = values;
		return o.dump();
	}

	void reset() {}

	void mutate() {
		std::uniform_int_distribution<int> d5050(0, 1);
		std::uniform_int_distribution<int> dInt(0, N - 1);
		values[dInt(getRandomEngine())] = d5050(getRandomEngine()) ? 1 : 0;
	}

	static ArrayDNA random() {
		ArrayDNA res;
		std::uniform_int_distribution<int> d5050(0, 1);
		for (size_t i = 0; i < N; ++i) res.values[i] = d5050(getRandomEngine());
		return res;
	}

	static std::mt19937& getRandomEngine() {
		static std::mt19937 gen;
		return gen;
	}

	ArrayDNA crossover(const ArrayDNA& other) {  // uniform crossover
		ArrayDNA res;
		std::uniform_int_distribution<int> d5050(0, 1);
		for (size_t i = 0; i < N; ++i)
			res.values[i] = d5050(getRandomEngine()) ? values[i] : other.values[i];
		return res;
	}
};
}  // namespace GAGA
