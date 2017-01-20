#ifndef DNA_HPP
#define DNA_HPP
#include <random>
#include "../include/json.hpp"

struct IntDNA {
	int value = 0;
	std::default_random_engine rndEngine;
	std::uniform_int_distribution<int> distribution =
	    std::uniform_int_distribution<int>(0, 1000000);
	IntDNA() {}
	// A valid dna must be able to be constructed from a json string
	explicit IntDNA(const std::string &js) {
		auto o = nlohmann::json::parse(js);
		value = o["value"];
	}
	// It must have a mutate method
	void mutate() { value = distribution(rndEngine); }
	// A crossover method
	IntDNA crossover(const IntDNA &other) {
		std::uniform_int_distribution<int> dist = std::uniform_int_distribution<int>(0, 1);
		if (dist(rndEngine) == 0) return *this;
		return other;
	}
	// A reset method (just to cleanup things before a new evaluation)
	void reset() {}
	// And a method that returns a json string
	std::string serialize() const {
		nlohmann::json o;
		o["value"] = value;
		return o.dump(2);
	}
	// optional random init
	static IntDNA random() {
		IntDNA d;
		return d;
	}
};
#endif
