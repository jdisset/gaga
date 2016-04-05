#define CATCH_CONFIG_MAIN
#include <random>
#include "../gaga.hpp"
#include "../json/json.hpp"
#include "Catch/single_include/catch.hpp"

std::default_random_engine rndEngine;
struct IntDNA {
	int value = 0;
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
	std::string toJSON() const {
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

template <typename T> void initGA() {
	GAGA::GA<T> ga(0, nullptr);
	ga.setEvaluator([](auto &i) { i.fitnesses["value"] = i.dna.value; });
	REQUIRE(ga.population.size() == 0);
	ga.setPopSize(400);
	ga.initPopulation([]() { return T::random(); });
	REQUIRE(ga.population.size() == 400);
	ga.step(10);
	REQUIRE(ga.population.size() == 400);
}
TEST_CASE("Population is initialized ok", "[population]") { initGA<IntDNA>(); }
