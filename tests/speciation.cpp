#include "../gaga.hpp"
#include "catch/catch.hpp"
#include "dna.hpp"

template <typename T> void speciationGA() {
	//const size_t popSize = 30;
	//GAGA::GA<T> ga(0, nullptr);
	//ga.setVerbosity(3);
	//ga.setEvaluator([](auto &i) { i.fitnesses["value"] = i.dna.value; });
	//ga.enableSpeciation();

	//REQUIRE((ga.population.size() == 0));
	//ga.setPopSize(popSize);
	//ga.initPopulation([]() { return T::random(); });
	//REQUIRE(ga.population.size() == popSize);
	//ga.step(10);
	//REQUIRE(ga.population.size() == popSize);
}
TEST_CASE("Classic optimization with speciation enabled", "[population]") {
	speciationGA<IntDNA>();
}
