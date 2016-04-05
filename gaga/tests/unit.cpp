#define CATCH_CONFIG_MAIN
#include "../gaga.hpp"
#include "../json/json.hpp"
#include "dna.hpp"
#include "Catch/single_include/catch.hpp"

template <typename T> void initGA() {
	GAGA::GA<T> ga(0, nullptr);
	ga.setVerbosity(0);
	ga.setEvaluator([](auto &i) { i.fitnesses["value"] = i.dna.value; });
	REQUIRE( (ga.population.size() == 0) );
	ga.setPopSize(400);
	ga.initPopulation([]() { return T::random(); });
	REQUIRE(ga.population.size() == 400);
	ga.step(10);
	REQUIRE(ga.population.size() == 400);
}
TEST_CASE("Population is initialized ok", "[population]") { initGA<IntDNA>(); }
