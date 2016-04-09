#include "../gaga.hpp"
#include "dna.hpp"
#include "catch/catch.hpp"

template <typename T> void paretoGA() {
	GAGA::GA<T> ga(0, nullptr);
	ga.setVerbosity(0);
	ga.setEvaluator([](auto &i) { i.fitnesses["value"] = i.dna.value; i.fitnesses["second"] = 0; });
	REQUIRE( (ga.population.size() == 0) );
	ga.setPopSize(400);
	ga.initPopulation([]() { return T::random(); });
	REQUIRE(ga.population.size() == 400);
	ga.step(10);
	REQUIRE(ga.population.size() == 400);
}
TEST_CASE("Pareto multi-objective optimization", "[population]") { paretoGA<IntDNA>(); }
