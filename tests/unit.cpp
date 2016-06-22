#include "../gaga.hpp"
#include "catch/catch.hpp"
#include "dna.hpp"
#include "grgen/classic.hpp"
#include "grgen/common.h"
#include "grgen/grn.hpp"

template <typename T> void initGA() {
	GAGA::GA<T> ga(0, nullptr);
	ga.setVerbosity(0);
	ga.setEvaluator([](auto &i) { i.fitnesses["value"] = i.dna.value; });
	REQUIRE((ga.population.size() == 0));
	ga.setPopSize(400);
	ga.initPopulation([]() { return T::random(); });
	REQUIRE(ga.population.size() == 400);
	ga.step(10);
	REQUIRE(ga.population.size() == 400);
}
TEST_CASE("Population is initialized ok", "[population]") { initGA<IntDNA>(); }

template <typename T> void GRNGA() {
	GAGA::GA<T> ga(0, nullptr);
	size_t popsize = 100;
	ga.setVerbosity(0);
	ga.setEvaluator(
	    [](auto &i) { i.fitnesses["length"] = i.dna.getProteinSize(ProteinType::regul); });
	REQUIRE((ga.population.size() == 0));
	ga.setPopSize(popsize);
	vector<GAGA::Individual<T>> pop;
	for (size_t i = 0; i < popsize; ++i) {
		T t;
		t.config.ADD_RATE = 1.0;
		t.config.DEL_RATE = 0.0;
		t.config.MODIF_RATE = 0.0;
		t.addRandomProtein(ProteinType::input, "input");
		t.addRandomProtein(ProteinType::output, "output");
		t.randomReguls(1);
		t.randomParams();
		pop.push_back(GAGA::Individual<T>(t));
	}
	ga.setPopulation(pop);
	ga.step(5);
	REQUIRE(ga.population.size() == popsize);
}
TEST_CASE("Test with GRGEN GRN", "[population]") { GRNGA<GRN<Classic>>(); }
