#include "../gaga.hpp"
#include "dna.hpp"
#include "grgen/common.h"
#include "grgen/grn.hpp"
#include "grgen/classic.hpp"
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

template <typename T> void GRNGA() {
	GAGA::GA<T> ga(0, nullptr);
	ga.setVerbosity(0);
	ga.setEvaluator([](auto &i) { i.fitnesses["length"] = i.dna.getProteinSize(ProteinType::regul); });
	REQUIRE( (ga.population.size() == 0) );
	ga.setPopSize(400);
  vector<GAGA::Individual<T>> pop;
  for (int i = 0; i < 400; ++i) {
    T t;
    t.addRandomProtein(ProteinType::input, "input");
    t.addRandomProtein(ProteinType::output, "output");
    t.randomReguls(1);
    t.randomParams();
    pop.push_back(GAGA::Individual<T>(t));
  }
	ga.setPopulation(pop);
	ga.step(10);
	REQUIRE(ga.population.size() == 400);
  /* TODO
  auto elites = ga.getElites(1);
  for (auto &e : elites) {
    for (auto &i : e.second) {
      REQUIRE(i.fitnesses.at(e.first) >= 1);
    }
  }
  */
}
TEST_CASE("Test with GRGEN GRN", "[population]") { GRNGA<GRN<Classic>>(); }
