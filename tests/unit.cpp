#include "../gaga.hpp"
#include "catch/catch.hpp"
#include "dna.hpp"
#include "grgen/classic.hpp"
#include "grgen/common.h"
#include "grgen/grn.hpp"

template <typename T> void initGA() {
	GAGA::GA<T> ga(0, nullptr);
	ga.setVerbosity(0);
	ga.setMutationProba(0.7);
	ga.setCrossoverProba(0.3);
	ga.setEvaluator([](auto &i) { i.fitnesses["value"] = i.dna.value; });
	REQUIRE((ga.population.size() == 0));
	ga.setPopSize(200);
	ga.initPopulation([]() { return T::random(); });
	ga.step(1);
	int initialBest = 0;
	for (const auto &i : ga.population)
		if (initialBest < i.dna.value) initialBest = i.dna.value;
	REQUIRE(ga.population.size() == 200);
	ga.step(50);
	int newBest = 0;
	for (const auto &i : ga.population)
		if (newBest < i.dna.value) newBest = i.dna.value;
	REQUIRE(ga.population.size() == 200);
	REQUIRE(newBest > initialBest);
}

TEST_CASE("Population is initialized ok, individuals are improving", "[population]") {
	initGA<IntDNA>();
}

void helpersMethods() {
	const int N = 50;
	GAGA::GA<IntDNA> ga(0, nullptr);
	ga.setVerbosity(0);
	ga.setEvaluator([](auto &i) {
		i.fitnesses["value"] = i.dna.value;
		i.fitnesses["other"] = N - i.dna.value;
	});
	int i = 0;
	ga.setPopSize(N);
	ga.setMutationProba(0);
	ga.setCrossoverProba(0);
	ga.initPopulation([&]() {
		IntDNA d;
		d.value = i++;
		return d;
	});
	REQUIRE(i == N);
	REQUIRE(ga.population[0].dna.value == 0);
	REQUIRE(ga.population[49].dna.value == N - 1);
	ga.evaluate();

	// Produce N Offsprings
	auto offsprings = ga.produceNOffsprings(2, ga.population);
	REQUIRE(offsprings.size() == 2);
	std::vector<GAGA::Individual<IntDNA> *> indPointers;
	for (size_t i = 0; i < ga.population.size(); ++i) {
		indPointers.push_back(&ga.population[i]);
	}
	auto newOffsprings = ga.produceNOffsprings(2, indPointers);
	REQUIRE(newOffsprings.size() == 2);

	// ELITISM
	// -> from main population
	auto elites = ga.getElites(1);
	REQUIRE(elites.size() == 2);
	REQUIRE(elites.count("value"));
	REQUIRE(elites.count("other"));
	REQUIRE(elites["value"].size() == 1);
	REQUIRE(elites["other"].size() == 1);
	REQUIRE(elites["value"][0].dna.value == N - 1);
	REQUIRE(elites["other"][0].dna.value == 0);
	elites = ga.getElites(2);
	REQUIRE(elites.size() == 2);
	REQUIRE(elites.count("value"));
	REQUIRE(elites.count("other"));
	REQUIRE(elites["value"].size() == 2);
	REQUIRE(elites["other"].size() == 2);
	REQUIRE(elites["value"][0].dna.value != elites["value"][1].dna.value);
	REQUIRE(elites["value"][0].dna.value + elites["value"][1].dna.value == 2 * N - 3);
	REQUIRE(elites["other"][0].dna.value != elites["other"][1].dna.value);
	REQUIRE(elites["other"][0].dna.value * elites["other"][1].dna.value == 0);
	REQUIRE(elites["other"][0].dna.value + elites["other"][1].dna.value == 1);
	// -> from sub population pointers
	std::vector<GAGA::Individual<IntDNA> *> subPop;
	for (size_t c = 0; c < 10; ++c) subPop.push_back(&ga.population[c]);
	elites = ga.getElites(1, subPop);
	REQUIRE(elites.size() == 2);
	REQUIRE(elites.count("value"));
	REQUIRE(elites.count("other"));
	REQUIRE(elites["value"].size() == 1);
	REQUIRE(elites["other"].size() == 1);
	REQUIRE(elites["value"][0].dna.value == 9);
	REQUIRE(elites["other"][0].dna.value == 0);
}

TEST_CASE("Helpers methods ok", "[methods]") { helpersMethods(); }

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
