#define GAGA_TESTING
#define CATCH_CONFIG_MAIN

#include "../extra/gagazmq/gagazmq.hpp"
#include "../gaga.hpp"
#include "catch/catch.hpp"
#include "dna.hpp"
#include "gagazmq_tests.hpp"
#include "sqlite_tests.hpp"

template <typename T> void initGA() {
	GAGA::GA<T> ga;
	ga.setVerbosity(0);
	ga.setMutationRate(0.7);
	ga.setCrossoverRate(0.3);
	ga.setEvaluator([](auto &i, int) { i.fitnesses["value"] = i.dna.value; });
	REQUIRE((ga.population.size() == 0));
	ga.setPopSize(200);
	ga.setNbThreads(10);
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

TEST_CASE("Pareto") {
	const int N = 50;
	GAGA::GA<IntDNA> ga;
	ga.setPopSize(N);
	ga.setVerbosity(0);
	int i = 0;
	ga.initPopulation([&]() {
		IntDNA d;
		d.value = i++;
		return d;
	});
	ga.setMutationRate(0);
	ga.setCrossoverRate(0);
	ga.setEvaluator([](auto &i, int) {
		i.fitnesses["value"] = i.dna.value;
		i.fitnesses["other"] = N - i.dna.value;
	});

	ga.evaluate();

	std::vector<typename GAGA::GA<IntDNA>::Ind_t *> indPtr;
	for (auto &i : ga.population) indPtr.push_back(&i);

	for (size_t i = 0; i < ga.population.size() - 1; ++i) {
		REQUIRE(ga.paretoDominates(ga.population[i + 1], ga.population[i], {{"value"}}));
		REQUIRE(!ga.paretoDominates(ga.population[i], ga.population[i + 1], {{"value"}}));

		REQUIRE(ga.paretoDominates(ga.population[i], ga.population[i + 1], {{"other"}}));
		REQUIRE(!ga.paretoDominates(ga.population[i + 1], ga.population[i], {{"other"}}));

		REQUIRE(ga.getParetoRank(indPtr, i, {{"other"}}) == i + 1);
		REQUIRE(ga.getParetoRank(indPtr, i, {{"value"}}) == N - i);

		std::unordered_set<std::string> objs;
		objs.insert("other");
		objs.insert("value");
		REQUIRE(!ga.paretoDominates(ga.population[i + 1], ga.population[i], objs));
		REQUIRE(!ga.paretoDominates(ga.population[i], ga.population[i + 1], objs));
		REQUIRE(ga.getParetoRank(indPtr, i, objs) == 1);
	}
}

void helpersMethods() {
	const int N = 50;
	GAGA::GA<IntDNA> ga;
	ga.setVerbosity(0);
	ga.setNbThreads(10);
	ga.setEvaluator([](auto &i, int) {
		i.fitnesses["value"] = i.dna.value;
		i.fitnesses["other"] = N - i.dna.value;
	});
	int i = 0;
	ga.setPopSize(N);
	ga.setMutationRate(0);
	ga.setCrossoverRate(0);
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
	auto offsprings =
	    ga.produceNOffsprings(2, ga.population, 0, ga.getAllObjectives(ga.population[0]));
	REQUIRE(offsprings.size() == 2);
	std::vector<GAGA::Individual<IntDNA> *> indPointers;
	for (size_t i = 0; i < ga.population.size(); ++i) {
		indPointers.push_back(&ga.population[i]);
	}
	auto newOffsprings =
	    ga.produceNOffsprings(2, indPointers, 0, ga.getAllObjectives(ga.population[0]));
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
