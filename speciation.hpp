#pragma once
#include <chrono>
#include <vector>

#include "gaga.hpp"

namespace GAGA {

// TODO:
// [ ] enable save of species infos to sql
// [ ] write example
// [ ] write tests
// [ ] make it work with minimization -> change worst fitness computations

template <typename GA> struct SpeciationExtension {
	using Ind_t = typename GA::Ind_t;
	using Iptr = typename GA::Iptr;
	using sig_t = typename Ind_t::sig_t;

	size_t targetSpeciesNumber = 10;   // how many species you want to aim for
	double speciationThreshold = 0.2;  // initial max distance btw two dna of same specie
	double minSpeciationThreshold = 0.03;
	double maxSpeciationThreshold = 0.5;
	double speciationThresholdIncrement = 0.03;  // for auto adjustment of the min distance

	// THE DISTANCE FUNCTION: takes 2 individuals and returns their genotype distance
	std::function<double(const Ind_t &, const Ind_t &)> indDistanceFunction =
	    [](const auto &, const auto &) { return 0.0; };

	std::vector<double> speciationThresholds;  // spec thresholds per specie
	std::vector<std::vector<Iptr>> species;    // pointers to the individuals of the species

	void onRegister(GA &gagaInstance) {
		gagaInstance.addPostEvaluationMethod(
		    [this](GA &ga) { updateNovelty(ga.population, ga); });
		gagaInstance.addPrintStartMethod([this](const GA &) { printStart(); });
	}

	void speciationNextGen(GA &ga) {
		// ---------------------------------------------------------
		// speciation nextGen method:
		// ---------------------------------------------------------
		// - select the new "leaders" of each species among previous species
		// - create the new population through intra-specie select + mutate + cross
		// - reform the species from the new population using genetic distance-to-leader
		// ( a new specie is created if distace < speciationThreshold)
		// - add the missing individuals through mutation from random species

		auto objectivesList = getEnabledObjectives(ga.population[0]);
		assert(ga.nbElites < minSpecieSize);
		if (species.size() == 0) {  // we put all the population in one species
			ga.printDbg("No species available. Creating one");
			species.resize(1);
			for (auto &i : ga.population) species[0].push_back(&i);
			speciationThresholds.clear();
			speciationThresholds.resize(1);
			speciationThresholds[0] = speciationThreshold;
		}
		assert(species.size() == speciationThresholds.size());

		std::vector<Ind_t> nextLeaders;  // New species leaders
		for (auto &s : species) {
			assert(s.size() > 0);
			std::uniform_int_distribution<size_t> d(0, s.size() - 1);
			nextLeaders.push_back(*s[d(ga.globalRand())]);
		}

		std::unordered_map<std::string, double> worstFitness;
		// First we want to offset all the adj fitnesses so they are in the positive range
		// we compute the lowest fitnesses for every objective
		for (const auto &o : objectivesList) {
			worstFitness[o] = std::numeric_limits<double>::max();
			for (const auto &i : ga.population)
				if (i.fitnesses.at(o) < worstFitness.at(o)) worstFitness[o] = i.fitnesses.at(o);
		}
		// We want the avg fitness per species (and per obj)
		std::vector<std::unordered_map<std::string, double>> perSpeciesAvg(species.size());
		std::unordered_map<std::string, double> perSpeciesAvg_sum;  // and the sum per obj
		for (const auto &o : objectivesList) {
			double total = 0;
			for (size_t i = 0; i < species.size(); ++i) {
				assert(species[i].size() > 0);
				double avg = 0;
				for (const auto &ind : species[i])
					avg += ind->fitnesses.at(o) - worstFitness.at(o) + 1;
				avg /= static_cast<double>(species[i].size());
				total += avg;
				perSpeciesAvg[i][o] = avg;
			}
			perSpeciesAvg_sum[o] = total;
		}
		if (ga.verbosity >= 3) {
			for (auto &af : perSpeciesAvg_sum)
				cerr << " - total \"" << af.first << "\" = " << af.second << std::endl;
		}

		std::vector<Ind_t> nextGen;  // creating the new population

		for (const auto &o : objectivesList) {
			assert(perSpeciesAvg_sum[o] != 0);
			for (size_t i = 0; i < species.size(); ++i) {
				auto &s = species[i];

				// nb of offsprings the specie is authorized to produce
				size_t nOffsprings =
				    static_cast<size_t>(static_cast<double>(ga.popSize) *
				                        (perSpeciesAvg[i][o] / perSpeciesAvg_sum[o]));

				nOffsprings /= static_cast<double>(objectivesList.size());  // per obj
				nOffsprings =
				    std::max(static_cast<size_t>(nOffsprings), 1ul);  // we create at least 1

				nextGen.reserve(nextGen.size() + nOffsprings);
				auto specieOffsprings = produceNOffsprings(nOffsprings, s, ga.nbElites);
				nextGen.insert(nextGen.end(), std::make_move_iterator(specieOffsprings.begin()),
				               std::make_move_iterator(specieOffsprings.end()));
			}
		}

		// correcting rounding errors by adding missing individuals
		while (nextGen.size() < ga.popSize) {  // we just add mutated leaders
			std::uniform_int_distribution<size_t> d(0, nextLeaders.size() - 1);
			nextGen.push_back(mutatedIndividual(nextLeaders[d(ga.globalRand())]));
		}
		while (nextGen.size() > ga.popSize) nextGen.pop_back();  // or delete the extra

		// creating new species
		species.clear();
		species.resize(nextLeaders.size());
		assert(species.size() > 0);

		for (auto &i : nextGen) {  // finding the closest leader
			size_t closestLeader = 0;
			double closestDist = std::numeric_limits<double>::max();
			bool foundSpecie = false;

			std::vector<double> distances(nextLeaders.size());

			std::vector<std::future<void>> futures;
			for (size_t l = 0; l < nextLeaders.size(); ++l)
				futures.push_back(ga.tp.push([=, &distances, &nextLeaders]() {
					distances[l] = indDistanceFunction(nextLeaders[l], i);
				}));
			for (auto &f : futures) f.get();

			for (size_t d = 0; d < distances.size(); ++d) {
				if (distances[d] < closestDist && distances[d] < speciationThreshold) {
					closestDist = distances[d];
					closestLeader = d;
					foundSpecie = true;
				}
			}
			if (foundSpecie) {  // we found your family
				species[closestLeader].push_back(&i);
			} else {                      // this one is too different, let's create a species
				nextLeaders.push_back(i);   // it becomes a leader
				species.push_back({{&i}});  // and of course a member of its species
				speciationThresholds.push_back(speciationThresholds.size() > 0 ?
				                                   speciationThresholds[0] :
				                                   speciationThreshold);
			}
		}

		ga.printDbg("Created the new species. Species size = ", species.size());

		assert(species.size() > 0);
		assert(species.size() == nextLeaders.size());
		assert(species.size() == speciationThresholds.size());
		for (const auto &s : species) assert(s.size() > 0);

		// adjusting speciation Thresholds
		size_t targetSpeciesSize =
		    static_cast<double>(ga.popSize) / static_cast<double>(targetSpeciesNumber);
		for (size_t i = 0; i < species.size(); ++i) {
			if (species[i].size() < targetSpeciesSize) {
				speciationThresholds[i] =
				    std::min(speciationThresholds[i] + speciationThresholdIncrement,
				             maxSpeciationThreshold);
			} else {
				speciationThresholds[i] =
				    std::max(speciationThresholds[i] - speciationThresholdIncrement,
				             minSpeciationThreshold);
			}
		}

		if (ga.verbosity >= 3) {
			cerr << "Speciation thresholds adjusted: " << std::endl;
			for (auto &s : speciationThresholds) {
				cerr << " " << s;
			}
			cerr << std::endl;
			cerr << "Species sizes : " << std::endl;
			for (auto &s : species) {
				cerr << " - " << s.size() << std::endl;
			}
		}

		savePopToPreviousGenerations(ga.population);
		ga.population = nextGen;
		setPopulationId(ga.population, ga.currentGeneration + 1);
	}

	void printStart() {
		std::cout << "  ▹ speciation is " << GAGA_COLOR_GREEN << "enabled"
		          << GAGA_COLOR_NORMAL << std::endl;
		std::cout << "    - targetSpeciesNumber = " << GAGA_COLOR_BLUE << targetSpeciesNumber
		          << GAGA_COLOR_NORMAL << std::endl;
		std::cout << "    - initial speciationThreshold = " << GAGA_COLOR_BLUE
		          << speciationThreshold << GAGA_COLOR_NORMAL << std::endl;
		std::cout << "    - speciationThresholdIncrement = " << GAGA_COLOR_BLUE
		          << speciationThresholdIncrement << GAGA_COLOR_NORMAL << std::endl;
		std::cout << "    - minSpeciationThreshold = " << GAGA_COLOR_BLUE
		          << minSpeciationThreshold << GAGA_COLOR_NORMAL << std::endl;
		std::cout << "    - maxSpeciationThreshold = " << GAGA_COLOR_BLUE
		          << maxSpeciationThreshold << GAGA_COLOR_NORMAL << std::endl;
	}
};
}  // namespace GAGA
