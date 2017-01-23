// Gaga: lightweight simple genetic algorithm library
// Copyright (c) Jean Disset 2016, All rights reserved.

// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library.

#ifndef GAMULTI_HPP
#define GAMULTI_HPP

/****************************************
 *       TO ENABLE PARALLELISATION
 * *************************************/
// before including this file,
// #define OMP if you want OpenMP parallelisation
// #define CLUSTER if you want MPI parralelisation
// #define CLUSTER if you want MPI parralelisation
#ifdef CLUSTER
#include <mpi.h>
#include <cstring>
#endif
#ifdef OMP
#include <omp.h>
#endif

#include <assert.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "include/json.hpp"

#define PURPLE "\033[35m"
#define PURPLEBOLD "\033[1;35m"
#define BLUE "\033[34m"
#define BLUEBOLD "\033[1;34m"
#define GREY "\033[30m"
#define GREYBOLD "\033[1;30m"
#define YELLOW "\033[33m"
#define YELLOWBOLD "\033[1;33m"
#define RED "\033[31m"
#define REDBOLD "\033[1;31m"
#define CYAN "\033[36m"
#define CYANBOLD "\033[1;36m"
#define GREEN "\033[32m"
#define GREENBOLD "\033[1;32m"
#define NORMAL "\033[0m"

namespace GAGA {

using std::vector;
using std::string;
using std::unordered_set;
using std::map;
using std::unordered_map;
using std::cout;
using std::cerr;
using std::endl;
using fpType = std::vector<std::vector<double>>;  // footprints for novelty
using json = nlohmann::json;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using std::chrono::system_clock;

/******************************************************************************************
 *                                 GAGA LIBRARY
 *****************************************************************************************/
// This file contains :
// 1 - the Individual class template : an individual's generic representation, with its
// dna, fitnesses and behavior footprints (for novelty)
// 2 - the main GA class template

// About parallelisation :
// before including this file,
// #define OMP if you want OpenMP parallelisation
// #define CLUSTER if you want MPI parallelisation

/*****************************************************************************
 *                         INDIVIDUAL CLASS
 * **************************************************************************/
// A valid DNA class must have (see examples folder):
// DNA mutate()
// DNA crossover(DNA& other)
// static DNA random(int argc, char** argv)
// json& constructor
// void reset()
// json toJson()

template <typename DNA> struct Individual {
	DNA dna;
	map<string, double> fitnesses;  // map {"fitnessCriterName" -> "fitnessValue"}
	fpType footprint;               // individual's footprint for novelty computation
	string infos;                   // custom infos, description, whatever...
	bool evaluated = false;
	bool wasAlreadyEvaluated = false;
	double evalTime = 0.0;

	Individual() {}
	explicit Individual(const DNA &d) : dna(d) {}

	explicit Individual(const json &o) {
		assert(o.count("dna"));
		dna = DNA(o.at("dna").dump());
		if (o.count("footprint")) footprint = o.at("footprint").get<fpType>();
		if (o.count("fitnesses")) fitnesses = o.at("fitnesses").get<decltype(fitnesses)>();
		if (o.count("infos")) infos = o.at("infos");
		if (o.count("evaluated")) evaluated = o.at("evaluated");
		if (o.count("alreadyEval")) wasAlreadyEvaluated = o.at("alreadyEval");
		if (o.count("evalTime")) evalTime = o.at("evalTime");
	}

	// Exports individual to json
	json toJSON() const {
		json o;
		o["dna"] = dna.serialize();
		o["fitnesses"] = fitnesses;
		o["footprint"] = footprint;
		o["infos"] = infos;
		o["evaluated"] = evaluated;
		o["alreadyEval"] = wasAlreadyEvaluated;
		o["evalTime"] = evalTime;
		return o;
	}

	// Exports a vector of individual to json
	static json popToJSON(const vector<Individual<DNA>> &p) {
		json o;
		json popArray;
		for (auto &i : p) popArray.push_back(i.toJSON());
		o["population"] = popArray;
		return o;
	}

	// Loads a vector of individual from json
	static vector<Individual<DNA>> loadPopFromJSON(const json &o) {
		assert(o.count("population"));
		vector<Individual<DNA>> res;
		json popArray = o.at("population");
		for (auto &ind : popArray) res.push_back(Individual<DNA>(ind));
		return res;
	}
};

/*********************************************************************************
 *                                 GA CLASS
 ********************************************************************************/
// DNA requirements : see Individual class;
//
// Evaluaor class requirements (see examples folder):
// constructor(int argc, char** argv)
// void operator()(const Individual<DNA>& ind)
// const string name
//
// TYPICAL USAGE :
//
// GA<DNAType, EvalType> ga;
// ga.setPopSize(400);
// return ga.start();

enum class SelectionMethod { paretoTournament, randomObjTournament };
template <typename DNA> class GA {
 protected:
	/*********************************************************************************
	 *                            MAIN GA SETTINGS
	 ********************************************************************************/
	unsigned int verbosity = 2;           // 0 = silent; 1 = generations stats;
	                                      // 2 = individuals stats; 3 = everything
	size_t popSize = 500;                 // nb of individuals in the population
	size_t nbElites = 1;                  // nb of elites to keep accross generations
	size_t nbSavedElites = 1;             // nb of elites to save
	size_t tournamentSize = 3;            // nb of competitors in tournament
	bool savePopEnabled = true;           // save the whole population
	unsigned int savePopInterval = 1;     // interval between 2 whole population saves
	unsigned int saveGenInterval = 1;     // interval between 2 elites/pareto saves
	string folder = "../evos/";           // where to save the results
	string evaluatorName;                 // name of the given evaluator func
	double crossoverProba = 0.2;          // crossover probability
	double mutationProba = 0.5;           // mutation probablility
	bool evaluateAllIndividuals = false;  // force evaluation of every individual
	bool doSaveParetoFront = false;       // save the pareto front
	bool doSaveGenStats = true;           // save generations stats to csv file
	bool doSaveIndStats = false;          // save individuals stats to csv file
	SelectionMethod selecMethod = SelectionMethod::paretoTournament;

	// for novelty:
	bool novelty = false;             // enable novelty
	double minNoveltyForArchive = 1;  // min novelty for being added to the general archive
	size_t KNN = 15;                  // size of the neighbourhood for novelty
	bool saveArchiveEnabled = true;   // save the novelty archive

	// for speciation:
	bool speciation = false;           // enable speciation
	double speciationThreshold = 0.2;  // min distance between two dna of same specie
	size_t minSpecieSize = 15;         // minimum specie size
	double minSpeciationThreshold = 0.03;
	double maxSpeciationThreshold = 0.5;
	double speciationThresholdIncrement = 0.01;
	std::function<double(const Individual<DNA> &, const Individual<DNA> &)>
	    indDistanceFunction = [](const auto &, const auto &) { return 0.0; };
	const unsigned int MAX_SPECIATION_TRIES = 100;
	vector<double> speciationThresholds;  // spec thresholds per specie

	/********************************************************************************
	 *                                 SETTERS
	 ********************************************************************************/
 public:
	using Iptr = Individual<DNA> *;
	using DNA_t = DNA;
	void enablePopulationSave() { savePopEnabled = true; }
	void disablePopulationSave() { savePopEnabled = false; }
	void enableArchiveSave() { saveArchiveEnabled = true; }
	void disableArchiveSave() { saveArchiveEnabled = false; }
	void setVerbosity(unsigned int lvl) { verbosity = lvl <= 3 ? (lvl >= 0 ? lvl : 0) : 3; }
	void setPopSize(size_t s) { popSize = s; }
	void setNbElites(size_t n) { nbElites = n; }
	void setNbSavedElites(size_t n) { nbSavedElites = n; }
	void setTournamentSize(size_t n) { tournamentSize = n; }
	void setPopSaveInterval(unsigned int n) { savePopInterval = n; }
	void setGenSaveInterval(unsigned int n) { saveGenInterval = n; }
	void setSaveFolder(string s) { folder = s; }
	void setCrossoverProba(double p) {
		crossoverProba = p <= 1.0 ? (p >= 0.0 ? p : 0.0) : 1.0;
	}
	void setMutationProba(double p) {
		mutationProba = p <= 1.0 ? (p >= 0.0 ? p : 0.0) : 1.0;
	}
	void setEvaluator(std::function<void(Individual<DNA> &)> e,
	                  std::string ename = "anonymousEvaluator") {
		evaluator = e;
		evaluatorName = ename;
		for (auto &i : population) {
			i.evaluated = false;
			i.wasAlreadyEvaluated = false;
		}
	}

	void setNewGenerationFunction(std::function<void(void)> f) {
		newGenerationFunction = f;
	}  // called before evaluating the current population

	void setIsBetterMethod(std::function<bool(double, double)> f) { isBetter = f; }
	void setSelectionMethod(const SelectionMethod &sm) { selecMethod = sm; }

	template <typename S> std::function<Individual<DNA> *(S &)> getSelectionMethod() {
		switch (selecMethod) {
			case SelectionMethod::paretoTournament:
				return [this](S &subPop) { return paretoTournament(subPop); };
			case SelectionMethod::randomObjTournament:
			default:
				return [this](S &subPop) { return randomObjTournament(subPop); };
		}
	}

	void setEvaluateAllIndividuals(bool m) { evaluateAllIndividuals = m; }
	void setSaveParetoFront(bool m) { doSaveParetoFront = m; }
	void setSaveGenStats(bool m) { doSaveGenStats = m; }
	void setSaveIndStats(bool m) { doSaveIndStats = m; }

	// main current and previous population containers
	vector<Individual<DNA>> population;
	vector<Individual<DNA>> lastGen;

	// for novelty:
	void enableNovelty() { novelty = true; }
	void disableNovelty() { novelty = false; }
	void setKNN(size_t n) { KNN = n; }
	void setMinNoveltyForArchive(double m) { minNoveltyForArchive = m; }

	// for speciation:
	void enableSpeciation() {
		nextGeneration = [this]() { speciationNextGen(); };
		speciation = true;
	}
	void disableSpeciation() {
		nextGeneration = [this]() { classicNextGen(); };
		speciation = false;
	}
	void setMinSpeciationThreshold(double s) { minSpeciationThreshold = s; }
	void setMaxSpeciationThreshold(double s) { maxSpeciationThreshold = s; }
	void setSpeciationThreshold(double s) { speciationThreshold = s; }
	void setSpeciationThresholdIncrement(double s) { speciationThresholdIncrement = s; }
	void setMinSpecieSize(double s) { minSpecieSize = s; }
	void setIndDistanceFunction(
	    std::function<double(const Individual<DNA> &, const Individual<DNA> &)> f) {
		indDistanceFunction = f;
	}
	vector<vector<Iptr>> species;  // pointers to the individuals of the species

	////////////////////////////////////////////////////////////////////////////////////

 protected:
	vector<Individual<DNA>>
	    archive;  // when novelty is enabled, we store the novel individuals there
	size_t currentGeneration = 0;
	bool customInit = false;
	// openmp/mpi stuff
	int procId = 0;
	int nbProcs = 1;
	int argc = 1;
	char **argv = nullptr;

	std::vector<std::map<std::string, std::map<std::string, double>>> genStats;

	std::random_device rd;
	std::default_random_engine globalRand = std::default_random_engine(rd());

	std::function<void(Individual<DNA> &)> evaluator;
	std::function<void(void)> newGenerationFunction = []() {};
	std::function<void(void)> nextGeneration = [this]() { classicNextGen(); };
	std::function<bool(double, double)> isBetter = [](double a, double b) { return a > b; };

	// returns a reference (transforms pointer into reference)
	template <typename T> inline T &ref(T &obj) { return obj; }
	template <typename T> inline T &ref(T *obj) { return *obj; }
	template <typename T> inline const T &ref(const T &obj) { return obj; }
	template <typename T> inline const T &ref(const T *obj) { return *obj; }

 public:
	/*********************************************************************************
	 *                              CONSTRUCTOR
	 ********************************************************************************/
	GA(int ac, char **av) : argc(ac), argv(av) {
		setSelectionMethod(selecMethod);
#ifdef CLUSTER
		MPI_Init(&argc, &argv);
		MPI_Comm_size(MPI_COMM_WORLD, &nbProcs);
		MPI_Comm_rank(MPI_COMM_WORLD, &procId);
		if (procId == 0) {
			if (verbosity >= 3) {
				std::cout << "   -------------------" << endl;
				std::cout << CYAN << " MPI STARTED WITH " << NORMAL << nbProcs << CYAN
				          << " PROCS " << NORMAL << endl;
				std::cout << "   -------------------" << endl;
				std::cout << "Initialising population in master process" << endl;
			}
		}
#endif
	}

	/*********************************************************************************
	 *                          START THE BOUZIN
	 ********************************************************************************/
	void setPopulation(const vector<Individual<DNA>> &p) {
		if (procId == 0) {
			population = p;
			if (population.size() != popSize)
				throw std::invalid_argument("Population doesn't match the popSize param");
			popSize = population.size();
		}
	}

	void initPopulation(const std::function<DNA()> &f) {
		if (procId == 0) {
			population.clear();
			population.reserve(popSize);
			for (size_t i = 0; i < popSize; ++i) {
				population.push_back(Individual<DNA>(f()));
				population[population.size() - 1].evaluated = false;
			}
		}
	}

	void evaluate() {
#ifdef OMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
		for (size_t i = 0; i < population.size(); ++i) {
			if (evaluateAllIndividuals || !population[i].evaluated) {
				auto t0 = high_resolution_clock::now();
				population[i].dna.reset();
				evaluator(population[i]);
				auto t1 = high_resolution_clock::now();
				population[i].evaluated = true;
				double indTime = std::chrono::duration<double>(t1 - t0).count();
				population[i].evalTime = indTime;
				population[i].wasAlreadyEvaluated = false;
			} else {
				population[i].evalTime = 0.0;
				population[i].wasAlreadyEvaluated = true;
			}
			if (verbosity >= 2) printIndividualStats(population[i]);
		}
	}

	// "Vroum vroum"
	void step(int nbGeneration = 1) {
		if (!evaluator) throw std::invalid_argument("No evaluator specified");
		if (currentGeneration == 0 && procId == 0) {
			createFolder(folder);
			if (verbosity >= 1) printStart();
		}
		for (int nbg = 0; nbg < nbGeneration; ++nbg) {
			newGenerationFunction();
			auto tg0 = high_resolution_clock::now();
#ifdef CLUSTER
			MPI_distributePopulation();
#endif
			evaluate();
#ifdef CLUSTER
			MPI_receivePopulation();
#endif
			if (procId == 0) {
				if (population.size() != popSize)
					throw std::invalid_argument("Population doesn't match the popSize param");
				if (novelty) updateNovelty();
				auto tg1 = high_resolution_clock::now();
				double totalTime = std::chrono::duration<double>(tg1 - tg0).count();
				auto tnp0 = high_resolution_clock::now();
				if (currentGeneration % savePopInterval == 0) {
					if (savePopEnabled) savePop();
					if (novelty && saveArchiveEnabled) saveArchive();
				}
				if (currentGeneration % saveGenInterval == 0) {
					if (doSaveParetoFront) {
						saveParetoFront();
					} else {
						saveBests(nbSavedElites);
					}
				}

				nextGeneration();

				updateStats(totalTime);
				if (verbosity >= 1) printGenStats(currentGeneration);
				if (doSaveGenStats) saveGenStats();
				if (doSaveIndStats) saveIndStats();

				auto tnp1 = high_resolution_clock::now();
				double tnp = std::chrono::duration<double>(tnp1 - tnp0).count();
				if (verbosity >= 2) {
					std::cout << "Time for save + next pop = " << tnp << " s." << std::endl;
				}
			}
			++currentGeneration;
		}
	}

	void finish() {
#ifdef CLUSTER
		MPI_Finalize();
#endif
	}

// MPI specifics
#ifdef CLUSTER
	void MPI_distributePopulation() {
		if (procId == 0) {
			// if we're in the master process, we send b(i)atches to the others.
			// master will have the remaining
			size_t batchSize = population.size() / nbProcs;
			for (size_t dest = 1; dest < (size_t)nbProcs; ++dest) {
				vector<Individual<DNA>> batch;
				for (size_t ind = 0; ind < batchSize; ++ind) {
					batch.push_back(population.back());
					population.pop_back();
				}
				string batchStr = Individual<DNA>::popToJSON(batch).dump();
				std::vector<char> tmp(batchStr.begin(), batchStr.end());
				tmp.push_back('\0');
				MPI_Send(tmp.data(), tmp.size(), MPI_BYTE, dest, 0, MPI_COMM_WORLD);
			}
		} else {
			// we're in a slave process, we welcome our local population !
			int strLength;
			MPI_Status status;
			MPI_Probe(0, 0, MPI_COMM_WORLD, &status);  // we want to know its size
			MPI_Get_count(&status, MPI_CHAR, &strLength);
			char *popChar = new char[strLength + 1];
			MPI_Recv(popChar, strLength, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			// and we dejsonize !
			auto o = json::parse(popChar);
			population = Individual<DNA>::loadPopFromJSON(o);  // welcome bros!
			if (verbosity >= 3) {
				std::ostringstream buf;
				buf << endl
				    << "Proc " << PURPLE << procId << NORMAL << " : reception of "
				    << population.size() << " new individuals !" << endl;
				cout << buf.str();
			}
		}
	}

	void MPI_receivePopulation() {
		if (procId != 0) {  // if slave process we send our population to our mighty leader
			string batchStr = Individual<DNA>::popToJSON(population).dump();
			std::vector<char> tmp(batchStr.begin(), batchStr.end());
			tmp.push_back('\0');
			MPI_Send(tmp.data(), tmp.size(), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
		} else {
			// master process receives all other batches
			for (size_t source = 1; source < (size_t)nbProcs; ++source) {
				int strLength;
				MPI_Status status;
				MPI_Probe(source, 0, MPI_COMM_WORLD, &status);  // determining batch size
				MPI_Get_count(&status, MPI_CHAR, &strLength);
				char *popChar = new char[strLength + 1];
				MPI_Recv(popChar, strLength + 1, MPI_BYTE, source, 0, MPI_COMM_WORLD,
				         MPI_STATUS_IGNORE);
				// and we dejsonize!
				auto o = json::parse(popChar);
				vector<Individual<DNA>> batch = Individual<DNA>::loadPopFromJSON(o);
				population.insert(population.end(), batch.begin(), batch.end());
				delete popChar;
				if (verbosity >= 3) {
					cout << endl
					     << "Proc " << procId << " : reception of " << batch.size()
					     << " treated individuals from proc " << source << endl;
				}
			}
		}
	}
#endif
	/*********************************************************************************
	 *                            NEXT POP GETTING READY
	 ********************************************************************************/
	void classicNextGen() {
		auto nextGen = produceNOffsprings(popSize, population, nbElites);
		lastGen = population;
		population = nextGen;
		if (verbosity >= 3) cerr << "Next generation ready" << endl;
	}

	// - evaluation de toute la pop, sans se soucier des espèces.
	// - choix des nouveaux représentants parmis les espèces précédentes (clonage)
	// - création d'une nouvelle population via selection/mutation/crossover intra-espece
	// - regroupement en nouvelles espèces en utilisant la distance aux représentants
	// (création d'une nouvelle espèce si distance < speciationThreshold)
	// - on supprime les espèces de taille < minSpecieSize
	// - on rajoute des individus en les faisant muter depuis une espèce aléatoire (nouvelle
	// espèce à chaque tirage) et en les rajoutants à cette espèce
	// - c'est reparti :D

	void speciationNextGen() {
		// TODO : for now it only works with mawimization
		// minimization would require a modification in the nOffsprings
		// and in the worstFitness computations

		if (verbosity >= 3) cerr << "Starting to prepare next speciated gen" << std::endl;
		assert(nbElites < minSpecieSize);

		if (species.size() == 0) {
			if (verbosity >= 3) cerr << "No specie available, creating one" << std::endl;
			// we put all the population in one specie
			species.resize(1);
			for (auto &i : population) species[0].push_back(&i);
			speciationThresholds.clear();
			speciationThresholds.resize(1);
			speciationThresholds[0] = speciationThreshold;
		}

		assert(species.size() == speciationThresholds.size());

		vector<Individual<DNA>> nextLeaders;
		// New species leaders
		for (auto &s : species) {
			std::uniform_int_distribution<size_t> d(0, s.size() - 1);
			nextLeaders.push_back(*s[d(globalRand)]);
		}
		if (verbosity >= 3) {
			cerr << "Found " << nextLeaders.size() << " leaders :" << std::endl;
			for (auto &l : nextLeaders) {
				//cerr << " -: " << l << std::endl;
			}
		}

		// list of objectives
		unordered_set<string> objectivesList;
		for (const auto &o : population[0].fitnesses) objectivesList.insert(o.first);
		assert(objectivesList.size() > 0);
		if (verbosity >= 3)
			cerr << "Found " << objectivesList.size() << " objectives" << std::endl;

		// computing afjustedFitnesses
		vector<unordered_map<string, double>> adjustedFitnessSum(species.size());
		unordered_map<string, double> worstFitness;
		for (const auto &o : objectivesList) {
			worstFitness[o] = std::numeric_limits<double>::max();
			for (const auto &i : population)
				if (i.fitnesses.at(o) < worstFitness.at(o)) worstFitness[o] = i.fitnesses.at(o);
		}
		// we want to offset all the adj fitnesses so they are in the positive range
		unordered_map<string, double> totalAdjustedFitness;
		for (const auto &o : objectivesList) {
			double total = 0;
			for (size_t i = 0; i < species.size(); ++i) {
				const auto &s = species[i];
				assert(s.size() > 0);
				double sum = 0;
				for (const auto &ind : s) sum += ind->fitnesses.at(o) - worstFitness.at(o) + 1;
				sum /= static_cast<double>(s.size());
				total += sum;
				adjustedFitnessSum[i][o] = sum;
			}
			totalAdjustedFitness[o] = total;
		}
		if (verbosity >= 3) {
			for (auto &af : totalAdjustedFitness) {
				cerr << " - total \"" << af.first << "\" = " << af.second << std::endl;
			}
		}

		// creating the new population
		lastGen = population;
		for (const auto &o : objectivesList) {
			assert(totalAdjustedFitness[o] != 0);
			for (size_t i = 0; i < species.size(); ++i) {
				auto &s = species[i];
				size_t nOffsprings =  // nb of offsprings the specie is authorized to produce
				    static_cast<size_t>((static_cast<double>(popSize) /
				                         static_cast<double>(objectivesList.size())) *
				                        adjustedFitnessSum[i][o] / totalAdjustedFitness[o]);

				population.reserve(population.size() + nOffsprings);
				auto specieOffsprings = produceNOffsprings(nOffsprings, s, nbElites);
				population.insert(population.end(),
				                  std::make_move_iterator(specieOffsprings.begin()),
				                  std::make_move_iterator(specieOffsprings.end()));
			}
		}
		population.clear();

		if (verbosity >= 3)
			cerr << "Created the new population. Population.size = " << population.size()
			     << std::endl;
		// correcting rounding errors by adding missing individuals
		while (population.size() < popSize) {
			std::uniform_int_distribution<size_t> d(0, nextLeaders.size() - 1);
			// we just add mutated leaders
			auto offspring = nextLeaders[d(globalRand)];
			offspring.dna.mutate();
			offspring.evaluated = false;
			population.push_back(offspring);
		}
		assert(population.size() == popSize);

		// creating new species
		species.clear();
		species.resize(nextLeaders.size());
		for (auto &i : population) {
			// finding the closest leader
			size_t closestLeader = 0;
			double closestDist = std::numeric_limits<double>::max();
			bool foundSpecie = false;
			for (size_t l = 0; l < nextLeaders.size(); ++l) {
				double dist = indDistanceFunction(nextLeaders[l], i);
				if (dist < closestDist && dist < speciationThresholds[l]) {
					closestDist = dist;
					closestLeader = l;
					foundSpecie = true;
				}
			}
			if (foundSpecie) {
				// we found your family
				species[closestLeader].push_back(&i);
			} else {
				// we found special snowflakes
				nextLeaders.push_back(i);
				species.push_back({{&i}});
				speciationThresholds.push_back(speciationThreshold);
			}
		}
		if (verbosity >= 3)
			cerr << "Created the new species. Species size = " << species.size() << std::endl;

		assert(species.size() == nextLeaders.size());
		assert(species.size() == speciationThresholds.size());

		// deleting small species
		vector<Iptr> toReplace;  // list of individuals without specie. We need to replace
		                         // them with new individuals. We use this because we cannot
		                         // directly delete individuals from the population without
		                         // invalidating all other pointers;
		for (auto it = species.begin(); it != species.end();) {
			if ((*it).size() < minSpecieSize) {
				for (auto &i : *it) toReplace.push_back(i);
				it = species.erase(it);
			} else
				++it;
		}
		if (verbosity >= 3)
			cerr << "Need to replace " << toReplace.size() << " individuals" << std::endl;

		// replacing all "deleted" individuals and putting them in existing species
		for (auto &i : toReplace) {
			// we choose one random specie and mutate individuals until the new ind can fit
			i->evaluated = false;
			auto selection = getSelectionMethod<vector<Iptr>>();
			std::uniform_int_distribution<size_t> d(0, nextLeaders.size() - 1);
			size_t leaderID = d(globalRand);
			unsigned int c = 0;
			do {
				if (c++ > MAX_SPECIATION_TRIES)
					throw std::runtime_error("Too many tries. Speciation thresholds too low.");
				i->dna = selection(species[leaderID])->dna;
				i->dna.mutate();
			} while (indDistanceFunction(*i, nextLeaders[leaderID]) >
			         speciationThresholds[leaderID]);
		}

		if (verbosity >= 3) cerr << "Done. " << std::endl;
		// adjusting speciation Thresholds
		size_t avgSpecieSize = 0;
		for (auto &s : species) avgSpecieSize += s.size();
		avgSpecieSize /= species.size();
		for (size_t i = 0; i < species.size(); ++i) {
			if (species[i].size() < avgSpecieSize) {
				speciationThresholds[i] =
				    std::min(speciationThresholds[i] + speciationThresholdIncrement,
				             maxSpeciationThreshold);
			} else {
				speciationThresholds[i] =
				    std::max(speciationThresholds[i] - speciationThresholdIncrement,
				             minSpeciationThreshold);
			}
		}
		if (verbosity >= 3) {
			cerr << "Speciation thresholds adjusted: " << std::endl;
			for (auto &s : speciationThresholds) {
				cerr << " " << s;
			}
			cerr << std::endl;
		}
	}

	template <typename I>  // I is ither Individual<DNA> or Individual<DNA>*
	vector<Individual<DNA>> produceNOffsprings(size_t n, vector<I> &popu,
	                                           size_t nElites = 0) {
		if (verbosity >= 3)
			cerr << "Going to produce " << n << " offsprings out of " << popu.size()
			     << " individuals" << endl;
		std::uniform_real_distribution<double> d(0.0, 1.0);
		vector<Individual<DNA>> nextGen;
		nextGen.reserve(n);
		// Elites are placed at the begining
		if (nElites > 0) {
			auto elites = getElites(nElites, popu);
			if (verbosity >= 3) cerr << "elites.size = " << elites.size() << endl;
			for (auto &e : elites)
				for (auto &i : e.second) nextGen.push_back(i);
		}

		auto selection = getSelectionMethod<vector<I>>();

		while (nextGen.size() < n) {
			if (verbosity >= 3) cerr << "nextGen.size = " << nextGen.size() << endl;
			// selection + crossover
			Individual<DNA> *p0 = selection(popu);
			Individual<DNA> offspring;
			if (d(globalRand) < crossoverProba) {
				if (verbosity >= 3) cerr << "crossover" << endl;
				Individual<DNA> *p1 = selection(popu);
				offspring = Individual<DNA>(p0->dna.crossover(p1->dna));
				offspring.evaluated = false;
				if (verbosity >= 3) cerr << "crossover ok" << endl;
			} else {
				if (verbosity >= 3) cerr << "no crossover" << endl;
				offspring = *p0;
			}
			// mutation
			if (d(globalRand) < mutationProba) {
				if (verbosity >= 3) cerr << "mutation" << endl;
				offspring.dna.mutate();
				offspring.evaluated = false;
			}
			if (verbosity >= 3) cerr << "pushing" << endl;
			nextGen.push_back(offspring);
		}
		if (verbosity >= 3) cerr << "done" << endl;
		assert(nextGen.size() == n);
		return nextGen;
	}

	bool paretoDominates(const Individual<DNA> &a, const Individual<DNA> &b) const {
		for (auto &o : a.fitnesses)
			if (!isBetter(o.second, b.fitnesses.at(o.first))) return false;
		return true;
	}

	vector<Individual<DNA> *> getParetoFront(
	    const std::vector<Individual<DNA> *> &ind) const {
		// naive algorithm. Should be ok for small ind.size()
		vector<Individual<DNA> *> pareto;
		for (size_t i = 0; i < ind.size(); ++i) {
			bool dominated = false;
			for (auto &j : pareto) {
				if (paretoDominates(*j, *ind[i])) {
					dominated = true;
					break;
				}
			}
			if (!dominated) {
				for (size_t j = i + 1; j < ind.size(); ++j) {
					if (paretoDominates(*ind[j], *ind[i])) {
						dominated = true;
						break;
					}
				}
				if (!dominated) {
					pareto.push_back(ind[i]);
				}
			}
		}
		return pareto;
	}

	template <typename I> Individual<DNA> *paretoTournament(vector<I> &subPop) {
		std::uniform_int_distribution<size_t> dint(0, subPop.size() - 1);
		std::vector<Individual<DNA> *> participants;
		for (size_t i = 0; i < tournamentSize; ++i)
			participants.push_back(&ref(subPop[dint(globalRand)]));
		auto pf = getParetoFront(participants);
		assert(pf.size() > 0);
		std::uniform_int_distribution<size_t> dpf(0, pf.size() - 1);
		return pf[dpf(globalRand)];
	}

	template <typename I> Individual<DNA> *randomObjTournament(vector<I> &subPop) {
		if (verbosity >= 3) cerr << "random obj tournament called" << endl;
		std::uniform_int_distribution<size_t> dint(0, subPop.size() - 1);
		std::vector<Individual<DNA> *> participants;
		for (size_t i = 0; i < tournamentSize; ++i)
			participants.push_back(&ref(subPop[dint(globalRand)]));
		auto champion = participants[0];
		// we pick the objective randomly
		std::string obj;
		if (champion->fitnesses.size() == 1) {
			obj = champion->fitnesses.begin()->first;
		} else {
			std::uniform_int_distribution<int> dObj(
			    0, static_cast<int>(champion->fitnesses.size()) - 1);
			auto it = champion->fitnesses.begin();
			std::advance(it, dObj(globalRand));
			obj = it->first;
		}
		for (size_t i = 1; i < tournamentSize; ++i) {
			if (isBetter(participants[i]->fitnesses.at(obj), champion->fitnesses.at(obj)))
				champion = participants[i];
		}
		if (verbosity >= 3) cerr << "champion found" << endl;
		return champion;
	}

	// getELites methods : returns a vector of N best individuals in the specified
	// subPopulations, for the specified fitnesses.
	// elites indivuduals are not ordered.
	unordered_map<string, vector<Individual<DNA>>> getElites(size_t n) {
		vector<string> obj;
		for (auto &o : population[0].fitnesses) obj.push_back(o.first);
		return getElites(obj, n, population);
	}

	template <typename I>
	unordered_map<string, vector<Individual<DNA>>> getElites(size_t n,
	                                                         const vector<I> &popVec) {
		vector<string> obj;
		for (auto &o : ref(popVec[0]).fitnesses) obj.push_back(o.first);
		return getElites(obj, n, popVec);
	}
	unordered_map<string, vector<Individual<DNA>>> getLastGenElites(size_t n) {
		vector<string> obj;
		for (auto &o : population[0].fitnesses) obj.push_back(o.first);
		return getElites(obj, n, lastGen);
	}
	template <typename I>
	unordered_map<string, vector<Individual<DNA>>> getElites(const vector<string> &obj,
	                                                         size_t n,
	                                                         const vector<I> &popVec) {
		if (verbosity >= 3) {
			cerr << "getElites : nbObj = " << obj.size() << " n = " << n << endl;
		}
		unordered_map<string, vector<Individual<DNA>>> elites;
		std::cerr << "yo 0" << std::endl;
		for (auto &o : obj) {
			std::cerr << "yo 1" << std::endl;
			elites[o] = vector<Individual<DNA>>();
			elites[o].push_back(ref(popVec[0]));
			size_t worst = 0;
			std::cerr << "yo 2" << std::endl;
			for (size_t i = 1; i < n && i < popVec.size(); ++i) {
				elites[o].push_back(ref(popVec[i]));
				if (isBetter(elites[o][worst].fitnesses.at(o), ref(popVec[i]).fitnesses.at(o)))
					worst = i;
			}
			std::cerr << "yo 3" << std::endl;
			for (size_t i = n; i < popVec.size(); ++i) {
				std::cerr << "yo 0 3" << std::endl;
				if (isBetter(ref(popVec[i]).fitnesses.at(o), elites[o][worst].fitnesses.at(o))) {
					std::cerr << "yo 1 3" << std::endl;
					elites[o][worst] = ref(popVec[i]);
					for (size_t j = 0; j < n; ++j) {
						if (isBetter(elites[o][worst].fitnesses.at(o), elites[o][j].fitnesses.at(o)))
							worst = j;
					}
					std::cerr << "yo 2 3" << std::endl;
				}
			}
		}
		std::cerr << "yo 4" << std::endl;
		return elites;
	}

 protected:
	/*********************************************************************************
	 *                          NOVELTY RELATED METHODS
	 ********************************************************************************/
	// Novelty works with footprints. A footprint is just a vector of vector of doubles.
	// It is recommended that those doubles are within a same order of magnitude.
	// Each vector<double> is a "snapshot": it represents the state of the evaluation of
	// one individual at a certain time. Thus, a complete footprint is a combination
	// of one or more snapshot taken at different points in the
	// simulation (a vector<vector<double>>).
	// Snapshot must be of same size accross individuals.
	// Footprint must be set in the evaluator (see examples)

	static double getFootprintDistance(const fpType &f0, const fpType &f1) {
		assert(f0.size() == f1.size());
		double d = 0;
		for (size_t i = 0; i < f0.size(); ++i) {
			for (size_t j = 0; j < f0[i].size(); ++j) {
				d += std::pow(f0[i][j] - f1[i][j], 2);
			}
		}
		return sqrt(d);
	}

	// computeAvgDist (novelty related)
	// returns the average distance of a footprint fp to its k nearest neighbours
	// in an archive of footprints
	static double computeAvgDist(size_t K, const vector<Individual<DNA>> &arch,
	                             const fpType &fp) {
		double avgDist = 0;
		if (arch.size() > 1) {
			size_t k = arch.size() < K ? static_cast<size_t>(arch.size()) : K;
			vector<Individual<DNA>> knn;
			knn.reserve(k);
			vector<double> knnDist;
			knnDist.reserve(k);
			std::pair<double, size_t> worstKnn = {getFootprintDistance(fp, arch[0].footprint),
			                                      0};  // maxKnn is the worst among the knn
			for (size_t i = 0; i < k; ++i) {
				knn.push_back(arch[i]);
				double d = getFootprintDistance(fp, arch[i].footprint);
				knnDist.push_back(d);
				if (d > worstKnn.first) {
					worstKnn = {d, i};
				}
			}
			for (size_t i = k; i < arch.size(); ++i) {
				double d = getFootprintDistance(fp, arch[i].footprint);
				if (d < worstKnn.first) {  // this one is closer than our worst knn
					knn[worstKnn.second] = arch[i];
					knnDist[worstKnn.second] = d;
					worstKnn.first = d;
					// we update maxKnn
					for (size_t j = 0; j < knn.size(); ++j) {
						if (knnDist[j] > worstKnn.first) {
							worstKnn = {knnDist[j], j};
						}
					}
				}
			}
			assert(knn.size() == k);
			for (size_t i = 0; i < knn.size(); ++i) {
				assert(getFootprintDistance(fp, knn[i].footprint) == knnDist[i]);
				avgDist += knnDist[i];
			}
			avgDist /= static_cast<double>(knn.size());
		}
		return avgDist;
	}
	void updateNovelty() {
		if (verbosity >= 2) {
			cout << endl << endl;
			std::stringstream output;
			cout << GREY << " ❯❯  " << YELLOW << "COMPUTING NOVELTY " << NORMAL << " ⤵  "
			     << endl
			     << endl;
		}
		auto savedArchiveSize = archive.size();
		for (auto &ind : population) {
			archive.push_back(ind);
		}
		std::pair<Individual<DNA> *, double> best = {&population[0], 0};
		vector<Individual<DNA>> toBeAdded;
		for (auto &ind : population) {
			double avgD = computeAvgDist(KNN, archive, ind.footprint);
			bool added = false;
			if (avgD > minNoveltyForArchive) {
				toBeAdded.push_back(ind);
				added = true;
			}
			if (avgD > best.second) best = {&ind, avgD};
			if (verbosity >= 2) {
				std::stringstream output;
				output << GREY << " ❯ " << endl
				       << NORMAL << ind.infos << endl
				       << " -> Novelty = " << CYAN << avgD << GREY
				       << (added ? " (added to archive)" : " (too low for archive)") << NORMAL;
				if (verbosity >= 3)
					output << "Footprint was : " << footprintToString(ind.footprint);
				output << endl;
				std::cout << output.str();
			}
			ind.fitnesses["novelty"] = avgD;
		}
		archive.resize(savedArchiveSize);
		archive.insert(std::end(archive), std::begin(toBeAdded), std::end(toBeAdded));
		if (verbosity >= 2) {
			std::stringstream output;
			output << " Added " << toBeAdded.size() << " new footprints to the archive."
			       << std::endl
			       << "New archive size = " << archive.size() << " (was " << savedArchiveSize
			       << ")." << std::endl;
			std::cout << output.str() << std::endl;
		}
		if (verbosity >= 2) {
			std::stringstream output;
			output << "Most novel individual (novelty = " << best.second
			       << "): " << best.first->infos << endl;
			cout << output.str();
		}
	}

	// panpan cucul
	static inline string footprintToString(const vector<vector<double>> &f) {
		std::ostringstream res;
		res << "👣  " << json(f).dump();
		return res.str();
	}

	string selectMethodToString(const SelectionMethod &sm) {
		switch (sm) {
			case SelectionMethod::paretoTournament:
				return "pareto tournament";
			case SelectionMethod::randomObjTournament:
				return "random objective tournament";
		}
		return "???";
	}

	/*********************************************************************************
	 *                           STATS, LOGS & PRINTING
	 ********************************************************************************/
	void printStart() {
		int nbCol = 55;
		std::cout << std::endl << GREY;
		for (int i = 0; i < nbCol - 1; ++i) std::cout << "━";
		std::cout << std::endl;
		std::cout << YELLOW << "              ☀     " << NORMAL << " Starting GAGA " << YELLOW
		          << "    ☀ " << NORMAL;
		std::cout << std::endl;
		std::cout << BLUE << "                      ¯\\_ಠ ᴥ ಠ_/¯" << std::endl << GREY;
		for (int i = 0; i < nbCol - 1; ++i) std::cout << "┄";
		std::cout << std::endl << NORMAL;
		std::cout << "  ▹ population size = " << BLUE << popSize << NORMAL << std::endl;
		std::cout << "  ▹ nb of elites = " << BLUE << nbElites << NORMAL << std::endl;
		std::cout << "  ▹ nb of tournament competitors = " << BLUE << tournamentSize << NORMAL
		          << std::endl;
		std::cout << "  ▹ selection = " << BLUE << selectMethodToString(selecMethod) << NORMAL
		          << std::endl;
		std::cout << "  ▹ mutation rate = " << BLUE << mutationProba << NORMAL << std::endl;
		std::cout << "  ▹ crossover rate = " << BLUE << crossoverProba << NORMAL << std::endl;
		std::cout << "  ▹ writing results in " << BLUE << folder << NORMAL << std::endl;
		if (novelty) {
			std::cout << "  ▹ novelty is " << GREEN << "enabled" << NORMAL << std::endl;
			std::cout << "    - KNN size = " << BLUE << KNN << NORMAL << std::endl;
		} else {
			std::cout << "  ▹ novelty is " << RED << "disabled" << NORMAL << std::endl;
		}
		if (speciation) {
			std::cout << "  ▹ speciation is " << GREEN << "enabled" << NORMAL << std::endl;
			std::cout << "    - minSpecieSize size = " << BLUE << minSpecieSize << NORMAL
			          << std::endl;
			std::cout << "    - speciationThreshold = " << BLUE << speciationThreshold << NORMAL
			          << std::endl;
			std::cout << "    - speciationThresholdIncrement = " << BLUE
			          << speciationThresholdIncrement << NORMAL << std::endl;
			std::cout << "    - minSpeciationThreshold = " << BLUE << minSpeciationThreshold
			          << NORMAL << std::endl;
			std::cout << "    - maxSpeciationThreshold = " << BLUE << maxSpeciationThreshold
			          << NORMAL << std::endl;
		} else {
			std::cout << "  ▹ speciation is " << RED << "disabled" << NORMAL << std::endl;
		}
#ifdef CLUSTER
		std::cout << "  ▹ MPI parallelisation is " << GREEN << "enabled" << NORMAL
		          << std::endl;
#else
		std::cout << "  ▹ MPI parallelisation is " << RED << "disabled" << NORMAL
		          << std::endl;
#endif
#ifdef OMP
		std::cout << "  ▹ OpenMP parallelisation is " << GREEN << "enabled" << NORMAL
		          << std::endl;
#else
		std::cout << "  ▹ OpenMP parallelisation is " << RED << "disabled" << NORMAL
		          << std::endl;
#endif
		std::cout << GREY;
		for (int i = 0; i < nbCol - 1; ++i) std::cout << "━";
		std::cout << NORMAL << std::endl;
	}
	void updateStats(double totalTime) {
		// stats organisations :
		// "global" -> {"genTotalTime", "indTotalTime", "maxTime", "nEvals", "nObjs"}
		// "obj_i" -> {"avg", "worst", "best"}
		assert(lastGen.size());
		std::map<std::string, std::map<std::string, double>> currentGenStats;
		currentGenStats["global"]["genTotalTime"] = totalTime;
		double indTotalTime = 0.0, maxTime = 0.0;
		int nEvals = 0;
		int nObjs = static_cast<int>(lastGen[0].fitnesses.size());
		for (const auto &o : lastGen[0].fitnesses) {
			currentGenStats[o.first] = {
			    {{"avg", 0.0}, {"worst", o.second}, {"best", o.second}}};
		}
		for (const auto &ind : lastGen) {
			indTotalTime += ind.evalTime;
			for (const auto &o : ind.fitnesses) {
				currentGenStats[o.first].at("avg") +=
				    (o.second / static_cast<double>(lastGen.size()));
				if (isBetter(o.second, currentGenStats[o.first].at("best")))
					currentGenStats[o.first].at("best") = o.second;
				if (!isBetter(o.second, currentGenStats[o.first].at("worst")))
					currentGenStats[o.first].at("worst") = o.second;
			}
			if (ind.evalTime > maxTime) maxTime = ind.evalTime;
			if (!ind.wasAlreadyEvaluated) ++nEvals;
		}
		currentGenStats["global"]["indTotalTime"] = indTotalTime;
		currentGenStats["global"]["maxTime"] = maxTime;
		currentGenStats["global"]["nEvals"] = nEvals;
		currentGenStats["global"]["nObjs"] = nObjs;
		if (speciation) {
			currentGenStats["global"]["nSpecies"] = species.size();
		}
		genStats.push_back(currentGenStats);
	}

	void printGenStats(size_t n) {
		const size_t l = 80;
		std::cout << tableHeader(l);
		std::ostringstream output;
		const auto &globalStats = genStats[n].at("global");
		output << "Generation " << CYANBOLD << n << NORMAL << " ended in " << BLUE
		       << globalStats.at("genTotalTime") << NORMAL << "s";
		std::cout << tableCenteredText(l, output.str(), BLUEBOLD NORMAL BLUE NORMAL);
		output = std::ostringstream();
		output << GREYBOLD << "(" << globalStats.at("nEvals") << " evaluations, "
		       << globalStats.at("nObjs") << " objs)" << NORMAL;
		std::cout << tableCenteredText(l, output.str(), GREYBOLD NORMAL);
		std::cout << tableSeparation(l);
		double timeRatio = 0;
		if (globalStats.at("genTotalTime") > 0)
			timeRatio = globalStats.at("indTotalTime") / globalStats.at("genTotalTime");
		output = std::ostringstream();
		output << "🕝  max: " << BLUE << globalStats.at("maxTime") << NORMAL << "s";
		output << ", 🕝  sum: " << BLUEBOLD << globalStats.at("indTotalTime") << NORMAL
		       << "s (x" << timeRatio << " ratio)";
		std::cout << tableCenteredText(l, output.str(), CYANBOLD NORMAL BLUE NORMAL "      ");
		std::cout << tableSeparation(l);
		for (const auto &o : genStats[n]) {
			if (o.first != "global") {
				output = std::ostringstream();
				output << GREYBOLD << "--◇" << GREENBOLD << std::setw(10) << o.first << GREYBOLD
				       << " ❯ " << NORMAL << " worst: " << YELLOW << std::setw(12)
				       << o.second.at("worst") << NORMAL << ", avg: " << YELLOWBOLD
				       << std::setw(12) << o.second.at("avg") << NORMAL << ", best: " << REDBOLD
				       << std::setw(12) << o.second.at("best") << NORMAL;
				std::cout << tableText(l, output.str(),
				                       "    " GREYBOLD GREENBOLD GREYBOLD NORMAL YELLOWBOLD NORMAL
				                           YELLOW NORMAL GREENBOLD NORMAL);
			}
		}
		std::cout << tableFooter(l);
	}

	void printIndividualStats(const Individual<DNA> &ind) {
		std::ostringstream output;
		output << GREYBOLD << "[" << YELLOW << procId << GREYBOLD << "]-▶ " << NORMAL;
		for (const auto &o : ind.fitnesses)
			output << " " << o.first << ": " << BLUEBOLD << std::setw(12) << o.second << NORMAL
			       << GREYBOLD << " |" << NORMAL;
		output << " 🕝 : " << BLUE << ind.evalTime << "s" << NORMAL;
		if (ind.wasAlreadyEvaluated)
			output << GREYBOLD << " (already evaluated)\n" << NORMAL;
		else
			output << "\n";
		if ((!novelty && verbosity >= 2) || verbosity >= 3) output << ind.infos << std::endl;
		std::cout << output.str();
	}

	std::string tableHeader(unsigned int l) {
		std::ostringstream output;
		output << "┌";
		for (auto i = 0u; i < l; ++i) output << "─";
		output << "┓\n";
		return output.str();
	}

	std::string tableFooter(unsigned int l) {
		std::ostringstream output;
		output << "┗";
		for (auto i = 0u; i < l; ++i) output << "─";
		output << "┛\n";
		return output.str();
	}

	std::string tableSeparation(unsigned int l) {
		std::ostringstream output;
		output << "|" << GREYBOLD;
		for (auto i = 0u; i < l; ++i) output << "-";
		output << NORMAL << "|\n";
		return output.str();
	}

	std::string tableText(int l, std::string txt, std::string unprinted = "") {
		std::ostringstream output;
		int txtsize = static_cast<int>(txt.size() - unprinted.size());
		output << "|" << txt;
		for (auto i = txtsize; i < l; ++i) output << " ";
		output << "|\n";
		return output.str();
	}

	std::string tableCenteredText(int l, std::string txt, std::string unprinted = "") {
		std::ostringstream output;
		int txtsize = static_cast<int>(txt.size() - unprinted.size());
		int space = l - txtsize;
		output << "|";
		for (int i = 0; i < space / 2; ++i) output << " ";
		output << txt;
		for (int i = (space / 2) + txtsize; i < l; ++i) output << " ";
		output << "|\n";
		return output.str();
	}

	/*********************************************************************************
	 *                         SAVING STUFF
	 ********************************************************************************/
	void saveBests(size_t n) {
		if (n > 0) {
			// save n bests dnas for all objectives
			vector<string> objectives;
			for (auto &o : population[0].fitnesses) {
				objectives.push_back(o.first);  // we need to know objective functions
			}
			auto elites = getElites(objectives, n, population);
			std::stringstream baseName;
			baseName << folder << "/gen" << currentGeneration;
			mkdir(baseName.str().c_str(), 0777);
			if (verbosity >= 3) {
				cerr << "created directory " << baseName.str() << endl;
			}
			for (auto &e : elites) {
				int id = 0;
				for (auto &i : e.second) {
					std::stringstream fileName;
					fileName << baseName.str() << "/" << e.first << "_" << i.fitnesses.at(e.first)
					         << "_" << id++ << ".dna";
					std::ofstream fs(fileName.str());
					if (!fs) {
						cerr << "Cannot open the output file." << endl;
					}
					fs << i.dna.serialize();
					fs.close();
				}
			}
		}
	}

	void saveParetoFront() {
		std::vector<Individual<DNA> *> pop;
		for (size_t i = 0; i < population.size(); ++i) {
			pop.push_back(&population[i]);
		}

		auto pfront = getParetoFront(pop);
		std::stringstream baseName;
		baseName << folder << "/gen" << currentGeneration;
		mkdir(baseName.str().c_str(), 0777);
		if (verbosity >= 3) {
			std::cout << "created directory " << baseName.str() << std::endl;
		}

		int id = 0;
		for (const auto &ind : pfront) {
			std::stringstream filename;
			filename << baseName.str() << "/";
			for (const auto &f : ind->fitnesses) {
				filename << f.first << f.second << "_";
			}
			filename << id++ << ".dna";

			std::ofstream fs(filename.str());
			if (!fs) {
				std::cerr << "Cannot open the output file.\n";
			}
			fs << ind->dna.serialize();
			fs.close();
		}
	}

	void saveGenStats() {
		std::stringstream csv;
		std::stringstream fileName;
		fileName << folder << "/gen_stats.csv";
		csv << "generation";
		if (genStats.size() > 0) {
			for (const auto &cat : genStats[0]) {
				std::stringstream column;
				column << cat.first << "_";
				for (const auto &s : cat.second) {
					csv << "," << column.str() << s.first;
				}
			}
			csv << endl;
			for (size_t i = 0; i < genStats.size(); ++i) {
				csv << i;
				for (const auto &cat : genStats[i]) {
					for (const auto &s : cat.second) {
						csv << "," << s.second;
					}
				}
				csv << endl;
			}
		}
		std::ofstream fs(fileName.str());
		if (!fs) cerr << "Cannot open the output file." << endl;
		fs << csv.str();
		fs.close();
	}

	// gen, idInd, fit0, fit1, time
	void saveIndStats() {
		std::stringstream csv;
		std::stringstream fileName;
		fileName << folder << "/ind_stats.csv";
		static bool indStatsWritten = false;
		if (!indStatsWritten) {
			csv << "generation,idInd,";
			for (auto &o : lastGen[0].fitnesses) csv << o.first << ",";
			csv << "time" << std::endl;
			indStatsWritten = true;
		}
		for (size_t i = 0; i < lastGen.size(); ++i) {
			const auto &p = lastGen[i];
			csv << currentGeneration << "," << i << ",";
			for (auto &o : p.fitnesses) csv << o.second << ",";
			csv << p.evalTime << std::endl;
		}
		std::ofstream fs;
		fs.open(fileName.str(), std::fstream::out | std::fstream::app);
		if (!fs) cerr << "Cannot open the output file." << endl;
		fs << csv.str();
		fs.close();
	}

	void saveIndStats_OneLinePerGen() {
		std::stringstream csv;
		std::stringstream fileName;
		fileName << folder << "/ind_stats.csv";

		static bool has_been_written = false;

		if (!has_been_written) {
			csv << "generation";
			size_t i = 0;
			for (const auto &ind : lastGen) {
				csv << ",ind" << i++;
				for (const auto &f : ind.fitnesses) {
					csv << "," << f.first;
				}
				csv << ",is_on_pareto_front,eval_time";
			}
			csv << endl;

			has_been_written = true;
		}

		std::vector<int> is_on_front(lastGen.size(), false);

		if (selecMethod == SelectionMethod::paretoTournament) {
			std::vector<Individual<DNA> *> pop;

			for (auto &p : lastGen) {
				pop.push_back(&p);
			}

			auto front = getParetoFront(pop);

			for (size_t i = 0; i < pop.size(); ++i) {
				Individual<DNA> *ind0 = pop[i];
				int found = 0;

				for (size_t j = 0; !found && (j < front.size()); ++j) {
					Individual<DNA> *ind1 = front[j];

					if (ind1 == ind0) {
						found = 1;
					}
				}

				is_on_front[i] = found;
			}
		}

		{
			csv << currentGeneration;
			size_t ind_id = 0;
			for (const auto &ind : lastGen) {
				csv << "," << ind_id;
				for (const auto &f : ind.fitnesses) {
					csv << "," << f.second;
				}

				csv << "," << is_on_front[ind_id];
				csv << "," << ind.evalTime;
				++ind_id;
			}
			csv << endl;
		}

		std::ofstream fs;
		fs.open(fileName.str(), std::fstream::out | std::fstream::app);
		if (!fs) {
			cerr << "Cannot open the output file." << endl;
		}
		fs << csv.str();
		fs.close();
	}

	int mkpath(char *file_path, mode_t mode) {
		assert(file_path && *file_path);
		char *p;
		for (p = strchr(file_path + 1, '/'); p; p = strchr(p + 1, '/')) {
			*p = '\0';
			if (mkdir(file_path, mode) == -1) {
				if (errno != EEXIST) {
					*p = '/';
					return -1;
				}
			}
			*p = '/';
		}
		return 0;
	}
	void createFolder(string baseFolder) {
		if (baseFolder.back() != '/') baseFolder += "/";
		struct stat sb;
		char bFChar[500];
		strncpy(bFChar, baseFolder.c_str(), 500);
		mkpath(bFChar, 0777);
		auto now = system_clock::now();
		time_t now_c = system_clock::to_time_t(now);
		struct tm *parts = localtime(&now_c);

		std::stringstream fname;
		fname << evaluatorName << "_" << parts->tm_mday << "_" << parts->tm_mon + 1 << "_";
		int cpt = 0;
		std::stringstream ftot;
		do {
			ftot.clear();
			ftot.str("");
			ftot << baseFolder << fname.str() << cpt;
			cpt++;
		} while (stat(ftot.str().c_str(), &sb) == 0 && S_ISDIR(sb.st_mode));
		folder = ftot.str();
		mkdir(folder.c_str(), 0777);
	}

 public:
	void loadPop(string file) {
		std::ifstream t(file);
		std::stringstream buffer;
		buffer << t.rdbuf();
		auto o = json::parse(buffer.str());
		assert(o.count("population"));
		if (o.count("generation")) {
			currentGeneration = o.at("generation");
		} else {
			currentGeneration = 0;
		}
		population.clear();
		for (auto ind : o.at("population")) {
			population.push_back(Individual<DNA>(DNA(ind.at("dna"))));
			population[population.size() - 1].evaluated = false;
		}
	}

	void savePop() {
		json o = Individual<DNA>::popToJSON(population);
		o["evaluator"] = evaluatorName;
		o["generation"] = currentGeneration;
		std::stringstream baseName;
		baseName << folder << "/gen" << currentGeneration;
		mkdir(baseName.str().c_str(), 0777);
		std::stringstream fileName;
		fileName << baseName.str() << "/pop" << currentGeneration << ".pop";
		std::ofstream file;
		file.open(fileName.str());
		file << o.dump();
		file.close();
	}
	void saveArchive() {
		json o = Individual<DNA>::popToJSON(archive);
		o["evaluator"] = evaluatorName;
		std::stringstream baseName;
		baseName << folder << "/gen" << currentGeneration;
		mkdir(baseName.str().c_str(), 0777);
		std::stringstream fileName;
		fileName << baseName.str() << "/archive" << currentGeneration << ".pop";
		std::ofstream file;
		file.open(fileName.str());
		file << o.dump();
		file.close();
	}
};
}  // namespace GAGA
#endif
