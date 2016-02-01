// Gaga: lightweight simple genetic algorithm library
// Copyright (c) Jean Disset 2015, All rights reserved.

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

#include <sys/stat.h>
#include <sys/types.h>
#include <assert.h>
#include <vector>
#include <chrono>
#include <fstream>
#include <unordered_set>
#include <unordered_map>
#include <deque>
#include <random>
#include <utility>
#include <map>
#include <string>
#include "json/json.hpp"

/****************************************
 *       TO ENABLE PARALLELISATION
 * *************************************/
// before including this file,
// #define OMP if you want OpenMP parallelisation
// #define CLUSTER if you want MPI parralelisation
#ifdef OMP
#include <omp.h>
#endif
#ifdef CLUSTER
#include <mpi.h>
#endif

#define PURPLE "\033[1;35m"
#define BLUE "\033[34m"
#define GREY "\033[1;30m"
#define YELLOW "\033[1;33m"
#define RED "\033[1;31m"
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
using fpType = std::vector<std::vector<double>>;          // footprints for novelty
using archType = std::vector<std::pair<fpType, double>>;  // collection of footprints
using json = nlohmann::json;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using std::chrono::system_clock;

/******************************************************************************************
 *                                 GAGA LIBRARY
 *****************************************************************************************/
// This file contains :
// 1 - the Individual class template : an individual's generic representation, with its
// dna, fitnesses and
// behavior footprints (for novelty)
// 2 - the main GA class template
//
// About parallelisation :
// before including this file,
// #define OMP if you want OpenMP parallelisation
// #define CLUSTER if you want MPI parralelisation

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
	bool evaluated = false;

	explicit Individual(const DNA &d) : dna(d) {}

	explicit Individual(const json &o) {
		assert(o.count("dna"));
		dna = DNA(o.at("dna"));
		if (o.count("footprint")) {
			json fp(o.at("footprint"));
			for (auto &fCol : fp) {
				vector<double> fc;
				for (string f : fCol) {
					double val;
					sscanf(f.c_str(), "%lf", &val);
					fc.push_back(val);
				}
				footprint.push_back(fp);
			}
		}
		if (o.count("fitnesses")) {
			json fitObj = o.at("fitnesses");
			for (json::iterator it = fitObj.begin(); it != fitObj.end(); ++it) {
				double val;
				sscanf(it.value().get<string>().c_str(), "%lf", &val);
				fitnesses[it.key()] = val;
			}
		}
	}

	// Exports individual to json
	json toJSON() const {
		json fitObject;
		for (auto &f : fitnesses) {
			char buf[50];
			snprintf(buf, sizeof(buf), "%a", f.second);
			fitObject[f.first] = buf;
		}
		vector<vector<string>> fpstr;
		for (auto &f0 : footprint) {
			vector<string> f;
			for (auto &f1 : f0) {
				char buf[50];
				snprintf(buf, sizeof(buf), "%a", f1);
				f.push_back(buf);
			}
			fpstr.push_back(f);
		}
		json o;
		o["dna"] = json::parse(dna.toJSON());
		o["fitnesses"] = fitObject;
		o["footprint"] = fpstr;
		return o;
	}

	// Exports a vector of individual to json
	static json popToJSON(const vector<Individual<DNA>> &p) {
		json o;
		json popArray;
		for (auto &i : p) {
			popArray.push_back(i.toJSON());
		}
		o["population"] = popArray;
		return o;
	}

	// Loads a vector of individual from json
	static vector<Individual<DNA>> loadPopFromJSON(const json &o) {
		assert(o.count("population"));
		vector<Individual<DNA>> res;
		json popArray = o.at("population");
		for (auto &ind : popArray) {
			res.push_back(Individual<DNA>(ind));
		}
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

template <typename DNA, typename Evaluator> class GA {
 protected:
	/*********************************************************************************
	 *                            MAIN GA SETTINGS
	 ********************************************************************************/
	bool novelty = false;  // is novelty enabled ?
	unsigned int verbosity =
	    2;  // 0 = silent; 1 = generations stats; 2 = individuals stats; 3 = everything
	unsigned int popSize = 500;       // nb of individuals in the population
	unsigned int nbElites = 1;        // nb of elites to keep accross generations
	unsigned int nbSavedElites = 1;   // nb of elites to save
	unsigned int tournamentSize = 3;  // nb of competitors in tournament
	unsigned int nbGen = 500;         // nb of generations
	double minNoveltyForArchive =
	    0.0001;                      // min novelty for being added to the general archive
	unsigned int KNN = 5;           // size of the neighbourhood for novelty
	unsigned int saveInterval = 1;  // interval between 2 whole population saves
	string folder = "../evos/";     // where to save the results
	double crossoverProba = 0.2;    // crossover probability
	double mutationProba = 0.5;     // mutation probablility
	unordered_map<string, double>
	    proportions;  // {{"baseObj", 0.25}, {"novelty", 0.75}};  // fitness weight
	                  // proportions contains the relative weights of the objectives
	                  // if an objective is not present here but still used at evaluation
	                  // a default non weighted average will be used
	/********************************************************************************
	 *                                 SETTERS
	 ********************************************************************************/
 public:
	void enableNovelty() { novelty = true; }
	void disableNovelty() { novelty = false; }
	void setVerbosity(unsigned int lvl) { verbosity = lvl <= 3 ? (lvl >= 0 ? lvl : 0) : 3; }
	void setPopSize(unsigned int s) { popSize = s; }
	void setNbElites(unsigned int n) { nbElites = n; }
	void setNbSavedElites(unsigned int n) { nbSavedElites = n; }
	void setTournamentSize(unsigned int n) { tournamentSize = n; }
	void setNbGenerations(unsigned int n) { nbGen = n; }
	void setKNN(unsigned int n) { KNN = n; }
	void setPopSaveInterval(unsigned int n) { saveInterval = n; }
	void setSaveFolder(string s) { folder = s; }
	void setCrossoverProba(double p) {
		crossoverProba = p <= 1.0 ? (p >= 0.0 ? p : 0.0) : 1.0;
	}
	void setMutationProba(double p) {
		mutationProba = p <= 1.0 ? (p >= 0.0 ? p : 0.0) : 1.0;
	}
	void setObjectivesDistribution(map<string, double> d) { proportions = d; }
	void setObjectivesDistribution(string o, double d) { proportions[o] = d; }

	////////////////////////////////////////////////////////////////////////////////////

 protected:
	Evaluator evaluate;
	archType archive;  // where we store behaviors footprints for novelty
	vector<Individual<DNA>> population;
	unsigned int currentGeneration = 0;
	// openmp/mpi stuff
	unsigned int procId = 0;
	unsigned int nbProcs = 1;
	int argc = 1;
	char **argv = nullptr;

	vector<unordered_map<string, unordered_map<string, double>>> stats;  // fitnesses stats
	std::default_random_engine globalRand = std::default_random_engine(
	    std::chrono::system_clock::now().time_since_epoch().count());

 public:
	/*********************************************************************************
	 *                              CONSTRUCTOR
	 ********************************************************************************/
	GA(int ac, char **av) : evaluate(ac, av), argc(ac), argv(av) {
#ifdef CLUSTER
		MPI_Init(&argc, &argv);
		MPI_Comm_size(MPI_COMM_WORLD, &nbProcs);
		MPI_Comm_rank(MPI_COMM_WORLD, &procId);
		if (procId == 0) {
			if (verbosity >= = 3) {
				cerr << "   -------------------" << endl;
				cerr << CYAN << " MPI STARTED WITH " << NORMAL << nbProcs << CYAN << " PROCS "
				     << NORMAL << endl;
				cerr << "   -------------------" << endl;
				cerr << "Initialising population in master process" << endl;
			}
#endif
#ifdef CLUSTER
		}
#endif
	}

	/*********************************************************************************
	 *                          START THE BOUZIN
	 ********************************************************************************/
	// "Vroum vroum"
	int start() {
		for (unsigned int i = 0; i < popSize; ++i) {
			population.push_back(Individual<DNA>(DNA::random(argc, argv)));
			population[population.size() - 1].evaluated = false;
		}
		createFolder(folder);
		bool finished = false;
		if (verbosity >= 1) {
			int nbCol = 55;
			std::cout << std::endl << GREY;
			for (int i = 0; i < nbCol - 1; ++i) std::cout << "━";
			std::cout << std::endl;
			std::cout << YELLOW << "              ☀     " << NORMAL << " Starting GAGA "
			          << YELLOW << "    ☀ " << NORMAL;
			std::cout << std::endl << GREY;
			for (int i = 0; i < nbCol - 1; ++i) std::cout << "┄";
			std::cout << std::endl << NORMAL;
			std::cout << "  ▹ population size = " << BLUE << popSize << NORMAL << std::endl;
			std::cout << "  ▹ nb of elites = " << BLUE << nbElites << NORMAL << std::endl;
			std::cout << "  ▹ nb of tournament competitors = " << BLUE << tournamentSize
			          << NORMAL << std::endl;
			std::cout << "  ▹ mutation rate = " << BLUE << mutationProba << NORMAL << std::endl;
			std::cout << "  ▹ crossover rate = " << BLUE << crossoverProba << NORMAL
			          << std::endl;
			std::cout << "  ▹ writing results in " << BLUE << folder << NORMAL << std::endl;
			if (novelty) {
				std::cout << "  ▹ novelty is " << GREEN << "enabled" << NORMAL << std::endl;
				std::cout << "    - KNN size = " << BLUE << KNN << NORMAL << std::endl;
			} else {
				std::cout << "  ▹ novelty is " << RED << "disabled" << NORMAL << std::endl;
			}
#ifdef CLUSTER
			std::cout << "  ▹ MPI parralelisation is " << GREEN << "enabled" << NORMAL
			          << std::endl;
#else
			std::cout << "  ▹ MPI parralelisation is " << RED << "disabled" << NORMAL
			          << std::endl;
#endif
#ifdef OMP
			std::cout << "  ▹ OpenMP parralelisation is " << GREEN << "enabled" << NORMAL
			          << std::endl;
#else
			std::cout << "  ▹ OpenMP parralelisation is " << RED << "disabled" << NORMAL
			          << std::endl;
#endif
			std::cout << GREY;
			for (int i = 0; i < nbCol - 1; ++i) std::cout << "━";
			std::cout << std::endl << NORMAL;
		}
		while (!finished) {
			auto tg0 = high_resolution_clock::now();
#ifdef CLUSTER
			if (procId == 0) {
				// if we're in the master process, we send b(i)atches to the others.
				// master will have the remaining
				unsigned int batchSize = population.size() / nbProcs;
				for (unsigned int dest = 1; dest < (unsigned int)nbProcs; ++dest) {
					vector<Individual<DNA>> batch;
					for (size_t ind = 0; ind < batchSize; ++ind) {
						batch.push_back(population.back());
						population.pop_back();
					}
					ostringstream batchOSS;
					batchOSS << Individual<DNA>::popToJSON(batch);
					string batchStr = batchOSS.str();
					MPI_Send(batchStr.c_str(), batchStr.length() + 1, MPI_BYTE, dest, 0,
					         MPI_COMM_WORLD);
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
				auto o = json::Parse(popChar);
				population = Individual<DNA>::loadPopFromJSON(o);  // welcome bros!
				if (verbosity >= 3) {
					cerr << endl
					     << "Proc " << procId << " : reception of " << population.size()
					     << " new individuals !" << endl;
				}
			}
#endif
			if (verbosity >= 3) {
				cerr << "population.size() = " << population.size() << endl;
			}

#ifdef OMP
			omp_lock_t statsLock;
			omp_init_lock(&statsLock);
			int nbAlreadyEvaluated = 0;
			for (const auto &p : population) {
				if (p.evaluated) ++nbAlreadyEvaluated;
			}
#pragma omp parallel for schedule(dynamic, 1)
#endif
			for (size_t i = 0; i < population.size(); ++i) {
				population[i].dna.reset();
				double totalTime = 0.0;
				std::ostringstream indStatus;
				if (population[i].evaluated) {
					if (verbosity >= 3) {
						std::stringstream msg;
						msg << "-◇- Ind " << i << " already evaluated" << std::endl;
						std::cout << msg.str();
					}
				} else {
					if (verbosity >= 3) {
						std::stringstream msg;
						msg << "-◇- Evaluation starting for ind " << i << std::endl;
						std::cout << msg.str();
					}

					auto t0 = high_resolution_clock::now();
					evaluate(population[i]);
					auto t1 = high_resolution_clock::now();
					population[i].evaluated = true;
					if (verbosity >= 3) {
						indStatus << "-◇- Evaluation ended for ind " << i << std::endl;
						if (novelty) {
							indStatus << " 👣  Footprint : "
							          << footprintToString(population[i].footprint) << std::endl;
						}
					}
					totalTime = std::chrono::duration<double>(t1 - t0).count();
				}
				// STATS:
				unordered_set<string> best;
#ifdef OMP
				omp_set_lock(&statsLock);
#endif
				while (stats.size() <= currentGeneration) {
					stats.push_back(unordered_map<string, unordered_map<string, double>>());
				}
				for (auto &o : population[i].fitnesses) {
					if (!stats[currentGeneration].count(o.first)) {
						stats[currentGeneration][o.first]["max"] = -1e30;
						stats[currentGeneration][o.first]["min"] = 1e30;
						stats[currentGeneration][o.first]["avg"] = 0;
					}
					stats[currentGeneration][o.first]["avg"] += o.second;
					if (o.second > stats[currentGeneration][o.first]["max"]) {  // new best
						best.insert(o.first);
						stats[currentGeneration][o.first]["max"] = o.second;
					}
					if (o.second < stats[currentGeneration][o.first]["min"]) {
						stats[currentGeneration][o.first]["min"] = o.second;
					}
				}
				if (verbosity >= 1) {
					indStatus << CYANBOLD << "✷ " << NORMAL << "(" << PURPLE << procId << NORMAL
					          << ") ➳    Ind " << YELLOW << std::setw(3) << i << NORMAL
					          << " evaluated in " GREEN << std::setw(3) << std::setprecision(1)
					          << std::fixed << totalTime << NORMAL << "s" << std::endl;
					if (verbosity >= 4) {
						indStatus << " Ind " << i << "'s dna = " << population[i].dna.toJSON()
						          << std::endl;
					}
					for (auto &o : population[i].fitnesses) {
						indStatus << " " << o.first << " : ";
						if (best.count(o.first)) {
							indStatus << CYANBOLD;
						} else {
							indStatus << GREEN;
						}
						double val = o.second;
						indStatus << val << NORMAL << ";";
					}
					if (best.size() > 0) {
						indStatus << endl;
						indStatus << GA<DNA, Evaluator>::footprintToString(population[i].footprint);
						indStatus << endl;
					}
					if (verbosity >= 2) {
						cout << indStatus.str();
					}
				}
#ifdef OMP
				omp_unset_lock(&statsLock);
#endif
			}
#ifdef CLUSTER
			if (procId != 0) {  // if slave process we send our population to our mighty leader
				ostringstream batchOSS;
				batchOSS << Individual<DNA>::popToJSON(population);
				string batchStr = batchOSS.str();
				MPI_Send(batchStr.c_str(), batchStr.length() + 1, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
			} else {
				// master process receives all other batches
				for (unsigned int source = 1; source < (unsigned int)nbProcs; ++source) {
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
					if (verbosity >= 3) {
						cout << endl
						     << "Proc " << procId << " : reception of " << batch.size()
						     << " treated individuals from proc " << source << endl;
					}
				}
#endif
				// the end of a generation
				// now we update novelty
				if (novelty) {
					auto savedArchiveSize = archive.size();
					for (auto &ind : population) {
						archive.push_back({ind.footprint, 1});
					}
					archType toBeAdded;
					for (auto &ind : population) {
						double avgD = computeAvgDist(KNN, archive, ind.footprint);
						bool added = false;
						if (avgD > minNoveltyForArchive) {
							toBeAdded.push_back({ind.footprint, 1});
							added = true;
						}
						if (verbosity >= 2) {
							std::stringstream output;
							output << "[ " << footprintToString(ind.footprint) << "] novelty = " << CYAN
							       << avgD << NORMAL
							       << (added ? "(added to archive)" : "(too low for archive)") << endl;
							std::cout << output.str() << std::endl;
						}
						ind.fitnesses["novelty"] = avgD;
						if (stats[currentGeneration].count("novelty") == 0) {
							stats[currentGeneration]["novelty"]["max"] = -1e30;
							stats[currentGeneration]["novelty"]["min"] = 1e30;
							stats[currentGeneration]["novelty"]["avg"] = 0;
						}
						stats[currentGeneration]["novelty"]["avg"] += ind.fitnesses.at("novelty");
						if (avgD > stats[currentGeneration]["novelty"]["max"]) {  // new best
							stats[currentGeneration]["novelty"]["max"] = avgD;      // new best
							if (verbosity >= 2) {
								cout << " New best novelty: " << CYAN << avgD << NORMAL << endl;
								cout << footprintToString(ind.footprint);
							}
						}
					}
					if (currentGeneration > 0) {
						archive.resize(savedArchiveSize);
						archive.insert(std::end(archive), std::begin(toBeAdded), std::end(toBeAdded));
						if (verbosity >= 2) {
							std::stringstream output;
							output << " Added " << toBeAdded.size() << " new footprints to the archive."
							       << std::endl
							       << "New archive size = " << archive.size() << " (was "
							       << savedArchiveSize << ")." << std::endl;
							std::cout << output.str() << std::endl;
						}
					}
				}
				auto tg1 = high_resolution_clock::now();
				double totalTime = std::chrono::duration<double>(tg1 - tg0).count();
				for (auto &o : stats[currentGeneration]) {
					o.second["avg"] /= static_cast<double>(population.size());
				}
				if (verbosity >= 1) {
					int totalCol = 77;
					std::stringstream buf;
					cout << endl;
					buf << "GENERATION " << currentGeneration;
					cout << endl
					     << GREY << "+" << std::setfill('-') << std::setw(totalCol) << "-"
					     << "+" << NORMAL << endl;
					cout << GREY << "|" << std::setfill(' ') << std::setw(totalCol) << " "
					     << "|" << NORMAL << endl;
					cout << GREY << "|" << GREENBOLD;
					printCentered(totalCol, buf.str());
					cout << GREY << "|" << NORMAL << endl;
					cout << GREY << "|" << std::setfill(' ') << std::setw(totalCol) << " "
					     << "|" << NORMAL << endl;
					cout << GREY << "|" << std::setfill(' ') << std::setw(totalCol) << " "
					     << "|" << NORMAL << endl;
					buf = std::stringstream();
					buf << PURPLE << population.size() - nbAlreadyEvaluated << NORMAL
					    << " evaluations in " << BLUE << totalTime << NORMAL << "s";
					cout << GREY << "|";
					printCentered(totalCol, buf.str());
					cout << GREY << "|" << NORMAL << endl;
					cout << "+" << std::setfill('-') << std::setw(totalCol) << "-"
					     << "+" << endl;

					int nCol = totalCol - 2;
					cout << GREY << "|" << CYAN;
					printCentered(nCol / 3, "Obj name");
					cout << GREY << "|" << CYAN;
					printCentered(nCol / 3, "Best");
					cout << GREY << "|" << CYAN;
					printCentered(nCol / 3, "Avg");
					cout << GREY << "|" << NORMAL << endl;
					cout << "+" << GREY << std::setfill('-') << std::setw(totalCol) << "-" << NORMAL
					     << "+" << endl;
					for (auto &o : stats[currentGeneration]) {
						cout << GREY << "|" << GREEN;
						printCentered(nCol / 3, o.first);
						cout << GREY << "|" << NORMAL;
						buf = std::stringstream();
						buf << o.second["max"];
						printCentered(nCol / 3, buf.str());
						cout << GREY << "|" << NORMAL;
						buf = std::stringstream();
						buf << o.second["avg"];
						printCentered(nCol / 3, buf.str());
						cout << GREY << "|" << NORMAL << endl;
						cout << "+" << GREY << std::setfill('-') << std::setw(totalCol) << "-"
						     << NORMAL << "+" << endl;
					}
					cout << GREY << "+" << std::setfill('-') << std::setw(totalCol) << "-"
					     << "+" << NORMAL << endl;
				}
				stats[currentGeneration]["global"]["time"] = totalTime;
				// we save everybody
				if (currentGeneration % saveInterval == 0) savePop();
				saveBests(nbSavedElites);
				saveStats();
				// and prepare the next gen
				prepareNextPop();
				finished = (currentGeneration++ >= nbGen);
#ifdef CLUSTER
			}
#endif
		}
#ifdef CLUSTER
		MPI_Finalize();
#endif
		return 0;
	}

	// print helper
	void printCentered(unsigned int totalCol, const string &s) {
		int c = 2 * ((totalCol - (s.size())) / 2) + s.size() == totalCol ? 0 : 1;
		cout << std::setfill(' ') << std::setw((totalCol - s.size()) / 2) << " " << s
		     << std::setw(c + (totalCol - s.size()) / 2) << " ";
	}

	/*********************************************************************************
	 *                            NEXT POP GETTING READY
	 ********************************************************************************/
	// Là où qu'on fait les bébés.
	void prepareNextPop() {
		std::uniform_real_distribution<double> d(0.0, 1.0);
		if (verbosity >= 3) {
			cerr << "-◇- copying elites" << endl;
		}
		vector<Individual<DNA>> nextGen;
		vector<string> objectives;
		for (auto &o : population[0].fitnesses) {
			objectives.push_back(o.first);  // we need to know what the objectives are
		}
		if (verbosity >= 3) {
			std::cout << BLUE << "The objectives are" << NORMAL;
			for (auto &s : objectives) {
				std::cout << std::endl << " * " << s;
			}
			std::cout << std::endl;
		}
		map<string, vector<Individual<DNA>>> elites = getElites(objectives, nbElites);
		// we put the elites in the nextGen
		for (auto &e : elites) {
			for (auto &i : e.second) {
				nextGen.push_back(i);
			}
		}
		unsigned int firstNonEliteIndex = nextGen.size();
		if (verbosity >= 3) {
			cerr << "firstNonEliteIndex = " << firstNonEliteIndex << endl;
		}
		// now we can start the tournament
		std::uniform_int_distribution<int> dint(0, population.size() - 1);
		if (verbosity >= 3) {
			cerr << "-◇- starting tournaments" << endl;
		}
		// we need to know how much of each objective's representatives we will have
		// if the proportion map has been set, we use it. Else we split equally
		unordered_map<string, double> objPop;  // nb of ind per objective
		double sum = 0;  // we need to be sure these proportions are normalized
		for (auto &o : objectives) {
			if (proportions.count(o) > 0) {
				objPop[o] = proportions.at(o);
			} else {
				objPop[o] = 1.0 / static_cast<double>(objectives.size());
			}
			sum += objPop.at(o);
		}
		int availableSpots = population.size() - nextGen.size();
		int cpt = 0;
		int id = 0;
		for (auto &o : objPop) {
			o.second /= sum;  // normalization
			if (id++ < static_cast<int>(objPop.size() - 1)) {
				o.second = static_cast<int>(o.second * static_cast<double>(availableSpots));
				cpt += o.second;
			} else {  // if it's the last objective, we fill up the rest
				o.second = availableSpots - cpt;
			}
		}
		if (verbosity >= 3) {
			std::cout << "Population target per objectives after normalization: " << std::endl;
			for (auto &op : objPop) {
				std::cout << " * " << op.first << " = " << op.second << std::endl;
			}
		}

		for (auto &o : objPop) {
			unsigned int popGoal = nextGen.size() + static_cast<unsigned int>(o.second);

			if (verbosity >= 3) {
				std::cout << "Starting tournaments (" << tournamentSize
				          << " competitors) for obj " << o.first << std::endl;
			}
			while (nextGen.size() < popGoal) {
				std::vector<Individual<DNA> *> tournament;
				tournament.resize(tournamentSize);
				for (unsigned int i = 0; i < tournamentSize; ++i) {
					int selected = dint(globalRand);
					if (verbosity >= 3) {
						std::cout << GREY << "Selected ind " << selected << " with fitness " << NORMAL
						          << population[selected].fitnesses.at(o.first) << std::endl;
					}
					tournament[i] = &population[selected];
				}
				Individual<DNA> *winner = tournament[0];
				for (unsigned int i = 1; i < tournamentSize; ++i) {
					assert(tournament[i]->fitnesses.count(o.first));
					if (tournament[i]->fitnesses.at(o.first) > winner->fitnesses.at(o.first)) {
						winner = tournament[i];
					}
				}
				auto winnerclone = *winner;
				if (verbosity >= 3) {
					std::cout << "Winner's fitness = " << winnerclone.fitnesses.at(o.first)
					          << std::endl;
				}
				if (d(globalRand) <= mutationProba) {  // mutation
					winnerclone.dna.mutate();
					winnerclone.evaluated = false;
					if (verbosity >= 3) {
						std::cout << GREEN << "Winner has mutated" << NORMAL << std::endl;
					}
				}
				nextGen.push_back(winnerclone);
			}
		}
		if (verbosity >= 3) {
			cerr << "-◇- crossovers" << endl;
		}
		population.clear();
		for (unsigned int pi = 0; pi < nextGen.size(); ++pi) {
			auto &p1 = nextGen[pi];
			if (pi >= firstNonEliteIndex && d(globalRand) < crossoverProba) {
				unsigned int r = dint(globalRand);
				const Individual<DNA> &p2 = nextGen[r];
				if (verbosity >= 3) {
					std::cerr << "crossing ind " << BLUE << pi << NORMAL << " with ind " << YELLOW
					          << r << NORMAL << std::endl;
				}
				population.push_back(Individual<DNA>(p1.dna.crossover(p2.dna)));
				population[population.size() - 1].evaluated = false;
			} else {
				population.push_back(p1);
			}
		}
	}

	map<string, vector<Individual<DNA>>> getElites(const vector<string> &obj, int n) {
		if (verbosity >= 3) {
			cerr << "getElites : nbObj = " << obj.size() << " n = " << n << endl;
		}
		map<string, vector<Individual<DNA>>> elites;
		for (auto &o : obj) {
			elites[o] = vector<Individual<DNA>>();
			for (auto &i : population) {
				if (static_cast<int>(elites.at(o).size()) < n) {
					// we push the first inds without looking at their fitness
					elites.at(o).push_back(i);
				} else {
					// but if we already have enough elites we need to check if the one we have
					// really are the best (and replace them if they're not)
					auto worst = elites.at(o).begin();
					auto worstScore = worst->fitnesses.at(o);
					// first we find the worst individual we have saved until now
					for (auto j = elites.at(o).begin(); j != elites.at(o).end(); ++j) {
						if (worstScore > j->fitnesses.at(o)) {
							worstScore = j->fitnesses.at(o);
							worst = j;
						}
					}
					// then we replace it if our candidate is better
					if (worstScore < i.fitnesses.at(o)) {
						*worst = i;
					}
				}
			}
		}
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
	static double computeAvgDist(unsigned int k, const archType &arch, const fpType &fp) {
		double avgDist = 0;
		if (arch.size() > k) {
			archType knn;
			knn.reserve(k);
			vector<double> knnDist;
			knnDist.reserve(k);
			std::pair<double, size_t> maxKnn = {getFootprintDistance(fp, arch[0].first),
			                                    0};  // maxKnn is the worst among the knn
			for (unsigned int i = 0; i < k; ++i) {
				knn.push_back(arch[i]);
				double d = getFootprintDistance(fp, arch[i].first);
				knnDist.push_back(d);
				if (d > maxKnn.first) {
					maxKnn = {d, i};
				}
			}
			for (size_t i = k; i < arch.size(); ++i) {
				double d = getFootprintDistance(fp, arch[i].first);
				if (d < maxKnn.first) {
					knn[maxKnn.second] = arch[i];
					knnDist[maxKnn.second] = d;
					maxKnn.first = d;
					// then we update maxKnn
					for (size_t j = 0; j < knn.size(); ++j) {
						if (knnDist[j] > maxKnn.first) {
							maxKnn = {knnDist[j], j};
						}
					}
				}
			}
			assert(knn.size() == k);
			double divisor = 0;
			for (size_t i = 0; i < knn.size(); ++i) {
				assert(getFootprintDistance(fp, knn[i].first) == knnDist[i]);
				avgDist += knnDist[i] * static_cast<double>(knn[i].second);
				divisor += knn[i].second;
			}
			assert(divisor > 0);
			avgDist /= divisor;
		}
		return avgDist;
	}

	// panpan cucul
	static string footprintToString(const vector<vector<double>> &f) {
		std::ostringstream res;
		for (auto &p : f) {
			res << NORMAL << "     [" << GREY;
			for (const double &v : p) {
				res << " " << std::setw(4) << std::setprecision(3) << std::fixed << v;
			}
			res << NORMAL << " ]" << endl;
		}
		return res.str();
	}
	/*********************************************************************************
	 *                         SAVING STUFF
	 ********************************************************************************/
	void saveBests(int n) {
		// save n bests dnas for all objectives
		vector<string> objectives;
		for (auto &o : population[0].fitnesses) {
			objectives.push_back(o.first);  // we need to know what are the objective functions
		}
		map<string, vector<Individual<DNA>>> elites = getElites(objectives, n);
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
				fs << i.dna.toJSON();
				fs.close();
			}
		}
	}

	void saveStats() {
		std::stringstream csv;
		std::stringstream fileName;
		fileName << folder << "/stats.csv";
		csv << "generation";
		if (stats.size() > 0) {
			for (auto &cat : stats[0]) {
				std::stringstream column;
				column << cat.first << "_";
				for (auto &s : cat.second) {
					csv << "," << column.str() << s.first;
				}
			}
			csv << endl;
			for (size_t i = 0; i < stats.size(); ++i) {
				csv << i;
				for (auto &cat : stats[i]) {
					for (auto &s : cat.second) {
						csv << "," << s.second;
					}
				}
				csv << endl;
			}
		}
		std::ofstream fs(fileName.str());
		if (!fs) {
			cerr << "Cannot open the output file." << endl;
		}
		fs << csv.str();
		fs.close();
	}

	void createFolder(string baseFolder) {
		struct stat sb;
		mkdir(baseFolder.c_str(), 0777);
		auto now = system_clock::now();
		time_t now_c = system_clock::to_time_t(now);
		struct tm *parts = localtime(&now_c);

		std::stringstream fname;
		fname << evaluate.name << parts->tm_mday << "_" << parts->tm_mon + 1 << "_";
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
		o["evaluator"] = evaluate.name;
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
};
}  // namespace GAGA
#endif
