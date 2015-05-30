// Gaga: "grosse ambiance" genetic algorithm library
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

#include "individual.hpp"
#include "tools.h"
#include "jsonxx/jsonxx.h"
#include <vector>
#include <chrono>
#include <fstream>
#include <ctime>
#include <unordered_set>
#include <unordered_map>
#include <deque>
#include <sys/stat.h>
#include <sys/types.h>

//#define OMP
#ifdef OMP
#include <omp.h>
#endif
//#define CLUSTER
#ifdef CLUSTER
#include <mpi.h>
#endif

using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using std::chrono::system_clock;
using namespace std;

namespace GAGA {

/*********************************************************************************
 *                                 GA CLASS
 ********************************************************************************/
// DNA class must have :
// mutate()
// crossover(DNA& other)
// random ()
// reset()
// jsonxx:Object& constructor
// toJson()
//
// Evaluaor class must have
// operator()(const Individual<DNA>& ind)
// const name

template <typename DNA, typename Evaluator> class GA {
	/*********************************************************************************
	 *                            MAIN GA SETTINGS
	 ********************************************************************************/
	bool novelty = false;            // is novelty enabled ?
	unsigned int verbosity = 1;      // 0 = silent; 1 = generations stats; 2 = individuals stats; 3 = everything
	unsigned int popSize = 500;      // nb of individuals in the population
	unsigned int nbElites = 1;       // nb of elites to keep accross generations
	unsigned int nbSavedElites = 1;  // nb of elites to save
	unsigned int tournamentSize = 3; // nb of competitors in tournament
	unsigned int nbGen = 500;        // nb of generations
	unsigned int maxArchiveSize = 2000; // nb of footprints to keep for novelty computations
	unsigned int KNN = 15;              // size of the neighbourhood for novelty
	string folder = "../evos/";         // where to save the results
	double crossoverProba = 0.3;        // crossover probability
	double mutationProba = 0.5;         // mutation probablility
	map<string, double> proportions;    // {{"baseObj", 0.25}, {"novelty", 0.75}};  // fitness weight
	                                    // proportions contains the relative weights of the objectives
	                                    // if an objective is not present here but still used at evaluation
	                                    // a default non weighted average will be used
	/********************************************************************************
	 *                                 SETTERS
	 ********************************************************************************/
	void enableNovelty() { novelty = true; }
	void disableNovelty() { novelty = false; }
	void setVerbosity(unsigned int lvl) { verbosity = lvl <= 3 ? (lvl >= 0 ? lvl : 0) : 3; }
	void setPopSize(unsigned int s) { popSize = s; }
	void setNbElites(unsigned int n) { nbElites = n; }
	void setNbSavedElites(unsigned int n) { nbSavedElites = n; }
	void setTournamentSize(unsigned int n) { tournamentSize = n; }
	void setNbGenerations(unsigned int n) { nbGen = n; }
	void setMaxArchiveSize(unsigned int n) { maxArchiveSize = n; }
	void setKNN(unsigned int n) { KNN = n; }
	void setSaveFolder(string s) { folder = s; }
	void setCrossoverProba(double p) { crossoverProba = p <= 1.0 ? (p >= 0.0 ? p : 0.0) : 1.0; }
	void setMutationProba(double p) { mutationProba = p <= 1.0 ? (p >= 0.0 ? p : 0.0) : 1.0; }
	void setObjectivesDistribution(map<string, double> d) { proportions = d; }
	void setObjectivesDistribution(string o, double d) { proportions[o] = d; }

	////////////////////////////////////////////////////////////////////////////////////

	Evaluator ev;
	archType archive; // where for novelty
	vector<Individual<DNA>> population;
	unsigned int currentGeneration = 0;
	// openmp/mpi stuff
	unsigned int procId = 0;
	unsigned int nbProcs = 1;
	int argc = 1;
	char **argv = nullptr;

	map<string, pair<double, double>> stats;     // fitnesses stats. First = best, second = avg
	map<string, pair<double, double>> prevStats; // fitnesses stats. First = best, second = avg

	/*********************************************************************************
	 *                          NOVELTY RELATED METHODS
	 ********************************************************************************/
	// Notes about novelty
	// novelty works with footprints. A footprint is just a vector of vector of doubles
	// it is recommended that those doubles are within a same order of magnitude
	// each vector of double is a "snapshot" : it represents the state of the evaluation of one individual
	// at a certain time. Thus, a complete footprint is a combination (a vector) of one or more snapshot
	// taken at different points in the simulation. Snapshot must be of same size accross individuals
	// footprint must be set in the evaluator (see examples)

	static double getFootprintDistance(const vector<vector<double>> &f0, const vector<vector<double>> &f1) {
		assert(f0.size() == f1.size());
		double d = 0;
		for (unsigned int i = 0; i < f0.size(); ++i) {
			assert(f0[i].size() == f1[i].size());
			for (unsigned int j = 0; j < f0[i].size(); ++j) {
				d += pow(f0[i][j] - f1[i][j], 2);
			}
		}
		return d;
	}

	// computeAvgDist (novelty related)
	// returns the average distance of a foot print fp to its k nearest neighbours
	// in a collection of footprints arch
	static double computeAvgDist(unsigned int k, const archType &arch, const fpType &fp) {
		archType knn;
		double avgDist = 0;
		if (arch.size() > k) {
			for (unsigned int i = 0; i < k; ++i) {
				knn.push_back(arch[i]);
			}
			for (unsigned int i = k; i < arch.size(); ++i) {
				double d0 = GA<DNA, Evaluator>::getFootprintDistance(fp, arch[i].first);
				for (auto j = knn.begin(); j != knn.end(); ++j) {
					if (d0 < GA<DNA, Evaluator>::getFootprintDistance(fp, (*j).first)) {
						knn.erase(j);
						knn.push_back(arch[i]);
						break;
					}
				}
			}
			double divisor = 0;
			for (auto &i : knn) {
				double tmpD = GA<DNA, Evaluator>::getFootprintDistance(fp, i.first);
				double divCoef = exp(-2.0 * tmpD) * i.second;
				avgDist += tmpD * divCoef;
				divisor += divCoef;
				// avoid stagnation
				if (tmpD == 0.0 && i.second >= (double)k) return 0.0;
			}
			if (divisor == 0.0) { // if this happens there's probably a bug somewhere
				avgDist = 0;
			} else {
				avgDist /= divisor;
			}
		}
		return avgDist;
	}
	void computeIndNovelty(Individual<DNA> &i) {
		i.fitnesses["novelty"] = GA<DNA, Evaluator>::computeAvgDist(KNN, archive, i.footprint);
	}

	// panpan cucul
	static string footprintToString(const vector<vector<double>> &f) {
		ostringstream res;
		for (auto &p : f) {
			res << NORMAL << "     [" << GREY;
			for (const double &v : p) {
				res << " " << setw(4) << setprecision(3) << fixed << v;
			}
			res << NORMAL << " ]" << endl;
		}
		return res.str();
	}

public:
	/*********************************************************************************
	 *                              CONSTRUCTOR
	 ********************************************************************************/
	GA(int ac, char **av) : argc(ac), argv(av) {
#ifdef CLUSTER
		MPI_Init(&argc, &argv);
		MPI_Comm_size(MPI_COMM_WORLD, &nbProcs);
		MPI_Comm_rank(MPI_COMM_WORLD, &procId);
		if (procId == 0) {
			cerr << "   -------------------" << endl;
			cerr << CYAN << " MPI STARTED WITH " << NORMAL << nbProcs << CYAN << " PROCS " << NORMAL << endl;
			cerr << "   -------------------" << endl;
			cerr << "Initialising population in master process" << endl;
#endif
			for (unsigned int i = 0; i < popSize; ++i) {
				population.push_back(Individual<DNA>(DNA::random()));
			}
			createFolder(folder);
#ifdef CLUSTER
		}
#endif
	}

	/*********************************************************************************
	 *                          START THE BOUZIN
	 ********************************************************************************/
	// "Vroum vroum"
	int start() {
		bool finished = false;
		if (verbosity >= 1) {
			cout << GREEN << " Starting GA " << NORMAL << endl;
		}
		while (!finished) {
			int nbEval = 0;
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
					MPI_Send(batchStr.c_str(), batchStr.length() + 1, MPI_BYTE, dest, 0, MPI_COMM_WORLD);
				}
			} else {
				// we're in a slave process, we welcome our local population !
				int strLength;
				MPI_Status status;
				MPI_Probe(0, 0, MPI_COMM_WORLD, &status); // we want to know its size
				MPI_Get_count(&status, MPI_CHAR, &strLength);
				char *popChar = new char[strLength + 1];
				MPI_Recv(popChar, strLength, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				// and we dejsonize !
				jsonxx::Object o;
				bool success = o.parse(popChar);
				if (!success) {
					cerr << "parse failed. Str = " << popChar << endl;
				}
				population = Individual<DNA>::loadPopFromJSON(o); // welcome bros!
				if (verbosity >= 3) {
					cerr << endl
					     << "Proc " << procId << " : reception of " << population.size() << " new individuals !"
					     << endl;
				}
			}
#endif
			if (verbosity >= 3) {
				cerr << "population.size() = " << population.size() << endl;
			}

#ifdef OMP
			omp_lock_t statsLock;
			omp_init_lock(&statsLock);
#pragma omp parallel for schedule(dynamic, 1)
#endif
			for (size_t i = 0; i < population.size(); ++i) {
				population[i].dna.reset();
				if (!population[i].evaluated) {
#ifndef CLUSTER
					nbEval++;
#endif
					auto t0 = high_resolution_clock::now();
					ev(population[i]);
					auto t1 = high_resolution_clock::now();
					// STATS:
					if (verbosity >= 1) {
						milliseconds totalTime = std::chrono::duration_cast<milliseconds>(t1 - t0);
						ostringstream indStatus;
						unordered_set<string> best;
#ifdef OMP
						omp_set_lock(&statsLock);
#endif
						for (auto &o : population[i].fitnesses) {
							if (stats.count(o.first) == 0) {
								stats[o.first] = make_pair(0.0, 0.0);
							}
							try {
								stats.at(o.first).second += o.second;
							} catch (const out_of_range &e) {
								cerr << "Error : " << e.what() << " ; " << o.first << " is out of stats range" << endl;
							}
							if (o.second > stats.at(o.first).first) { // new best
								best.insert(o.first);
								stats.at(o.first).first = o.second;
							}
						}
						indStatus << endl
						          << " Proc " << PURPLE << procId << NORMAL << " : Ind " << YELLOW << setw(3) << i
						          << NORMAL << " evaluated in " GREEN << setw(3) << setprecision(1) << fixed
						          << totalTime.count() / 1000.0f << NORMAL << "s :";
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
#ifdef OMP
						omp_unset_lock(&statsLock);
#endif
					}
				}
			}
#ifdef CLUSTER
			if (procId != 0) { // if slave process we send our population to our mighty leader
				ostringstream batchOSS;
				batchOSS << Individual<DNA>::popToJSON(population);
				string batchStr = batchOSS.str();
				MPI_Send(batchStr.c_str(), batchStr.length() + 1, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
			} else {
				// master process receives all other batches
				for (unsigned int source = 1; source < (unsigned int)nbProcs; ++source) {
					int strLength;
					MPI_Status status;
					MPI_Probe(source, 0, MPI_COMM_WORLD, &status); // determining batch size
					MPI_Get_count(&status, MPI_CHAR, &strLength);
					char *popChar = new char[strLength + 1];
					MPI_Recv(popChar, strLength + 1, MPI_BYTE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					// and we dejsonize!
					jsonxx::Object o;
					bool success = o.parse(popChar);
					if (!success) {
						cerr << endl << "parse failed. Str = " << popChar << endl;
					}
					vector<Individual<DNA>> batch = Individual<DNA>::loadPopFromJSON(o);
					population.insert(population.end(), batch.begin(), batch.end());
					if (verbosity >= 3) {
						cerr << endl
						     << "Proc " << procId << " : reception of " << batch.size()
						     << " treated individuals from proc " << source << endl;
					}
				}
				// now we update novelty
				if (novelty) {
					for (auto &ind : population) {
						nbEval++;
						double avgD = GA<DNA, Evaluator>::computeAvgDist(KNN, archive, ind.footprint);
						ind.fitnesses["novelty"] = avgD;
						if (stats.count("novelty") == 0) {
							stats["novelty"] = make_pair(0.0, 0.0);
						}
						stats.at("novelty").second += ind.fitnesses.at("novelty");
						if (avgD > stats.at("novelty").first) { // new best
							stats.at("novelty").first = avgD;
							if (verbosity >= 2) {
								cerr << " New best novelty: " << CYAN << avgD << NORMAL << endl;
								cerr << footprintToString(ind.footprint);
							}
						}
					}
				}
#endif
				// the end of a generation
				auto tg1 = high_resolution_clock::now();
				milliseconds totalTime = std::chrono::duration_cast<milliseconds>(tg1 - tg0);
				if (verbosity >= 1) {
					cout << endl;
					cout << YELLOW << " --------------------------------------------------- " << NORMAL << endl;
					cout << PURPLE << " --------------------------------------------------- " << NORMAL << endl;
					cout << GREENBOLD << "     generation " << currentGeneration << NORMAL << " done" << endl;
					cout << "     " << GREEN << nbEval << NORMAL << " evaluations in " << (totalTime.count() / 1000.0f)
					     << "s." << endl;
					for (auto &o : stats) {
						cout << "     " << o.first << " : best = " << CYAN << o.second.first
						     << NORMAL " ; avg = " << PURPLE << o.second.second / nbEval << NORMAL << endl;
						o.second.second = 0;
						if (prevStats.count(o.first) == 0) {
							prevStats[o.first] = make_pair(0.0, 0.0);
						}
						if (o.second.first < prevStats.at(o.first).first) {
							cerr << RED << endl
							     << endl
							     << " xxxxxxxx--xxxxxxxxx--xxxxxxxxx ERREUR : prev best for " << o.first << " = "
							     << prevStats.at(o.first).first << "; now = " << o.second.first << NORMAL << endl
							     << endl
							     << endl;
						}
					}
					prevStats = stats;
					if (stats.count("novelty")) {
						stats.at("novelty").first = 0;
					}
					cout << PURPLE << " --------------------------------------------------- " << NORMAL << endl;
					cout << YELLOW << " --------------------------------------------------- " << NORMAL << endl << endl;
				}
				// we save everybody
				savePop();
				saveBests(nbSavedElites);
				// an prepare the next gen
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

	/*********************************************************************************
	 *                   NEXT POP GETTING READY FOR THE DANCEFLOOR
	 ********************************************************************************/
	// Là où qu'on fait les bébés.
	void prepareNextPop() {
		uniform_real_distribution<double> d(0.0, 1.0);
		if (novelty) {
			// first we archive the footprints
			archive.reserve(population.size());
			for (auto &i : population) {
				archive.push_back(make_pair(i.footprint, 1));
			}
			vector<vector<double>> distanceMatrix;
			if (archive.size() > maxArchiveSize) {
				if (verbosity >= 3) {
					cout << " ... merging " << RED << archive.size() << NORMAL << " into " << PURPLE << maxArchiveSize
					     << NORMAL << " footprints ..." << endl;
				}
				distanceMatrix.resize(archive.size());
				for (unsigned int i = 0; i < archive.size(); ++i) {
					distanceMatrix[i].resize(archive.size());
					for (unsigned int j = i + 1; j < archive.size(); ++j) {
						distanceMatrix[i][j] =
						    GA<DNA, Evaluator>::getFootprintDistance(archive[i].first, archive[j].first);
					}
				}
			}
			while (archive.size() > maxArchiveSize) {
				// we merge the closest ones
				int closestId0 = 0;
				int closestId1 = archive.size() - 1;
				double closestDist =
				    GA<DNA, Evaluator>::getFootprintDistance(archive[closestId0].first, archive[closestId1].first);
				for (unsigned int id0 = 0; id0 < archive.size(); ++id0) {
					for (unsigned int id1 = id0 + 1; id1 < archive.size(); ++id1) {
						double dist = distanceMatrix[id0][id1];
						if (dist < closestDist) {
							closestDist = dist;
							closestId0 = id0;
							closestId1 = id1;
						} else if (dist == closestDist) {
							if (d(globalRand) < 0.001) {
								closestDist = dist;
								closestId0 = id0;
								closestId1 = id1;
							}
						}
					}
				}
				vector<vector<double>> mean;
				mean.resize(archive[closestId0].first.size());
				for (size_t j = 0; j < archive[closestId0].first.size(); ++j) {
					for (size_t k = 0; k < archive[closestId0].first[j].size(); ++k) {
						mean[j].push_back((archive[closestId0].first[j][k] * archive[closestId0].second +
						                   archive[closestId1].first[j][k] * archive[closestId1].second) /
						                  (archive[closestId0].second + archive[closestId1].second));
					}
				}
				archive[closestId0].first = mean;
				archive[closestId0].second++;
				archive.erase(archive.begin() + closestId1);
				// update distanceMatrix
				distanceMatrix.erase(distanceMatrix.begin() + closestId1);
				for (auto &dm : distanceMatrix) {
					dm.erase(dm.begin() + closestId1);
				}
				for (unsigned int i = 0; i < (unsigned int)closestId0; ++i) {
					distanceMatrix[i][closestId0] =
					    GA<DNA, Evaluator>::getFootprintDistance(archive[i].first, archive[closestId0].first);
				}
				for (unsigned int i = closestId0 + 1; i < archive.size(); ++i) {
					distanceMatrix[closestId0][i] =
					    GA<DNA, Evaluator>::getFootprintDistance(archive[i].first, archive[closestId0].first);
				}
			}
		}
		if (verbosity >= 3) {
			cerr << " ... copying elites" << endl;
		}
		// then we copy the elite
		vector<Individual<DNA>> nextGen;
		vector<string> objectives;
		for (auto &o : population[0].fitnesses) {
			objectives.push_back(o.first); // we need to know what the objectives are
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
		uniform_int_distribution<int> dint(0, population.size() - 1);
		if (verbosity >= 3) {
			cerr << " ... starting tournaments" << endl;
		}
		// we need to know how much of each objective's representatives we will have
		// if the proportion map has been set, we use it. Else we split equally
		map<string, double> objPop; // nb of ind per objective
		double sum = 0;             // we need to be sure these proportions are normalized
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
			o.second /= sum; // normalization
			if (id++ < (int)objPop.size() - 1) {
				o.second = static_cast<int>(o.second * static_cast<double>(availableSpots));
				cpt += o.second;
			} else { // if it's the last obj, we complete
				o.second = availableSpots - cpt;
			}
		}

		for (auto &o : objPop) {
			unsigned int popGoal = nextGen.size() + static_cast<unsigned int>(o.second);
			while (nextGen.size() < popGoal) {
				vector<Individual<DNA>> tournament;
				for (unsigned int i = 0; i < tournamentSize; ++i) {
					int id = dint(globalRand);
					tournament.push_back(population[id]);
				}
				Individual<DNA> winner = tournament[0];
				for (unsigned int i = 1; i < tournamentSize; ++i) {
					assert(tournament[i].fitnesses.count(o.first));
					if (tournament[i].fitnesses.at(o.first) > winner.fitnesses.at(o.first)) {
						winner = tournament[i];
					}
				}
				if (d(globalRand) < mutationProba) { // mutation
					winner.dna.mutate();
				}
				nextGen.push_back(winner);
			}
		}
		if (verbosity >= 3) {
			cerr << " ... crossovers" << endl;
		}
		population.clear();

		// uniform_int_distribution<int> dint2(firstNonEliteIndex, nextGen.size() - 1);
		for (unsigned int pi = 0; pi < nextGen.size(); ++pi) {
			auto &p1 = nextGen[pi];
			if (pi >= firstNonEliteIndex && d(globalRand) < crossoverProba) {
				unsigned int r = dint(globalRand);
				Individual<DNA> &p2 = nextGen[r];
				population.push_back(Individual<DNA>(p1.dna.crossover(p2.dna)));
			} else
				population.push_back(p1);
		}
	}

	map<string, vector<Individual<DNA>>> getElites(const vector<string> &obj, int n) {
		assert(obj.size() > 0);
		if (verbosity >= 3) {
			cerr << "getElites : nbObj = " << obj.size() << " n = " << n << endl;
		}
		map<string, vector<Individual<DNA>>> elites;
		for (auto &o : obj) {
			elites[o] = vector<Individual<DNA>>();
			for (auto &i : population) {
				if ((int)elites.at(o).size() < n) {
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

	/*********************************************************************************
	 *                         EAT / SAVE / RAVE / REPEAT
	 ********************************************************************************/
	void saveBests(int n) {
		// save n bests dnas for all objectives
		vector<string> objectives;
		for (auto &o : population[0].fitnesses) {
			objectives.push_back(o.first); // we need to know what are the objective functions
		}
		map<string, vector<Individual<DNA>>> elites = getElites(objectives, n);
		stringstream baseName;
		baseName << folder << "/gen" << currentGeneration;
		mkdir(baseName.str().c_str(), 0777);
		if (verbosity >= 3) {
			cerr << "created directory " << baseName.str() << endl;
		}
		for (auto &e : elites) {
			int id = 0;
			for (auto &i : e.second) {
				stringstream fileName;
				fileName << baseName.str() << "/" << e.first << "_" << i.fitnesses.at(e.first) << "_" << id++
				         << ".dna";
				ofstream fs(fileName.str());
				if (!fs) {
					cerr << "Cannot open the output file." << endl;
				}
				fs << i.dna.toJSON().json();
				fs.close();
			}
		}
	}

	void loadPop(string file) {
		ifstream t(file);
		stringstream buffer;
		buffer << t.rdbuf();
		jsonxx::Object o;
		o.parse(buffer.str());
		if (o.has<jsonxx::Number>("generation")) {
			currentGeneration = o.get<jsonxx::Number>("generation");
		} else {
			currentGeneration = 0;
		}
		jsonxx::Array popArray(o.get<jsonxx::Array>("population"));
		population.clear();
		for (size_t i = 0; i < popArray.size(); ++i) {
			jsonxx::Object ind = popArray.get<jsonxx::Object>(i);
			population.push_back(Individual<DNA>(DNA(ind.get<jsonxx::Object>("dna"))));
		}
	}

	void savePop() {
		jsonxx::Object o = Individual<DNA>::popToJSON(population);
		o << "evaluator" << ev.name;
		o << "generation" << currentGeneration;
		stringstream baseName;
		baseName << folder << "/gen" << currentGeneration;
		mkdir(baseName.str().c_str(), 0777);
		stringstream fileName;
		fileName << baseName.str() << "/pop" << currentGeneration << ".pop";
		ofstream file;
		file.open(fileName.str());
		file << o;
		file.close();
	}

	void createFolder(string baseFolder) {
		struct stat sb;
		mkdir(baseFolder.c_str(), 0777);
		auto now = system_clock::now();
		time_t now_c = system_clock::to_time_t(now);
		struct tm *parts = localtime(&now_c);

		stringstream fname;
		fname << ev.name << parts->tm_mday << "_" << parts->tm_mon + 1 << "_";
		int cpt = 0;
		stringstream ftot;
		do {
			ftot.clear();
			ftot.str("");
			ftot << baseFolder << fname.str() << cpt;
			cpt++;
		} while (stat(ftot.str().c_str(), &sb) == 0 && S_ISDIR(sb.st_mode));
		folder = ftot.str();
		mkdir(folder.c_str(), 0777);
	}
};
}
#endif
