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

#include <vector>
#include <chrono>
#include <fstream>
#include <ctime>
#include <unordered_set>
#include <unordered_map>
#include <deque>
#include <sys/stat.h>
#include <sys/types.h>
#include "jsonxx/jsonxx.h"
#include "tools.h"


//#define CLUSTER
#ifdef CLUSTER
#include <mpi.h>
#endif

#define NOVELTY false  // is novelty enabled ?
// Notes about novelty
// novelty works with footprints. A footprint is just a vector of vector of doubles
// it is recommended that those doubles are within a same order of magnitude
// each vector of double is a "snapshot" : it represents the state of the evaluation of one individual
// at a certain time. Thus, a complete footprint is a combination (a vector) of one or more snapshot
// taken at different points in the simulation. Snapshot must be of same size accross individuals
// footprint must be set in the evaluator (see example)

#define NB_ELITE 1  // NB of elites to keep
#define POP_SIZE 320
#define TOURNAMENT_SIZE 3
#define CROSSOVER_PROBA 0.3
#define MUTATION_PROBA 0.5
#define NB_GEN 4000             // nb of generations
#define MAX_ARCHIVE_SIZE 2000   // nb of footprints to keep for novelty computations
#define K_KNN 15                // size of the neighbourhood for novelty
#define FOLDER_PATH "../evos/"  // where to save the results

using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using std::chrono::system_clock;
using namespace std;

typedef vector<vector<double>> fpType;          // footprints for novelty
typedef vector<pair<fpType, double>> archType;  // collection of footprints

/*********************************
 * Individual representation
 * ******************************/
// DNA class must have :
// mutate()
// crossover(DNA& other)
// toJson()

template <typename DNA> struct Individual {
	Individual(const DNA& d) : dna(d) {}
	Individual(const jsonxx::Object& o) {
		dna = DNA(o.get<jsonxx::Object>("dna"));
		jsonxx::Object fpObj(o.get<jsonxx::Object>("footprint"));
		jsonxx::Array fpArr(fpObj.get<jsonxx::Array>("array"));
		size_t nbFp = fpObj.get<jsonxx::Number>("n");
		footprint.resize(nbFp);
		for (size_t i = 0; i < fpArr.size(); ++i) {
			double val;
			sscanf(fpArr.get<jsonxx::String>(i).c_str(), "%lf", &val);
			footprint[nbFp * i / fpArr.size()].push_back(val);
		}

		const map<string, jsonxx::Value*>& fitMap = o.get<jsonxx::Object>("fitnesses").kv_map();
		for (auto& f : fitMap) {
			double val;
			sscanf(f.second->get<jsonxx::String>().c_str(), "%lf", &val);
			fitnesses[f.first] = val;
		}
	}

	DNA dna;
	map<string, double> fitnesses;  // map {"fitnessCriterName" -> "fitnessValue"}
	fpType footprint;               // individual's footprint for novelty computation
	bool evaluated = false;

	// Exports individual to json
	jsonxx::Object toJSON() const {
		jsonxx::Object fitObject;
		for (auto& f : fitnesses) {
			char buf[50];
			sprintf(buf, "%a", f.second);
			fitObject << f.first << buf;
		}

		int fpSize = 0;
		jsonxx::Array fpArr;
		for (size_t i = 0; i < footprint.size(); ++i) {
			fpSize = footprint.size();
			for (size_t j = 0; j < footprint[i].size(); ++j) {
				char buf[50];
				sprintf(buf, "%a", footprint[i][j]);
				fpArr << buf;
			}
		}
		// TODO: patch this shit (low priority)
		// there seems to be a bug with jsonxx matrices...
		// so we use a flat array instead
		jsonxx::Object fpObj;
		fpObj << "array" << fpArr;
		fpObj << "n" << fpSize;
		jsonxx::Object o;
		o << "dna" << dna.toJSON();
		o << "fitnesses" << fitObject;
		o << "footprint" << fpObj;
		return o;
	}

	// Exports a vector of individual to json
	static jsonxx::Object popToJSON(const vector<Individual<DNA>>& p) {
		jsonxx::Object o;
		jsonxx::Array popArray;
		for (auto& i : p) {
			popArray << i.toJSON();
		}
		o << "population" << popArray;
		return o;
	}

	// Loads a vector of individual from json
	static vector<Individual<DNA>> loadPopFromJSON(const jsonxx::Object& o) {
		vector<Individual<DNA>> res;
		jsonxx::Array popArray = o.get<jsonxx::Array>("population");
		for (size_t i = 0; i < popArray.size(); ++i) {
			res.push_back(Individual<DNA>(popArray.get<jsonxx::Object>(i)));
		}
		return res;
	}
};

template <typename DNA, typename Evaluator> class GA {
	unsigned int popSize = POP_SIZE;
	Evaluator ev;
	archType archive;  // for novelty
	vector<Individual<DNA>> population;
	int currentGeneration = 0;
	string folder;
	// openmp/mpi stuff
	int procId = 0;
	int nbProcs = 1;

	// list of options to be passed to the dna constructor
	map<string, pair<double, double>> stats;  // fitnesses stats. First = best, second = avg

	// proportions contains the relative weights of the objectives
	// if an objective is not present here but used in the evaluation anyway
	// a defaut uniform ponderation will be used
	map<string, double> proportions = {{"baseObj", 0.25}, {"novelty", 0.75}};

	/*********************************************************************************
	 *                          NOVELTY RELATED METHODS
	 ********************************************************************************/
	static double getFootprintDistance(const vector<vector<double>>& f0, const vector<vector<double>>& f1) {
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
	static double computeAvgDist(unsigned int k, const archType& arch, const fpType& fp) {
		archType knn;
		double avgDist = 0;
		if (arch.size() > k) {
			for (unsigned int i = 0; i < k; ++i) {
				knn.push_back(arch[i]);
			}
			for (unsigned int i = k; i < arch.size(); ++i) {
				double d0 = GA<DNA,Evaluator>::getFootprintDistance(fp, arch[i].first);
				for (auto j = knn.begin(); j != knn.end(); ++j) {
					if (d0 < GA<DNA,Evaluator>::getFootprintDistance(fp, (*j).first)) {
						knn.erase(j);
						knn.push_back(arch[i]);
						break;
					}
				}
			}
			double divisor = 0;
			for (auto& i : knn) {
				double tmpD = GA<DNA,Evaluator>::getFootprintDistance(fp, i.first);
				double divCoef = exp(-2.0 * tmpD) * i.second;
				avgDist += tmpD * divCoef;
				divisor += divCoef;
				// avoid stagnation
				if (tmpD == 0.0 && i.second >= (double)k) return 0.0;
			}
			if (divisor == 0.0) {  // if this happens there's probably a bug somewhere
				avgDist = 0;
			} else {
				avgDist /= divisor;
			}
		}
		return avgDist;
	}
	void computeIndNovelty(Individual<DNA>& i) {
		i.fitnesses["novelty"] = GA<DNA,Evaluator>::computeAvgDist(K_KNN, archive, i.footprint);
	}
	// panpan cucul
	static string footprintToString(const vector<vector<double>>& f) {
		ostringstream res;
		for (auto& p : f) {
			res << NORMAL << "     [" << GREY;
			for (const double& v : p) {
				res << " " << setw(4) << setprecision(3) << fixed << v;
			}
			res << NORMAL << " ]" << endl;
		}
		return res.str();
	}


 public:

	GA(string baseFolder = FOLDER_PATH) {
#ifdef CLUSTER
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
			createFolder(baseFolder);
#ifdef CLUSTER
		}
#endif
	}

	void createFolder(string baseFolder) {
		struct stat sb;
		mkdir(baseFolder.c_str(), 0777);
		auto now = system_clock::now();
		time_t now_c = system_clock::to_time_t(now);
		struct tm* parts = localtime(&now_c);

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

	/*********************************************************************************
	 *                          START THE BOUZIN
	 ********************************************************************************/
	// "Vroum vroum"
	void start() {
		bool finished = false;
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
				MPI_Probe(0, 0, MPI_COMM_WORLD, &status);  // we want to know its size
				MPI_Get_count(&status, MPI_CHAR, &strLength);
				char* popChar = new char[strLength + 1];
				MPI_Recv(popChar, strLength, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				// and we dejsonize !
				jsonxx::Object o;
				bool success = o.parse(popChar);
				if (!success) {
					cerr << "parse failed. Str = " << popChar << endl;
				}
				population = Individual<DNA>::loadPopFromJSON(o);  // welcome bros!
				cerr << endl << "Proc " << procId << " : reception of " << population.size() << " new individuals !"
				     << endl;
			}
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
					// STATS & PRINTING:
					milliseconds totalTime = std::chrono::duration_cast<milliseconds>(t1 - t0);
					ostringstream indStatus;
					unordered_set<string> best;
					for (auto& o : population[i].fitnesses) {
						if (stats.count(o.first) == 0) {
							stats[o.first] = make_pair(0.0, 0.0);
						}
						stats.at(o.first).second += o.second;
						if (o.second > stats.at(o.first).first) {  // new besta
							best.insert(o.first);
							stats.at(o.first).first = o.second;
						}
					}
					indStatus << endl << " Proc " << PURPLE << procId << NORMAL << " : Ind " << YELLOW << setw(3) << i
					          << NORMAL << " evaluated in " GREEN << setw(3) << setprecision(1) << fixed
					          << totalTime.count() / 1000.0f << NORMAL << "s :";
					for (auto& o : population[i].fitnesses) {
						indStatus << " " << o.first << " : ";
						if (best.count(o.first)) {
							indStatus << CYANBOLD;
						} else {
							indStatus << GREEN;
						}
						double val = o.second;
						indStatus << setw(10) << setprecision(4) << scientific << val << NORMAL << ";";
					}
					if (best.size() > 0) {
						indStatus << endl;
						indStatus << GA<DNA,Evaluator>::footprintToString(population[i].footprint);
						indStatus << endl;
					}
					cerr << indStatus.str();
				}
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
					char* popChar = new char[strLength + 1];
					MPI_Recv(popChar, strLength + 1, MPI_BYTE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					// and we dejsonize!
					jsonxx::Object o;
					bool success = o.parse(popChar);
					if (!success) {
						cerr << endl << "parse failed. Str = " << popChar << endl;
					}
					vector<Individual<DNA>> batch = Individual<DNA>::loadPopFromJSON(o);
					population.insert(population.end(), batch.begin(), batch.end());
					cerr << endl << "Proc " << procId << " : reception of " << batch.size()
					     << " treated individuals from proc " << source << endl;
				}
				// now we update novelty
				if (NOVELTY) {
					for (auto& ind : population) {
						nbEval++;
						double avgD = GA<DNA,Evaluator>::computeAvgDist(K_KNN, archive, ind.footprint);
						ind.fitnesses["novelty"] = avgD;
						if (stats.count("novelty") == 0) {
							stats["novelty"] = make_pair(0.0, 0.0);
						}
						stats.at("novelty").second += ind.fitnesses.at("novelty");
						if (avgD > stats.at("novelty").first) {  // new best
							stats.at("novelty").first = avgD;
							cerr << " New best novelty: " << CYAN << avgD << NORMAL << endl;
							cerr << footprintToString(ind.footprint);
						}
					}
				}
#endif
				// the end of a generation
				auto tg1 = high_resolution_clock::now();
				milliseconds totalTime = std::chrono::duration_cast<milliseconds>(tg1 - tg0);
				cerr << endl;
				cerr << YELLOW << " --------------------------------------------------- " << NORMAL << endl;
				cerr << PURPLE << " --------------------------------------------------- " << NORMAL << endl;
				cerr << GREENBOLD << "     generation " << currentGeneration << NORMAL << " done" << endl;
				cerr << "     " << GREEN << nbEval << NORMAL << " evaluations in " << (totalTime.count() / 1000.0f)
				     << "s." << endl;
				for (auto& o : stats) {
					cerr << "     " << o.first << " : best = " << CYAN << o.second.first << NORMAL " ; avg = " << PURPLE
					     << o.second.second / nbEval << NORMAL << endl;
					o.second.second = 0;
				}
				if (stats.count("novelty")) {
					stats.at("novelty").first = 0;
				}
				cerr << PURPLE << " --------------------------------------------------- " << NORMAL << endl;
				cerr << YELLOW << " --------------------------------------------------- " << NORMAL << endl << endl;
				// we save everybody
				savePop();
				// and the N bests of each objectives
				saveBests(2);
				// better prepare the next one!
				prepareNextPop();
				finished = (currentGeneration++ >= NB_GEN);
#ifdef CLUSTER
			}
#endif
		}
	}

	/*********************************************************************************
	 *                   NEXT POP GETTING READY FOR THE DANCEFLOOR
	 ********************************************************************************/
	// Là où qu'on fait les bébés.
	void prepareNextPop() {
		uniform_real_distribution<double> d(0.0, 1.0);
		if (NOVELTY) {
			// first we archive the footprints
			archive.reserve(population.size());
			for (auto& i : population) {
				archive.push_back(make_pair(i.footprint, 1));
			}
			vector<vector<double>> distanceMatrix;
			if (archive.size() > MAX_ARCHIVE_SIZE) {
				cout << " ... merging " << RED << archive.size() << NORMAL << " into " << PURPLE << MAX_ARCHIVE_SIZE
				     << NORMAL << " footprints ..." << endl;
				distanceMatrix.resize(archive.size());
				for (unsigned int i = 0; i < archive.size(); ++i) {
					distanceMatrix[i].resize(archive.size());
					for (unsigned int j = i + 1; j < archive.size(); ++j) {
						distanceMatrix[i][j] = GA<DNA,Evaluator>::getFootprintDistance(archive[i].first, archive[j].first);
					}
				}
			}
			while (archive.size() > MAX_ARCHIVE_SIZE) {
				// we merge the closest ones
				int closestId0 = 0;
				int closestId1 = archive.size() - 1;
				double closestDist =
				    GA<DNA,Evaluator>::getFootprintDistance(archive[closestId0].first, archive[closestId1].first);
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
				for (auto& dm : distanceMatrix) {
					dm.erase(dm.begin() + closestId1);
				}
				for (unsigned int i = 0; i < (unsigned int)closestId0; ++i) {
					distanceMatrix[i][closestId0] =
					    GA<DNA,Evaluator>::getFootprintDistance(archive[i].first, archive[closestId0].first);
				}
				for (unsigned int i = closestId0 + 1; i < archive.size(); ++i) {
					distanceMatrix[closestId0][i] =
					    GA<DNA,Evaluator>::getFootprintDistance(archive[i].first, archive[closestId0].first);
				}
			}
		}
		cerr << " ... copying elites" << endl;
		// then we copy the elite
		vector<Individual<DNA>> nextGen;
		vector<string> objectives;
		for (auto& o : population[0].fitnesses) {
			objectives.push_back(o.first);  // we need to know what the objectives are
		}
		map<string, vector<Individual<DNA>>> elites = getElites(objectives, NB_ELITE);
		// we put the elites in the nextGen
		for (auto& e : elites) {
			for (auto& i : e.second) {
				nextGen.push_back(i);
			}
		}
		unsigned int nbElites = nextGen.size();
		// now we can start the tournament
		uniform_int_distribution<int> dint(0, population.size() - 1);
		cerr << " ... starting tournaments" << endl;
		// we need to know how much of each objective's representatives we will have
		// if the proportion map has been set, we use it. Else we split equally
		map<string, double> objPop;  // nb of ind per objective
		double sum = 0;              // we need to be sure these proportions are normalized
		for (auto& o : objectives) {
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
		for (auto& o : objPop) {
			o.second /= sum;  // normalization
			if (id++ < (int)objPop.size() - 1) {
				o.second = static_cast<int>(o.second * static_cast<double>(availableSpots));
				cpt += o.second;
			} else {  // if it's the last obj, we complete
				o.second = availableSpots - cpt;
			}
		}

		for (auto& o : objPop) {
			unsigned int popGoal = nextGen.size() + static_cast<unsigned int>(o.second);
			while (nextGen.size() < popGoal) {
				vector<Individual<DNA>> tournament;
				for (int i = 0; i < TOURNAMENT_SIZE; ++i) {
					int id = dint(globalRand);
					tournament.push_back(population[id]);
				}
				Individual<DNA> winner = tournament[0];
				for (int i = 1; i < TOURNAMENT_SIZE; ++i) {
					assert(tournament[i].fitnesses.count(o.first));
					if (tournament[i].fitnesses.at(o.first) > winner.fitnesses.at(o.first)) {
						winner = tournament[i];
					}
				}
				if (d(globalRand) < MUTATION_PROBA) {  // mutation
					winner.dna.mutate();
				}
				nextGen.push_back(winner);
			}
		}
		cerr << " ... crossovers" << endl;
		population.clear();

		for (unsigned int pi = 0; pi < nextGen.size(); ++pi) {
			auto& p1 = nextGen[pi];
			if (pi > nbElites && d(globalRand) < CROSSOVER_PROBA) {
				unsigned int r = dint(globalRand);
				Individual<DNA>& p2 = nextGen[r];
				population.push_back(Individual<DNA>(p1.dna.crossover(p2.dna)));
			} else
				population.push_back(p1);
		}
	}

	map<string, vector<Individual<DNA>>> getElites(const vector<string>& obj, int n) {
		assert(obj.size() > 0);
		map<string, vector<Individual<DNA>>> elites;
		for (auto& o : obj) {
			elites[o] = vector<Individual<DNA>>();
			for (auto& i : population) {
				if ((int)elites.at(o).size() < n) {
					// we push the first inds without looking at their fitness
					elites.at(o).push_back(i);
				} else {
					for (auto j = elites.at(o).begin(); j != elites.at(o).end(); ++j) {
						// but if we already have enough elites we need to check if the one we have
						// really are the best (and replace them if they're not)
						if (j->fitnesses.at(o) < i.fitnesses.at(o)) {
							elites.at(o).erase(j);
							elites.at(o).push_back(i);
							break;
						}
					}
				}
			}
		}
		return elites;
	}

	/*********************************************************************************
	 *                         EAT / SAVE / RAVE / REPEAT
	 ********************************************************************************/
	// and stuff...
	void saveBests(int n) {
		// save n bests for all objectives
		vector<string> objectives;
		for (auto& o : population[0].fitnesses) {
			objectives.push_back(o.first);  // we need to know what are the objective functions
		}
		map<string, vector<Individual<DNA>>> elites = getElites(objectives, n);
		stringstream baseName;
		baseName << folder << "/gen" << currentGeneration;
		mkdir(baseName.str().c_str(), 0777);
		cerr << "created directory " << baseName.str() << endl;
		for (auto& e : elites) {
			int id = 0;
			for (auto& i : e.second) {
				stringstream fileName;
				fileName << baseName.str() << "/" << e.first << "_" << i.fitnesses.at(e.first) << "_" << id++
				         << ".dna";
				i.dna.saveToFile(fileName.str());
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
		stringstream popJson;
		jsonxx::Array popArray;
		for (auto& i : population) {
			jsonxx::Object ind;
			ind << "dna" << i.dna.toJSON();
			popArray << ind;
		}
		jsonxx::Object o;
		o << "evaluator" << ev.name;
		o << "generation" << currentGeneration;
		o << "population" << popArray;
		stringstream baseName;
		baseName << folder << "/gen" << currentGeneration;
		mkdir(baseName.str().c_str(), 0777);
		cerr << "created folder " << baseName.str() << endl;
		stringstream fileName;
		fileName << baseName.str() << "/pop" << currentGeneration << ".pop";
		ofstream file;
		file.open(fileName.str());
		file << o;
		file.close();
	}
};

#endif
