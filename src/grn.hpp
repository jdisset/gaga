#ifndef GRN_HPP
#define GRN_HPP
#include <vector>
#include <random>
#include <limits>
#include <iostream>
#include <utility>
#include <unordered_map>
#include "jsonxx/jsonxx.h"
#include "tools.h"

#define BETA_RANGE 40.0
#define DELTA_RANGE 5.0
#define ALPHA_RANGE 100.0
#define MAX_REGULS 50
#define MODIF_RATE 0.9
#define ADD_RATE 0.1
#define DEL_RATE 0.1

#define INIT_CONCENTRATION 0.5
#define ALIGN_TRESHOLD 0.4

using namespace std;

// always keep input as first and output as last
enum proteinType { input = 0, regul = 1, output = 2 };
struct Protein {
	double enh, inh, id, c;
	Protein(double enh, double inh, double id, double c) : enh(enh), inh(inh), id(id), c(c){};
	Protein(const Protein& p) : enh(p.enh), inh(p.inh), id(p.id), c(p.c){};
	Protein() : c(INIT_CONCENTRATION) {
		uniform_real_distribution<double> distribution(0.0, 1.0);
		enh = distribution(globalRand);
		inh = distribution(globalRand);
		id = distribution(globalRand);
	}
	jsonxx::Object toJSON() const {
		jsonxx::Object o;
		char buf[50];
		sprintf(buf, "%a", id);
		o << "id" << buf;
		sprintf(buf, "%a", enh);
		o << "enh" << buf;
		sprintf(buf, "%a", inh);
		o << "inh" << buf;
		return o;
	}
	void reset() { c = INIT_CONCENTRATION; }
	static string typeToString(proteinType t) {
		switch (t) {
			case input:
				return "input";
			case regul:
				return "regul";
			case output:
				return "output";
		}
		return "unknown_type";
	}
	double getDistanceWith(const Protein& p) {
		return sqrt(0.75 * pow((id - p.id), 2) + 0.125 * pow((enh - p.enh), 2) + 0.125 * pow((inh - p.inh), 2));
	}
};

class GRN {
 protected:
	vector<map<string, Protein>> proteins;
	map<Protein*, map<Protein*, pair<double, double>>> signatures;
	double beta = 1.0;
	double delta = 1.0;
	double alpha = 1.0;
	int currentStep = 0;

 public:
	GRN(const GRN& grn)
	    : proteins(grn.proteins),
	      beta(grn.beta),
	      delta(grn.delta),
	      alpha(grn.alpha),
	      currentStep(grn.currentStep) {
		updateSignatures();
	}

	GRN() {
		proteins.resize(3);
		updateSignatures();
	}

	GRN(jsonxx::Object o) {
		proteins.resize(3);
		assert(o.has<jsonxx::String>("beta"));
		assert(o.has<jsonxx::String>("delta"));
		assert(o.has<jsonxx::String>("alpha"));
		assert(o.has<jsonxx::Object>("proteins"));
		sscanf(o.get<jsonxx::String>("beta").c_str(), "%lf", &beta);
		sscanf(o.get<jsonxx::String>("delta").c_str(), "%lf", &delta);
		sscanf(o.get<jsonxx::String>("alpha").c_str(), "%lf", &alpha);
		for (size_t t = input; t <= output; ++t) {
			const map<string, jsonxx::Value*>& prots =
			    o.get<jsonxx::Object>("proteins")
			        .get<jsonxx::Object>(Protein::typeToString((proteinType)t))
			        .kv_map();
			for (auto p = prots.begin(); p != prots.end(); ++p) {
				jsonxx::Object jsonProt = p->second->get<jsonxx::Object>();
				double enh, inh, id;
				sscanf(jsonProt.get<jsonxx::String>("enh").c_str(), "%lf", &enh);
				sscanf(jsonProt.get<jsonxx::String>("inh").c_str(), "%lf", &inh);
				sscanf(jsonProt.get<jsonxx::String>("id").c_str(), "%lf", &id);
				proteins[t][p->first] = Protein(enh, inh, id, INIT_CONCENTRATION);
			}
		}
		updateSignatures();
	}


	void swap(GRN& first, GRN& second) {
		using std::swap;
		swap(first.proteins, second.proteins);
		swap(first.signatures, second.signatures);
		swap(first.beta, second.beta);
		swap(first.delta, second.delta);
		swap(first.alpha, second.alpha);
		swap(first.currentStep, second.currentStep);
	}

	GRN& operator=(const GRN& other) {
		GRN tmp(other);
		swap(*this, tmp);
		return *this;
	}

	/**************************************
	 *           GET & SET
	 *************************************/
	size_t getProteinSize(proteinType t) const { return proteins[t].size(); }
	Protein& getProtein(proteinType t, string name) {
		assert(proteins[t].count(name) > 0);
		return proteins[t][name];
	}
	void reset() {
		updateSignatures();
		for (size_t t0 = 0; t0 < proteins.size(); ++t0) {
			for (auto p0 = proteins[t0].begin(); p0 != proteins[t0].end(); ++p0) {
				p0->second.reset();
			}
		}
	}
	vector<map<string, Protein>> getProteins() const { return proteins; }
	double getBeta() { return beta; }
	double getDelta() { return delta; }
	double getAlpha() { return alpha; }
	void setBeta(double b) { beta = b; }
	void setDelta(double d) { delta = d; }
	void setAlpha(double a) { alpha = a; }
	void setProteinConcentration(string name, proteinType t, double c) { proteins[t][name].c = c; }
	double getProteinConcentration(string name, proteinType t) const { return proteins[(size_t)t].at(name).c; }

	/**************************************
	 *          ADDING PROTEINS
	 *************************************/
	// good for body building
	void addProtein(proteinType t, string name, Protein p) { proteins[t].insert(make_pair(name, Protein(p))); }
	void addProteins(map<string, Protein>& prots, proteinType t) {
		for (auto p = prots.begin(); p != prots.end(); ++p) {
			addProtein(t, p->first, p->second);
		}
	}
	void randomReguls(unsigned int n) {
		ostringstream name;
		proteins[regul].clear();
		for (unsigned int i = 0; i < n; ++i) {
			name.str("");
			name.clear();
			name << "r" << i;
			addProtein(regul, name.str(), Protein());
		}
	}

	/**************************************
	 *         UPDATE & STEP
	 *************************************/
	void updateSignatures() {
		for (size_t t0 = 0; t0 < proteins.size(); ++t0) {
			for (auto& p0 : proteins[t0]) {
				map<Protein*, pair<double, double>> tmpSign;
				for (size_t t1 = 0; t1 < proteins.size(); ++t1) {
					for (auto& p1 : proteins[t1]) {
						double e = exp(-beta * BETA_RANGE * abs(p0.second.enh - p1.second.id));
						double i = exp(-beta * BETA_RANGE * abs(p0.second.inh - p1.second.id));
						tmpSign[&(p1.second)] = make_pair(max(e, 0.0), max(i, 0.0));
					}
				}
				signatures[&(p0.second)] = tmpSign;
			}
		}
	}

	void step() {
		vector<map<string, Protein>> tmp = proteins;
		for (size_t t0 = regul; t0 < proteins.size(); ++t0) {
			for (auto& p0 : proteins[t0]) {
				double enhance = 0;
				double inhibit = 0;
				double n = 0;
				for (size_t t1 = 0; t1 < output; ++t1) {
					for (auto& p1 : proteins[t1]) {
						pair<double, double> sign = signatures.at(&p1.second).at(&p0.second);
						enhance += p1.second.c * sign.first;
						inhibit += p1.second.c * sign.second;
						n++;
					}
				}
				double dc = abs(p0.second.c - 0.5);
				double dt = (1.0 - (dc / (dc + DELTA_RANGE * delta))) * alpha * ALPHA_RANGE * (enhance - inhibit) / n;
				dt *= (1.0 - (delta * abs(dt)));
				double val = p0.second.c + dt;
				tmp[t0].at(p0.first).c = max(0.0, min(1.0, val));
			}
		}
		for (size_t t0 = regul; t0 < proteins.size(); ++t0) {
			for (auto& p0 : proteins[t0]) {
				p0.second.c = tmp[t0].at(p0.first).c;
			}
		}
		currentStep++;
	}

	void updateRegulNames() {
		int id = 0;
		map<string, Protein> newReguls;
		for (auto& i : proteins[regul]) {
			ostringstream name;
			name << "r" << id++;
			newReguls[name.str()] = i.second;
		}
		proteins[regul] = newReguls;
	};

	/**************************************
	 *           GA RELATED
	 *************************************/
	void mutate() {
		uniform_real_distribution<double> d(0.0, 1.0);
		vector<string> reguls;
		ostringstream name;
		int nbReguls = getProteinSize(regul);
		for (int i = 0; i < nbReguls; ++i) {
			name.str("");
			name.clear();
			name << "r" << i;
			reguls.push_back(name.str());
		}
		// modification
		if (d(globalRand) <= MODIF_RATE / (MODIF_RATE + ADD_RATE + DEL_RATE)) {
			if (nbReguls > 0) {
				uniform_int_distribution<int> di(0, nbReguls - 1);
				int v = di(globalRand);
				proteins[regul][reguls[v]] = Protein();
			}
		}
		// ajout
		if (d(globalRand) <= ADD_RATE / (MODIF_RATE + ADD_RATE + DEL_RATE)) {
			name.str("");
			name.clear();
			name << "r" << nbReguls;
			addProtein(regul, name.str(), Protein());
		}
		nbReguls = getProteinSize(regul);
		// suppression
		if (d(globalRand) <= DEL_RATE / (MODIF_RATE + ADD_RATE + DEL_RATE)) {
			if (nbReguls > 0) {
				uniform_int_distribution<int> di(0, nbReguls - 1);
				auto it = proteins[regul].begin();
				advance(it, di(globalRand));
				proteins[regul].erase(it);
				updateRegulNames();
			}
		}
		updateSignatures();
	}

	GRN crossover(const GRN& other) { return GRN::crossover(*this, other); }

	static GRN crossover(const GRN& g0, const GRN& g1) {
		assert(g0.proteins.size() == g1.proteins.size());
		assert(g0.proteins[input].size() == g1.proteins[input].size());
		assert(g0.proteins[output].size() == g1.proteins[output].size());
		GRN offspring;
		uniform_int_distribution<int> distrib(0, 1);
		uniform_real_distribution<double> doubleDistrib(0.0, 1.0);
		// 50/50 for beta, delta, alpha, inputs and outputs
		offspring.beta = distrib(globalRand) ? g0.beta : g1.beta;
		offspring.delta = distrib(globalRand) ? g0.delta : g1.delta;
		offspring.alpha = distrib(globalRand) ? g0.alpha : g1.alpha;
		offspring.proteins[input] = g0.proteins[input];
		offspring.proteins[output] = g0.proteins[output];
		for (auto& i : g1.proteins[input])
			if (distrib(globalRand)) offspring.proteins[input][i.first] = i.second;
		for (auto& i : g1.proteins[output])
			if (distrib(globalRand)) offspring.proteins[output][i.first] = i.second;
		// find closest pairs
		map<string, Protein> r0 = g0.proteins[regul];
		map<string, Protein> r1 = g1.proteins[regul];
		vector<pair<Protein, Protein>> aligned;  // first = g0's proteins, second = g1's
		double minDist = 0;
		while (minDist < ALIGN_TRESHOLD && r0.size() > 0 && r1.size() > 0 && aligned.size() < MAX_REGULS) {
			pair<string, string> closest;
			minDist = numeric_limits<double>::infinity();
			for (auto i = r0.begin(); i != r0.end(); ++i) {
				for (auto j = r1.begin(); j != r1.end(); ++j) {
					double dist = i->second.getDistanceWith(j->second);
					if (dist < minDist) {
						closest = make_pair(i->first, j->first);
						minDist = dist;
					}
				}
			}
			if (minDist < ALIGN_TRESHOLD) {
				aligned.push_back(make_pair(r0.at(closest.first), r1.at(closest.second)));
				r0.erase(closest.first);
				r1.erase(closest.second);
			}
		}
		// regul : 50/50 with aligned
		int id = offspring.proteins[regul].size();
		for (auto& i : aligned) {
			ostringstream name;
			name << "r" << id++;
			if (distrib(globalRand))
				offspring.proteins[regul][name.str()] = i.first;
			else
				offspring.proteins[regul][name.str()] = i.second;
		}
		// append the rest (about 1/2 chance)
		for (auto& i : r0) {
			if (offspring.proteins[regul].size() < MAX_REGULS) {
				if (doubleDistrib(globalRand) < 0.7) {
					ostringstream name;
					name << "r" << id++;
					offspring.proteins[regul][name.str()] = i.second;
				}
			}
		}
		for (auto& i : r1) {
			if (offspring.proteins[regul].size() < MAX_REGULS) {
				if (doubleDistrib(globalRand) < 0.7) {
					ostringstream name;
					name << "r" << id++;
					offspring.proteins[regul][name.str()] = i.second;
				}
			}
		}
		offspring.updateSignatures();
		return offspring;
	}

	/**************************************
	 *              toJSON
	 *************************************/
	jsonxx::Object toJSON() const {
		jsonxx::Object prot;
		for (size_t t = 0; t < proteins.size(); ++t) {
			jsonxx::Object pr;
			for (auto p = proteins[t].begin(); p != proteins[t].end(); ++p) {
				pr << p->first << p->second.toJSON();
			}
			prot << Protein::typeToString((proteinType)t) << pr;
		}

		jsonxx::Object o;
		char buf[50];
		sprintf(buf, "%a", beta);
		o << "beta" << buf;
		sprintf(buf, "%a", delta);
		o << "delta" << buf;
		sprintf(buf, "%a", alpha);
		o << "alpha" << buf;
		o << "proteins" << prot;
		return o;
	}
};
#endif
