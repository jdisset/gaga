#ifndef PROTEIN_HPP
#define PROTEIN_HPP

#include <assert.h>
#include <array>
#include <random>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>
#include "common.h"

#define MULTIPLE_MUTATION_PROBA 0.1
template <unsigned int nbCoords, typename CoordsType = double, int minCoord = 0,
          int maxCoord = 1>
struct Protein {
	using json = nlohmann::json;

	std::array<CoordsType, nbCoords>
	    coords{};                       // proteins coords (id, enh, inh for example)
	double c = INIT_CONCENTRATION;      // current concentration
	double prevc = INIT_CONCENTRATION;  // previous concentration

	bool operator==(const Protein &b) const {
		for (size_t i = 0; i < nbCoords; ++i)
			if (coords[i] != b.coords[i]) return false;
		return c == b.c;
	}

	bool operator!=(const Protein &b) const { return !(*this == b); }

	// switching between integral or real random distribution
	template <typename T = CoordsType>
	typename std::enable_if<!std::is_integral<T>::value, T>::type getRandomCoord() {
		std::uniform_real_distribution<double> distribution(static_cast<double>(minCoord),
		                                                    static_cast<double>(maxCoord));
		return static_cast<CoordsType>(distribution(grnRand));
	}
	template <typename T = CoordsType>
	typename std::enable_if<std::is_integral<T>::value, T>::type getRandomCoord() {
		std::uniform_int_distribution<int> distribution(minCoord, maxCoord);
		return static_cast<CoordsType>(distribution(grnRand));
	}

	Protein(const decltype(coords) &co, double conc) : coords(co), c(conc), prevc(conc) {
		for (auto &coo : coords) {
			coo = std::max(static_cast<CoordsType>(minCoord),
			               std::min(static_cast<CoordsType>(maxCoord), coo));
		}
	}
	Protein(const Protein &p) : coords(p.coords), c(p.c), prevc(p.prevc){};
	Protein() {
		// Constructs a protein with random coords
		for (auto &i : coords) i = getRandomCoord();
	}

	explicit Protein(const json &o) {
		// constructs a protein from a json object
		assert(o.count("coords"));
		assert(o.count("c"));
		c = o.at("c");
		if (o.count("pc")) prevc = o.at("pc");
		auto vcoords = o.at("coords").get<std::vector<CoordsType>>();
		assert(vcoords.size() == coords.size());
		for (size_t i = 0; i < vcoords.size(); ++i) coords[i] = vcoords[i];
	}

	void reset() {
		c = INIT_CONCENTRATION;
		prevc = c;
	}

	void mutate() {
		std::uniform_int_distribution<int> dInt(0, nbCoords);
		int mutated = dInt(grnRand);
		coords[mutated] = getRandomCoord();
	}

	json toJSON() const {
		json o;
		o["coords"] = coords;
		o["c"] = c;
		o["pc"] = prevc;
		return o;
	}

	static constexpr double getMaxDistance() {
		return sqrt(std::pow((maxCoord - minCoord), 2) * nbCoords);
	}

	double getDistanceWith(const Protein &p) {
		double sum = 0;
		for (size_t i = 0; i < nbCoords; ++i) {
			sum += std::pow(
			    static_cast<double>(coords.at(i)) - static_cast<double>(p.coords.at(i)), 2);
		}
		return sqrt(sum) / getMaxDistance();
	}
};

template <unsigned int nbCoords, typename CoordsType = double, int minCoord = 0,
          int maxCoord = 1>
struct HiProtein {
	using json = nlohmann::json;
	using Base = Protein<nbCoords, CoordsType, minCoord, maxCoord>;
	static constexpr int getMinCoord() { return minCoord; }
	static constexpr int getMaxCoord() { return maxCoord; }
	std::array<CoordsType, nbCoords>
	    coords{};                       // proteins coords (id, enh, inh for example)
	double c = INIT_CONCENTRATION;      // current concentration
	double prevc = INIT_CONCENTRATION;  // previous concentration
	bool input = false;
	bool output = false;
	bool modifiable = true;

	bool operator==(const HiProtein &b) const {
		for (size_t i = 0; i < nbCoords; ++i)
			if (coords[i] != b.coords[i]) return false;
		if (input != b.input || output != b.output) return false;
		return b.modifiable == modifiable && c == b.c;
	}

	HiProtein(const decltype(coords) &co, double conc, bool i, bool o, bool m = true)
	    : coords(co), c(conc), prevc(conc), input(i), output(o), modifiable(m) {
		for (auto &coo : coords) {
			coo = std::max(static_cast<CoordsType>(minCoord),
			               std::min(static_cast<CoordsType>(maxCoord), coo));
		}
	}
	HiProtein(const HiProtein &p)
	    : coords(p.coords),
	      c(p.c),
	      prevc(p.prevc),
	      input(p.input),
	      output(p.output),
	      modifiable(p.modifiable){};

	HiProtein() {
		// Constructs a protein with random coords and random I/O
		for (auto &i : coords) i = getRandomCoord();
		std::uniform_int_distribution<int> dInt(0, 1);
		input = dInt(grnRand);
		output = dInt(grnRand);
		modifiable = true;
	}

	explicit HiProtein(const json &o) {
		// constructs a protein from a json object
		assert(o.count("coords"));
		assert(o.count("c"));
		assert(o.count("I"));
		assert(o.count("O"));
		assert(o.count("M"));
		c = o.at("c");
		input = o.at("I");
		output = o.at("O");
		modifiable = o.at("M");
		if (o.count("pc")) prevc = o.at("pc");
		auto vcoords = o.at("coords").get<std::vector<CoordsType>>();
		assert(vcoords.size() == coords.size());
		for (size_t i = 0; i < vcoords.size(); ++i) coords[i] = vcoords[i];
	}

	json toJSON() const {
		json o;
		o["coords"] = coords;
		o["c"] = c;
		o["pc"] = prevc;
		o["I"] = input;
		o["O"] = output;
		o["M"] = modifiable;
		return o;
	}

	void setConcentration(double con) {
		prevc = c;
		c = con;
	}

	// void mutate() {
	// std::uniform_int_distribution<size_t> dInt(0, nbCoords + 2);
	// size_t mutated = dInt(grnRand);
	// if (mutated < coords.size())
	// coords[mutated] = getRandomCoord();
	// else if (modifiable) {
	// if (mutated == coords.size())
	// input = !input;
	// else
	// output = !output;
	//}
	//}
	void mutate() {
		std::uniform_int_distribution<size_t> dInt(0, coords.size() - 1);
		size_t mutated = dInt(grnRand);
		coords[mutated] = getRandomCoord();
		if (modifiable) {
			std::uniform_int_distribution<int> dBool(0, 1);
			if (dBool(grnRand)) {
				input = dBool(grnRand);
				output = dBool(grnRand);
			}
		}
	}

	void reset() {
		c = INIT_CONCENTRATION;
		prevc = c;
	}

	static constexpr double getMaxDistance() {
		return sqrt(std::pow((maxCoord - minCoord), 2) * nbCoords);
	}

	static double relativeDistance(const HiProtein &a, const HiProtein &b) {
		return a.getDistanceWith(b);
	}

	double getDistanceWith(const HiProtein &p) const {
		double sum = 0;
		for (size_t i = 0; i < nbCoords; ++i) {
			sum += std::pow(
			    static_cast<double>(coords.at(i)) - static_cast<double>(p.coords.at(i)), 2);
		}
		return sqrt(sum) / getMaxDistance();
	}
	std::string typeToString(ProteinType t) const {
		switch (t) {
			case ProteinType::input:
				return "input";
			case ProteinType::regul:
				return "regul";
			case ProteinType::output:
				return "output";
		}
		return "unknown_type";
	}
	// switching between integral or real random distribution
	template <typename T = CoordsType>
	typename std::enable_if<!std::is_integral<T>::value, T>::type getRandomCoord() {
		std::uniform_real_distribution<double> distribution(static_cast<double>(minCoord),
		                                                    static_cast<double>(maxCoord));
		return static_cast<CoordsType>(distribution(grnRand));
	}
	template <typename T = CoordsType>
	typename std::enable_if<std::is_integral<T>::value, T>::type getRandomCoord() {
		std::uniform_int_distribution<int> distribution(minCoord, maxCoord);
		return static_cast<CoordsType>(distribution(grnRand));
	}
};
#endif
