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

#ifndef INDIVIDUAL_HPP
#define INDIVIDUAL_HPP

#include "tools.h"
#include "jsonxx/jsonxx.h"

using namespace std;

/*********************************
 * Individual representation
 * ******************************/
// DNA class must have :
// mutate()
// crossover(DNA& other)
// jsonxx:Object& constructor
// toJson()

namespace GAGA {
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
}
#endif
