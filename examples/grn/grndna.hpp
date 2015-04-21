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

#ifndef GRNDNA_HPP
#define GRNDNA_HPP
#include "grn.hpp"
#include <fstream>
using namespace std;



// ESSENTIELLEMENT UN WRAPPER DE GRN
// le truc le plus important est la fonciton random, c'est comme ça que seront initialisé tes grns
class GRNDNA {
	GRN grn;

 public:
	GRNDNA(){}
	GRNDNA(GRN g) : grn(g) {}

	GRNDNA crossover(const GRNDNA& other) {
		GRN g = grn.crossover(other.grn);
		GRNDNA res(g);
		return res;
	}
	void mutate() { grn.mutate(); }
	void reset() { grn.reset(); }
	// RANDOM INITIALIZATION (first generation, usually)
	static GRNDNA random() {
		GRN g;
		// adding inputs and outputs
		// ..

		// then adding a few reguls
		g.randomReguls(3);
		GRNDNA res(g);

		return res;
	}

	jsonxx::Object toJSON() const { return grn.toJSON(); }
	void saveToFile(string filePath) const {
		ofstream fs(filePath);
		if (!fs) {
			cerr << "Cannot open the output file." << endl;
		}
		fs << grn.toJSON().json();
		fs.close();
	}
};
#endif
