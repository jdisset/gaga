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

#ifndef EVALUATOR_HPP
#define EVALUATOR_HPP
#include <vector>
#include "tools.h"
using namespace std;

/*********************************************************************************
 *                                 EVALUATORS
 ********************************************************************************/
// they receive an individual and must handle its setup and update routine and finally give a fitness
// don't forget to give them a name !

template <class Individu> struct MonEvaluateur {
	const string name = "monEvaluateur";

	void operator()(Individu& ind) {
		const unsigned int maxStep = 6500;
		//  NOVELTY RELATED: when we need to take footprints:
		const vector<double> footprintSnaps = {0.1, 0.2, 0.4, 0.6, 0.8};
		// update :
		for (unsigned int step = 0; step < maxStep; ++step) {
			// do stuff with our Individual
			// ...
			// NOVELTY RELATED: take snapshot and complete individual's footprint
			for (unsigned int i = 0; i < footprintSnaps.size(); ++i) {
				// if we are at the right step we take a snap
				if (step == footprintSnaps[i] * maxStep && ind.footprint.size() < i + 1) {
					// ind.footprint.push_back(ind.computeFootprint());
				}
			}
		}
		uniform_real_distribution<double> d(0.0, 1.0);
		// then we set the fitnesses
		ind.fitnesses["monObjectif"] = 42;
		ind.fitnesses["monAutreObjectif"] = d(globalRand);
	}

};


template <class Individu> struct MonAutreEvaluateur {
	const string name = "monAutreEvaluateur";

	void operator()(Individu& ind) {
		for (unsigned int step = 0; step < 100; ++step) {
			// do stuff with our Individual
		}
		// then we set the fitnesses
		ind.fitnesses["monObjectifUnique"] = 42;
	}
};
#endif
