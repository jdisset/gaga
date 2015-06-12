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
#include <string>

template <class Individu> struct BiggestIntEvaluator {
	const std::string name = "BiggestIntEvaluator";

	BiggestIntEvaluator(int, char **) {}
	void operator()(const Individu &ind) {
		ind.fitnesses["value1"] = ind.dna.getValue1();
		ind.fitnesses["value2"] = ind.dna.getValue2();
		ind.fitnesses["sum"] = ind.dna.getValue1() + ind.dna.getValue2();
	}
};
