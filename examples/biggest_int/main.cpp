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
#include "../../gaga/gaga.hpp"
#include "evaluator.hpp"
#include "dna.hpp"

std::default_random_engine IntDNA::globalRand = std::default_random_engine(
    std::chrono::system_clock::now().time_since_epoch().count());
using namespace GAGA;
int main(int argc, char **argv) {
	GA<IntDNA, BiggestIntEvaluator<Individual<IntDNA>>> evo(argc, argv);
	evo.setVerbosity(1);
	evo.setNbGenerations(400);
	return evo.start();
}
