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

#include "ga.hpp"
#include "grndna.hpp"
#include "evaluators.hpp"
using namespace std;

int main(int argc, char **argv) {
#ifdef CLUSTER
	MPI_Init(&argc, &argv);
#endif
	cout << GREEN << " Starting GA " << endl;
	GA<GRNDNA, MonEvaluateur<Individual<GRNDNA>>> evo;
	evo.start();
	cout << endl << GREEN << "This is the end." << endl;
#ifdef CLUSTER
	MPI_Finalize();
#endif
	return 0;
}
