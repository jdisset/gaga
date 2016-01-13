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
#include <random>
#include "../../gaga/json/json.hpp"

/***********************************
 *        A VERY BASIC DNA
 **********************************/
class IntDNA {
	using json = nlohmann::json;

 private:
	std::uniform_int_distribution<int> distribution =
	    std::uniform_int_distribution<int>(0, 1000000);
	int value1 = 0;
	int value2 = 0;

 public:
	IntDNA(int v1, int v2) : value1(v1), value2(v2) {}

	/********************************************************
	 *                MANDATORY METHODS
	 *******************************************************/

	// A valid dna must be able to be constructed from a json object
	explicit IntDNA(const json &o) {
		value1 = o["value1"];
		value2 = o["value2"];
	}

	// It must have a static "random" constructor which will be used to create the first
	// generation
	static IntDNA random(int, char **) {
		std::uniform_int_distribution<int> dist =
		    std::uniform_int_distribution<int>(0, 1000000);
		return IntDNA(dist(globalRand), dist(globalRand));
	}

	// It must also have a mutate method
	void mutate() {
		value1 = distribution(globalRand);
		value2 = distribution(globalRand);
	}

	// A crossover method
	IntDNA crossover(const IntDNA &other) {
		std::uniform_int_distribution<int> dist = std::uniform_int_distribution<int>(0, 1);
		int r = dist(globalRand);
		if (r == 0) return IntDNA(other.value1, value2);
		return IntDNA(value1, other.value2);
	}

	// A reset method (just to cleanup things before a new evaluation)
	void reset() {}

	// And a method that returns a json string
	std::string toJSON() const {
		json o;
		o["value1"] = value1;
		o["value2"] = value2;
		return o.dump(2);
	}
	/*******************************************************/

	int getValue1() { return value1; }
	int getValue2() { return value2; }
	static std::default_random_engine globalRand;
};
