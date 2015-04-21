#include "../../gaga/jsonxx/jsonxx.h"
#include "../../gaga/tools.h"
#include <random>

using namespace std;

/***********************************
 *        A VERY BASIC DNA
 **********************************/
class IntDNA {
 private:
	uniform_int_distribution<int> distribution = uniform_int_distribution<int>(0, 1000000);
	int value1 = 0;
	int value2 = 0;

 public:
	IntDNA(int v1, int v2) : value1(v1), value2(v2) {}

	/********************************************************
	 *                MANDATORY METHODS
	 *******************************************************/

	// A valid dna must be able to be constructed from a json object
	IntDNA(const jsonxx::Object& o) {
		value1 = o.get<jsonxx::Number>("value1");
		value2 = o.get<jsonxx::Number>("value2");
	}

	// It must have a static "random" constructor which will be used to create the first generation
	static IntDNA random() {
		uniform_int_distribution<int> dist = uniform_int_distribution<int>(0, 1000000);
		return IntDNA(dist(GAGA::globalRand), dist(GAGA::globalRand));
	}

	// It must also have a mutate method
	void mutate() {
		value1 = distribution(GAGA::globalRand);
		value2 = distribution(GAGA::globalRand);
	}

	// A crossover method
	IntDNA crossover(const IntDNA& other) {
		uniform_int_distribution<int> dist = uniform_int_distribution<int>(0, 1);
		int r = dist(GAGA::globalRand);
		if (r == 0) return IntDNA(other.value1, value2);
		return IntDNA(value1, other.value2);
	}

	// A reset method (just to cleanup things before a new evaluation)
	void reset() {}

	// And a method that returns a json object
	jsonxx::Object toJSON() const {
		jsonxx::Object o;
		o << "value1" << value1;
		o << "value2" << value2;
		return o;
	}
	/*******************************************************/

	int getValue1() { return value1; }
	int getValue2() { return value2; }
};
