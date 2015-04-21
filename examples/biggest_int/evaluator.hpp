#include <vector>
using namespace std;

template <class Individu> struct BiggestIntEvaluator {
	const string name = "BiggestIntEvaluator";
	void operator()(Individu& ind) { 
		ind.fitnesses["value1"] = ind.dna.getValue1();
		ind.fitnesses["value2"] = ind.dna.getValue2();
		ind.fitnesses["sum"] = ind.dna.getValue1() + ind.dna.getValue2();
	}
};
