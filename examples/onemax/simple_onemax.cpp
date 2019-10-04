#include <cassert>
#include <random>
#include <sstream>
#include "../../../gaga/gaga.hpp"
#include "mydna.hpp"

#ifdef SQLITE_SAVE
#include "../../../gaga/extra/sqlitesave/sqlitesave.hpp"
#endif


int main(int, char**) {
	globalRand = std::default_random_engine(0);

	using GA_t = GAGA::GA<MyDNA>;
	GA_t ga;  // declaration of the GAGA instance, with dna type MyDNA

	ga.setEvaluator(
	    [](auto& individu, int) {  // second argument of the evaluator funciton is the cpuId
		    int n = 0;
		    for (int a : individu.dna.numbers) n += a;
		    std::this_thread::sleep_for(std::chrono::milliseconds(1));  // we simulate load
		    individu.fitnesses["number of ones"] = n;                   // only one objective
	    },
	    "sum");  // name of the evaluator, just used for saving purposes

#ifdef SQLITE_SAVE
	// OPTIONAL: we set up an sqlite saver
	SQLiteSaver<GA_t> sql("onemax.sql");
	sql.newRun("optional string describing conf of this run");
#endif

	// setting a few basic parameters.
	// see documentation for comprehensive list
	ga.setPopSize(200);
	ga.setMutationRate(0.8);
	ga.setCrossoverRate(0.2);
	ga.setVerbosity(2);
	ga.setNbThreads(8);

	// we initialize the population with random DNA. The function passed to
	// initPopulation is called enough time to fill the population vector
	ga.initPopulation([]() { return MyDNA::random(); });

	for (size_t i = 0; i < 10; ++i) {  // we run the ga for 10 generations
		ga.step();                       // next generation

#ifdef SQLITE_SAVE
		sql.newGen(ga);  // saving the generation to sql
#endif
	}

	return 0;
}
