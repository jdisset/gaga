#include <random>
#include <sstream>
#include "../../../gaga/gaga.hpp"
#include "../../../gaga/novelty.hpp"
#include "mydna.hpp"

#ifdef SQLITE_SAVE
#include "../../../gaga/extra/sqlitesave/sqlitesave.hpp"
#endif

int main(int, char**) {
	globalRand = std::default_random_engine(0);

	using signature_t =
	    std::array<int, MyDNA::N>;  // signature type: the "behavioral caracterization"
	                                // format for each individual.

	using Ind_t = GAGA::NoveltyIndividual<
	    MyDNA,
	    signature_t>;  // we want to use the type of individual that the novelty extension
	                   // provide. These individuals just have 2 extra fields: signature
	                   // (their behavioral caracterization) and archived (which tells if
	                   // they are part of the novelty archive)
	                   //
	using GA_t =
	    GAGA::GA<MyDNA, Ind_t>;  // we need to specify both our dna type and individual type

	GA_t ga;  // declaration of the GAGA instance, with dna type MyDNA

	// -- -- -- -- -- -- SPECIFIC TO THE NOVELTY EXTENSION: -- -- -- -- -- -- --
	GAGA::NoveltyExtension<GA_t> nov;  // novelty extension instance
	// Distance function (compares 2 signatures). Here a simple Euclidian distance.
	auto euclidianDist = [](const auto& fpA, const auto& fpB) {
		double sum = 0;
		for (size_t i = 0; i < fpA.size(); ++i) sum += std::pow(fpA[i] - fpB[i], 2);
		return sqrt(sum);
	};
	nov.setComputeSignatureDistanceFunction(euclidianDist);
	nov.K = 10;  // size of the neighbourhood to compute novelty.
	//(Novelty = avg dist to the K Nearest Neighbors)

	ga.useExtension(nov);  // we have to tell gaga we want to use this extension
	// -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#ifdef SQLITE_SAVE  // OPTIONAL: we set up an sqlite saver
	SQLiteSaver<GA_t> sql("one_novelty.sql");
	// -- -- -- -- -- -- SPECIFIC TO THE NOVELTY EXTENSION: -- -- -- -- -- -- --
	sql.useExtension(nov);  // novelty also adds columns to some sql tables.
	// -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
	sql.newRun("optional string describing conf");
#endif

	ga.setEvaluator(
	    [](auto& individu, int) {  // second argument of the evaluator funciton is the cpuId
		    // -- -- -- -- -- -- SPECIFIC TO THE NOVELTY EXTENSION: -- -- -- -- -- -- --
		    individu.signature = individu.dna.numbers;
			individu.infos = nlohmann::json(individu.dna.numbers).dump();
		    assert(individu.signature == individu.dna.numbers);
		    // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
	    },
	    "sum");  // name of the evaluator, just used for saving purposes

	// the rest is similar to the non-novelty version...
	// setting a few basic parameters.
	// see documentation for comprehensive list
	ga.setPopSize(200);
	ga.setMutationRate(0.8);
	ga.setCrossoverRate(0.2);
	ga.setVerbosity(1);
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

	for (auto& i : ga.previousGenerations.back()) {
		assert(i.signature == i.dna.numbers);
	}

	return 0;
}
