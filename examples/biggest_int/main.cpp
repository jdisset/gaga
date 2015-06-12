#include "../../gaga/gaga.hpp"
#include "evaluator.hpp"
#include "dna.hpp"

std::default_random_engine IntDNA::globalRand =
    std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count());
using namespace GAGA;
int main(int argc, char **argv) {
	GA<IntDNA, BiggestIntEvaluator<Individual<IntDNA>>> evo(argc, argv);
	return evo.start();
}
