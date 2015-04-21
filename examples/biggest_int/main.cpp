#include "../../gaga/gaga.h"
#include "evaluator.hpp"
#include "dna.hpp"

using namespace GAGA;
int main(int argc, char** argv) {
	GA<IntDNA, BiggestIntEvaluator<Individual<IntDNA>>> evo(argc, argv);
	return evo.start();
}
