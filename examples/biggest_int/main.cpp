#include "../../gaga/gaga.hpp"
#include "evaluator.hpp"
#include "dna.hpp"

std::default_random_engine IntDNA::globalRand =
    std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count());
using namespace GAGA;
int main(int argc, char **argv) {
	int tablo[5];
	for (int i = 0; i < 600; ++i) {
		cout << tablo[i] << endl;
	}
	GA<IntDNA, BiggestIntEvaluator<Individual<IntDNA>>> evo(argc, argv);
	return evo.start();
}
