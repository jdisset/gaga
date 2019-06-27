#include <array>
#include <cassert>
#include <random>
#include <sstream>
#include "../../../gaga/extra/sqlitesave/sqlitesave.hpp"
#include "../../../gaga/gaga.hpp"

static std::default_random_engine globalRand;

struct MyDNA {
	// MyDNA is a simple example of a DNA class
	// it contains an array of N integers,
	// implements a simple mutation
	// and a simple uniform crossover
	// Through initialization & mutation, we ensure that the dna will only contain 0 or 1

	static const constexpr size_t N = 40;
	std::array<int, N> numbers;

	MyDNA() {}

	// deserialization : we just read a line of numbers separated by a space
	MyDNA(const std::string& s) {
		std::stringstream ss(s);
		std::string item;
		int i = 0;
		while (std::getline(ss, item, ' ')) numbers[i++] = stoi(item);
		assert(i == N);
	}

	// serialization : write every numbers on a line, separated by a space
	std::string serialize() const {
		std::stringstream ss;
		for (const auto& n : numbers) ss << n << " ";
		ss.seekp(-1, std::ios_base::end);
		return ss.str();
	}

	// mutation consists in replacing one of the numbers by a random number
	void mutate() {
		std::uniform_int_distribution<int> d5050(0, 1);
		std::uniform_int_distribution<int> dInt(0, N - 1);
		numbers[dInt(globalRand)] = d5050(globalRand) ? 1 : 0;
	}

	// this is a uniform crossover :
	// we randomly decide for each number if we take its value from parent 1 or parent 2
	MyDNA crossover(const MyDNA& other) {
		MyDNA res;
		std::uniform_int_distribution<int> d5050(0, 1);
		for (size_t i = 0; i < N; ++i)
			res.numbers[i] = d5050(globalRand) ? numbers[i] : other.numbers[i];
		return res;
	}

	// this is just a static  method that will generate a new random dna
	// we will use it to initialize the population
	static MyDNA random() {
		MyDNA res;
		std::uniform_int_distribution<int> d5050(0, 1);
		for (size_t i = 0; i < N; ++i) res.numbers[i] = d5050(globalRand);
		return res;
	}

};

int main(int, char**) {
	globalRand = std::default_random_engine(0);

	GAGA::GA<MyDNA> ga;  // declaration of the GAGA instance, with dna type MyDNA

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
	std::string sqlFilename = "onemax.sql";
	SQLiteSaver sql(sqlFilename, "");  // second argument is any configuration detail for
	                                   // the run you want to save in the database export
#endif

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

	return 0;
}
