#include <array>
#include <cassert>
#include <random>
#include <sstream>
#include "external/gaga/gaga.hpp"

static std::default_random_engine globalRand;

struct MyDNA {
	static const constexpr size_t N = 40;
	std::array<int, N> numbers;

	MyDNA() {}
	MyDNA(const std::string& s) {
		std::stringstream ss(s);
		std::string item;
		int i = 0;
		while (std::getline(ss, item, ' ')) numbers[i++] = stoi(item);
		assert(i == N);
	}

	std::string serialize() const {
		std::stringstream ss;
		for (const auto& n : numbers) ss << n << " ";
		ss.seekp(-1, std::ios_base::end);
		return ss.str();
	}

	void reset() {}
	void mutate() {
		std::uniform_int_distribution<int> d5050(0, 1);
		std::uniform_int_distribution<int> dInt(0, N - 1);
		numbers[dInt(globalRand)] = d5050(globalRand) ? 1 : 0;
	}

	static MyDNA random() {
		MyDNA res;
		std::uniform_int_distribution<int> d5050(0, 1);
		for (size_t i = 0; i < N; ++i) res.numbers[i] = d5050(globalRand);
		return res;
	}

	MyDNA crossover(const MyDNA& other) {
		MyDNA res;
		std::uniform_int_distribution<int> d5050(0, 1);
		for (size_t i = 0; i < N; ++i)
			res.numbers[i] = d5050(globalRand) ? numbers[i] : other.numbers[i];
		return res;
	}
};

int main(int argc, char** argv) {
	GAGA::GA<MyDNA> ga(argc, argv);
	globalRand = std::default_random_engine(0);

	ga.setEvaluator(
	    [](auto& individu, int) {
		    int n = 0;
		    for (int a : individu.dna.numbers) n += a;
		    std::this_thread::sleep_for(std::chrono::milliseconds(2));
		    individu.fitnesses["number of ones"] = n;
	    },
	    "sum");

	ga.setPopSize(200);
	ga.setMutationProba(0.8);
	ga.setCrossoverProba(0.2);
	ga.setVerbosity(0);
	ga.setNbThreads(8);
	ga.setPopSaveInterval(0);
	ga.setGenSaveInterval(0);
	ga.initPopulation([]() { return MyDNA::random(); });
	auto start = std::chrono::system_clock::now();
	ga.step(10);
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;
	std::cout << diff.count() << " s\n";
	return 0;
}
