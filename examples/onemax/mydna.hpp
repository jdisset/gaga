#include <array>
#include <random>
#include <sstream>
#include <string>

static std::default_random_engine globalRand;
struct MyDNA {
	// MyDNA is a simple example of a DNA class
	// it contains an array of N integers,
	// implements a simple mutation
	// and a simple uniform crossover
	// Through initialization & mutation, we ensure that the dna will only contain 0 or 1

	static const constexpr size_t N = 100;
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
