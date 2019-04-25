#define GAGA_TESTING
#include <thread>
#include "../dna/arraydna.hpp"
#include "../extra/gagazmq/gagazmq.hpp"
#include "../gaga.hpp"
#include "catch/catch.hpp"
#include "dna.hpp"

TEST_CASE("ZMQ") {
	using dna_t = GAGA::ArrayDNA<int, 100>;
	using GA_t = GAGA::GA<dna_t>;

	const bool COMPRESSION = true;

	auto onemax = [](auto& ind) {
		ind.fitnesses["sum"] =
		    std::accumulate(ind.dna.values.begin(), ind.dna.values.end(), 0);
	};

	auto createWorker = [=](std::string addr) {
		GAGA::ZMQWorker<GA_t> w(addr);
		w.evaluate = onemax;
		w.batchSize = 10;
		w.setCompression(COMPRESSION);
		w.start();
	};

	std::string port = "tcp://*:4321";
	std::string serverAddr = "tcp://localhost:4321";

	std::vector<std::thread> workers;

	const int NWORKERS = 5;

	for (int i = 0; i < NWORKERS; ++i) {
		workers.emplace_back(createWorker, serverAddr);
	}

	GAGA::ZMQServer<GA_t> server;
	server.setCompression(COMPRESSION);
	auto& ga = server.ga;
	ga.setPopSize(50);
	ga.setVerbosity(0);
	server.bind(port);

	ga.initPopulation([&]() { return dna_t::random(); });
	for (size_t i = 0; i < 50; ++i) ga.step();

	server.terminate();
	for (auto& w : workers) w.join();
}

TEST_CASE("Novelty") {
	using dna_t = GAGA::ArrayDNA<int, 100>;
	using GA_t = GAGA::GA<dna_t>;

	GA_t ga;
	ga.setVerbosity(3);
	ga.enableNovelty();
	ga.setEvaluator([](auto& i, int) {
		i.footprint = std::vector<double>(i.dna.values.data(),
		                                  i.dna.values.data() + i.dna.values.size());
	});
	ga.setCrossoverMethod([](const auto& a, const auto&) { return a; });
	ga.initPopulation([&]() { return dna_t::random(); });

	ga.setComputeFootprintDistanceFunction([](const auto& fpA, const auto& fpB) {
		double sum = 0;
		for (size_t i = 0; i < fpA.size(); ++i) sum += std::pow(fpA[i] - fpB[i], 2);
		return sqrt(sum);
	});

	REQUIRE(ga.archive.size() == 0);
	ga.step(3);
	REQUIRE(ga.archive.size() > 0);

	// test findKNN
	std::vector<std::vector<double>> dmat{{0, 1, 8, 9, 1},
	                                      {1, 0, 3, 2, 7},
	                                      {8, 3, 0, 4, 9},
	                                      {9, 2, 4, 0, 1},
	                                      {1, 7, 9, 1, 0}};

	{
		auto knn = ga.findKNN(3, 1, dmat);
		std::unordered_set<size_t> knn_set(knn.begin(), knn.end());
		std::unordered_set<size_t> shouldBe{4};
		REQUIRE(knn_set == shouldBe);
	}
	{
		auto knn = ga.findKNN(0, 3, dmat);
		std::unordered_set<size_t> knn_set(knn.begin(), knn.end());
		std::unordered_set<size_t> shouldBe{1, 2, 4};
		REQUIRE(knn_set == shouldBe);
	}
	{
		auto knn = ga.findKNN(4, 10000, dmat);
		std::unordered_set<size_t> knn_set(knn.begin(), knn.end());
		std::unordered_set<size_t> shouldBe{0, 1, 2, 3};
		REQUIRE(knn_set == shouldBe);
	}

	// tests for the computeDistanceMatrix method
	std::vector<std::vector<double>> distanceMatrix =
	    ga.defaultComputeDistanceMatrix(ga.archive);
	REQUIRE(distanceMatrix.size() == ga.archive.size());
	REQUIRE(distanceMatrix.size() == distanceMatrix[0].size());
	for (size_t i = 0; i < ga.archive.size(); ++i)
		for (size_t j = 0; j < ga.archive.size(); ++j)
			REQUIRE(distanceMatrix[i][j] ==
			        distanceMatrix[j][i]);  // /!\ dist(a,b) == dist(b,a) only for true metrics!
			                                // This test should not be applied with
			                                // simmilarities that aren't true metrics.
}
