#define GAGA_TESTING
#include <thread>
#include "../dna/arraydna.hpp"
#include "../extra/gagazmq/gagazmq.hpp"
#include "../gaga.hpp"
#include "../novelty.hpp"
#include "catch/catch.hpp"
#include "dna.hpp"

TEST_CASE("ZMQ") {
	using dna_t = GAGA::ArrayDNA<int, 100>;
	using Ind_t = GAGA::NoveltyIndividual<dna_t>;
	using GA_t = GAGA::GA<dna_t, Ind_t>;

	const bool COMPRESSION = true;

	auto onemax = [](auto& i) {
		// i.fitnesses["sum"] = std::accumulate(i.dna.values.begin(), i.dna.values.end(), 0);
		i.signature = std::vector<double>(i.dna.values.data(),
		                                  i.dna.values.data() + i.dna.values.size());
	};

	auto euclidian = [](const auto& fpA, const auto& fpB) {
		double sum = 0;
		for (size_t i = 0; i < fpA.size(); ++i) sum += std::pow(fpA[i] - fpB[i], 2);
		return sqrt(sum);
	};

	auto createWorker = [=](std::string addr) {
		GAGA::ZMQWorker<GA_t, Ind_t::sig_t> w(addr);
		w.evaluate = onemax;
		w.evalBatchSize = 1;
		w.distanceBatchSize = 10000;
		w.debug = false;
		w.setCompression(COMPRESSION);
		w.start(euclidian);
	};

	std::string port = "tcp://*:4321";
	std::string serverAddr = "tcp://localhost:4321";

	std::vector<std::thread> workers;
	const int NWORKERS = 5;
	for (int i = 0; i < NWORKERS; ++i) workers.emplace_back(createWorker, serverAddr);

	GAGA::ZMQServer<GA_t> server;
	server.setCompression(COMPRESSION);
	auto& ga = server.ga;
	GAGA::NoveltyExtension<GA_t> nov;
	ga.useExtension(nov);
	nov.enableDistributed(server);
	ga.setPopSize(50);
	ga.setVerbosity(0);
	ga.initPopulation([&]() { return dna_t::random(); });

	server.bind(port);
	for (size_t i = 0; i < 50; ++i) ga.step();
	server.terminate();

	for (auto& w : workers) w.join();
}

TEST_CASE("Novelty") {
	using dna_t = GAGA::ArrayDNA<int, 100>;
	using Ind_t = GAGA::NoveltyIndividual<dna_t>;
	using GA_t = GAGA::GA<dna_t, Ind_t>;

	GAGA::NoveltyExtension<GA_t> nov;

	GA_t ga;
	ga.useExtension(nov);
	ga.setVerbosity(0);
	ga.setEvaluator([](auto& i, int) {
		i.signature = std::vector<double>(i.dna.values.data(),
		                                  i.dna.values.data() + i.dna.values.size());
	});
	ga.setCrossoverMethod([](const auto& a, const auto&) { return a; });
	ga.initPopulation([&]() { return dna_t::random(); });

	nov.setComputeSignatureDistanceFunction([](const auto& fpA, const auto& fpB) {
		double sum = 0;
		for (size_t i = 0; i < fpA.size(); ++i) sum += std::pow(fpA[i] - fpB[i], 2);
		return sqrt(sum);
	});

	REQUIRE(nov.archive.size() == 0);
	ga.step(3);
	REQUIRE(nov.archive.size() > 0);

	// test findKNN
	std::vector<std::vector<double>> dmat{{0, 1, 8, 9, 1},
	                                      {1, 0, 3, 2, 7},
	                                      {8, 3, 0, 4, 9},
	                                      {9, 2, 4, 0, 1},
	                                      {1, 7, 9, 1, 0}};

	{
		auto knn = nov.findKNN(3, 1, dmat);
		std::unordered_set<size_t> knn_set(knn.begin(), knn.end());
		std::unordered_set<size_t> shouldBe{4};
		REQUIRE(knn_set == shouldBe);
	}
	{
		auto knn = nov.findKNN(0, 3, dmat);
		std::unordered_set<size_t> knn_set(knn.begin(), knn.end());
		std::unordered_set<size_t> shouldBe{1, 2, 4};
		REQUIRE(knn_set == shouldBe);
	}
	{
		auto knn = nov.findKNN(4, 10000, dmat);
		std::unordered_set<size_t> knn_set(knn.begin(), knn.end());
		std::unordered_set<size_t> shouldBe{0, 1, 2, 3};
		REQUIRE(knn_set == shouldBe);
	}

	// tests for the computeDistanceMatrix method
	std::vector<std::vector<double>> distanceMatrix =
	    nov.defaultComputeDistanceMatrix(nov.archive);
	REQUIRE(distanceMatrix.size() == nov.archive.size());
	REQUIRE(distanceMatrix.size() == distanceMatrix[0].size());
	for (size_t i = 0; i < nov.archive.size(); ++i)
		for (size_t j = 0; j < nov.archive.size(); ++j)
			REQUIRE(distanceMatrix[i][j] ==
			        distanceMatrix[j][i]);  // /!\ dist(a,b) == dist(b,a) only for true metrics!
			                                // This test should not be applied with
			                                // simmilarities that aren't true metrics.
}
