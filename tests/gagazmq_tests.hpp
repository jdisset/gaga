#pragma once
#define GAGA_TESTING
#include <thread>
#include "../dna/arraydna.hpp"
#include "../extra/gagazmq/gagazmq.hpp"
#include "../gaga.hpp"
#include "../novelty.hpp"
#include "catch/catch.hpp"
#include "dna.hpp"
TEST_CASE("Novelty") {
	using dna_t = GAGA::ArrayDNA<int, 100>;
	using sig_t = decltype(dna_t::values);  // signature type is an array of int
	using Ind_t = GAGA::NoveltyIndividual<dna_t, sig_t>;
	using GA_t = GAGA::GA<dna_t, Ind_t>;
	using dMat_t = std::vector<std::vector<double>>;

	const size_t NGENERATIONS = 20;

	GA_t ga;
	ga.setPopSize(100);
	ga.setVerbosity(0);
	auto onemax = [](auto& i) {
		i.fitnesses["sum"] = std::accumulate(i.dna.values.begin(), i.dna.values.end(), 0);
		i.signature = i.dna.values;
	};
	ga.setEvaluator([&](auto& i, int) { onemax(i); });

	ga.setCrossoverMethod([](const auto& a, const auto&) { return a; });
	ga.initPopulation([&]() { return dna_t::random(); });

	auto firstPop = ga.population;
	GAGA::NoveltyExtension<GA_t> nov;
	const size_t MAX_ARCHIVE_SIZE = 30;
	nov.maxArchiveSize = MAX_ARCHIVE_SIZE;
	nov.nbOfArchiveAdditionsPerGeneration = 14;
	ga.useExtension(nov);

	auto euclidian = [](const auto& fpA, const auto& fpB) {
		double sum = 0;
		for (size_t i = 0; i < fpA.size(); ++i) sum += std::pow(fpA[i] - fpB[i], 2);
		return sqrt(sum);
	};

	nov.setComputeSignatureDistanceFunction(euclidian);
	SECTION("novelty archive & distance Matrix are ok") {
		REQUIRE(nov.archive.size() == 0);
		for (size_t g = 0; g < NGENERATIONS; ++g) {
			ga.step();
			REQUIRE(nov.archive.size() > 0);
			REQUIRE(nov.archive.size() <= nov.maxArchiveSize);
			if (g > 2) REQUIRE(nov.archive.size() == MAX_ARCHIVE_SIZE);
			auto distMat = nov.defaultComputeDistanceMatrix(nov.archive);
			REQUIRE(distMat.size() == nov.archive.size());
			REQUIRE(distMat.size() == distMat[0].size());
			for (size_t i = 0; i < distMat.size(); ++i) {
				for (size_t j = 0; j < distMat.size(); ++j) {
					if (i == j)
						REQUIRE(distMat[i][j] == 0);
					else
						REQUIRE(distMat[i][j] ==
						        euclidian(nov.archive[i].signature, nov.archive[j].signature));
				}
			}
		}

		SECTION("distance Matrix is symmetric") {
			dMat_t distanceMatrix = nov.defaultComputeDistanceMatrix(nov.archive);
			REQUIRE(distanceMatrix.size() == nov.archive.size());
			REQUIRE(distanceMatrix.size() == distanceMatrix[0].size());
			for (size_t i = 0; i < nov.archive.size(); ++i)
				for (size_t j = 0; j < nov.archive.size(); ++j)
					REQUIRE(distanceMatrix[i][j] == distanceMatrix[j][i]);
		}
	}

	SECTION("WITH ZMQ") {
		const bool COMPRESSION = true;
		// worker creation
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

		GAGA::ZMQServer<GA_t> server(ga);
		server.setCompression(COMPRESSION);
		server.bind(port);

		nov.enableDistributed(server);

		SECTION("distributed distance matrix is the same as non distributed") {
			size_t prevArchiveSize = 0;
			for (size_t g = 0; g < NGENERATIONS; ++g) {
				std::cerr << " ---------------- GEN " << g << std::endl;
				ga.step();

				auto distMat_distributed =
				    nov.distributedComputeDistanceMatrix(nov.archive, server);
				auto distMat_nonDistributed = nov.defaultComputeDistanceMatrix(nov.archive);

				REQUIRE(distMat_nonDistributed.size() == nov.archive.size());

				for (size_t i = 0; i < distMat_nonDistributed.size(); ++i) {
					for (size_t j = 0; j < distMat_nonDistributed.size(); ++j) {
						REQUIRE(distMat_nonDistributed[i].size() == distMat_distributed[i].size());
						REQUIRE(distMat_nonDistributed[i][j] == distMat_distributed[i][j]);
						if (i == j)
							REQUIRE(distMat_nonDistributed[i][j] == 0);
						else {
							REQUIRE(distMat_nonDistributed[i][j] ==
							        euclidian(nov.archive[i].signature, nov.archive[j].signature));
						}
					}
				}
				REQUIRE((nov.archive.size() > prevArchiveSize ||
				         nov.archive.size() == MAX_ARCHIVE_SIZE));
				prevArchiveSize = nov.archive.size();
			}
		}

		SECTION("Distributed computations reproduce all non distributed results") {
			// we run distributed first
			std::vector<decltype(nov.archive)> archHistory;
			for (size_t g = 0; g < NGENERATIONS; ++g) {
				archHistory.push_back(nov.archive);
				ga.step();
			}

			// then disable distribution
			server.disable();
			nov.disableDistributed();
			nov.clear();

			// backup of the old run
			auto prevGens = ga.previousGenerations;

			ga.previousGenerations.clear();
			ga.population.clear();

			// starting
			std::vector<Ind_t> nextPop;
			for (size_t g = 0; g < NGENERATIONS; ++g) {
				nextPop = prevGens[g];
				// we clear the interesting parts
				for (auto& i : nextPop) {
					i.fitnesses.clear();
					i.evaluated = false;
					for (auto& s : i.signature) s = 0;
				}
				ga.setPopulation(nextPop);  // we seed with same initial pop
				nov.archive = archHistory[g];
				ga.step();
				REQUIRE(ga.previousGenerations[g].size() == prevGens[g].size());
				for (size_t i = 0; i < prevGens[g].size(); ++i) {
					const auto& a = ga.previousGenerations[g][i];
					const auto& b = prevGens[g][i];
					REQUIRE(a.fitnesses == b.fitnesses);
					REQUIRE(a.signature == b.signature);
					REQUIRE(a.evaluated == true);
					REQUIRE(a.signature == a.dna.values);
				}
			}
		}

		server.terminate();
		for (auto& w : workers) w.join();
	}
}

TEST_CASE("Novelty Internals") {
	using dna_t = GAGA::ArrayDNA<int, 100>;
	using sig_t = decltype(dna_t::values);  // signature type is an array of int
	using Ind_t = GAGA::NoveltyIndividual<dna_t, sig_t>;
	using GA_t = GAGA::GA<dna_t, Ind_t>;
	using dMat_t = std::vector<std::vector<double>>;

	GAGA::NoveltyExtension<GA_t> nov;
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
}
