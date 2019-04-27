#include <dna/arraydna.hpp>
#include <extra/gagazmq/gagazmq.hpp>
#include <gaga.hpp>
#include <iostream>

int main(int, char**) {
	using dna_t = GAGA::ArrayDNA<int, 100>;
	using GA_t = GAGA::GA<dna_t>;

	const bool COMPRESSION = false;
	const std::string port = "tcp://*:4321";

	GAGA::ZMQServer<GA_t> server;
	server.setCompression(COMPRESSION);
	server.enableDistributedDistanceMatrixComputation();

	auto& ga = server.ga;
	ga.setPopSize(50);
	ga.setVerbosity(2);
	ga.enableNovelty();
	ga.initPopulation([&]() { return dna_t::random(); });

	server.bind(port);
	for (size_t i = 0; i < 50; ++i) ga.step();
	server.terminate();

	return 0;
}
