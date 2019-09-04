#include "../dna/arraydna.hpp"
#include "../extra/sqlitesave/sqlitesave.hpp"
#include "../gaga.hpp"
#include "../novelty.hpp"
#include "catch/catch.hpp"
#include "dna.hpp"

TEST_CASE("SQL") {
	// types that will be used in this test
	using dna_t = GAGA::ArrayDNA<int, 256>;  // our DNA: 256 integers.
	using Ind_t = GAGA::Individual<dna_t>;   // Individual type is the default gaga one
	using GA_t = GAGA::GA<dna_t, Ind_t>;  // and finally the GA type (needs dna_t + ind_t)

	// SQL saver setup
	SQLiteSaver<GA_t> sql(":memory:");  //:memory: is a keyword that triggers in memory DB
	sql.newRun();                       // ready to start saving a new evo run

	std::string query =
	    "SELECT name FROM sqlite_master WHERE type ='table' AND name NOT LIKE 'sqlite_%';";

	std::vector<std::string> tables;
	sql.exec(query, [&](int argc, char** argv, char**) {
		for (int i = 0; i < argc; i++) tables.push_back(argv[i]);
	});
	REQUIRE(tables.size() == 6);

	// gaga setup
	GA_t ga;
	const int POPSIZE = 30;
	ga.setPopSize(POPSIZE);
	ga.setVerbosity(0);
	ga.initPopulation([&]() { return dna_t::random(); });
	int evaluationNumber = 0;
	ga.setEvaluator([&](auto& i, int) {
		// here we use only one fitness: = maximize the sum of all integers in the dna
		i.fitnesses["sum"] = std::accumulate(i.dna.values.begin(), i.dna.values.end(), 0);
		i.infos = std::to_string(evaluationNumber++);
	});

	// go!
	for (size_t i = 0; i < 10; ++i) {
		ga.step();
		sql.newGen(ga);  // saving the generation to sql
		sql.exec("SELECT count(*) FROM individual;", [&](int, char** argv, char**) {
			REQUIRE(std::atoi(argv[0]) == POPSIZE * (i + 1));
		});

		size_t ind_id = 0;
		auto& prevPop = ga.previousGenerations.back();
		query = "SELECT id,dna,infos FROM individual WHERE id_generation = " +
		        std::to_string(i + 1) + " ORDER BY id ASC;";
		sql.exec(query, [&](int, char** argv, char** col) {
			REQUIRE(prevPop[ind_id].dna.serialize() == std::string(argv[1]));
			REQUIRE(prevPop[ind_id].infos == std::string(argv[2]));
			++ind_id;
		});
	}
}

TEST_CASE("SQL + Novelty") {
	// types that will be used in this test
	using dna_t = GAGA::ArrayDNA<int, 256>;  // our DNA: 256 integers.
	using sig_t = decltype(dna_t::values);   // signature type is same as dna.values
	using Ind_t = GAGA::NoveltyIndividual<dna_t, sig_t>;  // Individual type
	using GA_t = GAGA::GA<dna_t, Ind_t>;  // and finally the GA type (needs dna_t + ind_t)

	// gaga setup
	GA_t ga;
	// novelty extension
	GAGA::NoveltyExtension<GA_t> nov;
	ga.useExtension(nov);

	// SQL saver setup
	SQLiteSaver<GA_t> sql(":memory:");  //:memory: keyword that triggers in-memory DB
	sql.useExtension(nov);
	sql.newRun();  // ready to start saving a new evo run

	const int POPSIZE = 30;
	ga.setPopSize(POPSIZE);
	ga.setVerbosity(0);
	ga.initPopulation([&]() { return dna_t::random(); });
	int evaluationNumber = 0;
	ga.setEvaluator([&](auto& i, int) {
		// here we use only one fitness: = maximize the sum of all integers in the dna
		i.fitnesses["sum"] = std::accumulate(i.dna.values.begin(), i.dna.values.end(), 0);
		i.signature = i.dna.values;
		i.infos = std::to_string(evaluationNumber++);
	});
	nov.setComputeSignatureDistanceFunction([](const auto& fpA, const auto& fpB) {
		double sum = 0;
		for (size_t i = 0; i < fpA.size(); ++i) sum += std::pow(fpA[i] - fpB[i], 2);
		return sqrt(sum);
	});

	// go!
	for (size_t i = 0; i < 10; ++i) {
		ga.step();
		sql.newGen(ga);  // saving the generation to sql
		sql.exec("SELECT count(*) FROM individual;", [&](int, char** argv, char**) {
			REQUIRE(std::atoi(argv[0]) == POPSIZE * (i + 1));
		});

		size_t ind_id = 0;
		auto& prevPop = ga.previousGenerations.back();
		std::string query =
		    "SELECT id,dna,infos,signature,original_id FROM individual WHERE id_generation "
		    "= " +
		    std::to_string(i + 1) + " ORDER BY id ASC;";
		sql.exec(query, [&](int, char** argv, char**) {
			REQUIRE(prevPop[ind_id].dna.serialize() == std::string(argv[1]));
			REQUIRE(prevPop[ind_id].infos == std::string(argv[2]));
			nlohmann::json jsSig = prevPop[ind_id].signature;
			REQUIRE(jsSig.dump() == std::string(argv[3]));
			REQUIRE(jsSig == nlohmann::json(prevPop[ind_id].dna.values));
			REQUIRE(ind_id == std::atoi(argv[4]));
			++ind_id;
		});

		////// archive tests
		{
			query =
			    "SELECT id_individual, individual.id_generation, individual.original_id FROM "
			    "archive_content, "
			    "individual WHERE "
			    "id_individual = individual.id AND archive_content.id_generation = " +
			    std::to_string(i + 1) + ";";

			int c = 0;
			sql.exec(query, [&](int, char** argv, char**) {
				REQUIRE(nov.isArchived(std::pair<size_t, size_t>(
				            std::atoi(argv[1]) - 1, (size_t)std::atoi(argv[2]))) == true);
				c++;
			});
			REQUIRE(c == nov.archive.size());
		}
	}
}
