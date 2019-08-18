#pragma once
#include <vector>
#include "gaga.hpp"

namespace GAGA {

// Just to be clear on the terms:
// signature = signature = phenotype/behavior/feature characterization
//

//
template <typename DNA, typename sig> struct NoveltyIndividual : public Individual<DNA> {
	using sig_t = sig;
	using base = Individual<DNA>;
	sig_t signature;  // individual's signature for novelty computation

	NoveltyIndividual(const DNA &d) : base(d) {}
	NoveltyIndividual(const json &o) : base(o) {
		if (o.count("signature")) signature = o.at("signature").get<sig_t>();
	}

	json toJSON() const {
		auto o = base::toJSON();
		o["signature"] = signature;
		return o;
	}
};

template <typename GA> struct NoveltyExtension {
	using Ind_t = typename GA::Ind_t;
	using Iptr = typename GA::Iptr;
	using sig_t = typename Ind_t::sig_t;
	using distanceMatrix_t = std::vector<std::vector<double>>;

	bool nslc = false;               // enable noverlty search with local competition
	std::vector<Ind_t> archive;      // novelty archive, Individuals are stored there.
	size_t K = 5;                    // size of the neighbourhood for novelty
	bool saveArchiveEnabled = true;  // save the novelty archive
	size_t nbOfArchiveAdditionsPerGeneration = 5;

	void clear() { archive.clear(); }

	template <typename F> void setComputeSignatureDistanceFunction(F &&f) {
		computeSignatureDistance = std::forward<F>(f);
	}
	template <typename F> void setComputDistanceMatrixFunction(F &&f) {
		computeDistanceMatrix = std::forward<F>(f);
	}

	std::function<double(const sig_t &, const sig_t &)> computeSignatureDistance =
	    [](const auto &, const auto &) { return 0; };

	std::function<distanceMatrix_t(const std::vector<Ind_t> &)> computeDistanceMatrix =
	    [&](const auto &ar) { return defaultComputeDistanceMatrix(ar); };

	// onRegister Hook, called when extension is registered to a ga instance
	void onRegister(GA &gagaInstance) {
		gagaInstance.addPostEvaluationMethod(
		    [this](GA &ga) { updateNovelty(ga.population, ga); });

		gagaInstance.addPrintStartMethod([this](const GA &) {
			std::cout << "  ▹ novelty is " << GAGA_COLOR_GREEN << "enabled" << GAGA_COLOR_NORMAL
			          << std::endl;
			std::cout << "    - Nearest Neighbors size = " << GAGA_COLOR_BLUE << K
			          << GAGA_COLOR_NORMAL << std::endl;
			std::cout << "    - Local Competition " << GAGA_COLOR_BLUE << (nslc ? "ON" : "OFF")
			          << GAGA_COLOR_NORMAL << std::endl;
			std::cout << "    - Individual additions per generation = " << GAGA_COLOR_BLUE
			          << nbOfArchiveAdditionsPerGeneration << GAGA_COLOR_NORMAL << std::endl;
		});
		gagaInstance.addPrintIndividualMethod(
		    [](const GA &ga, const auto &ind) -> std::string {
			    if (ga.getVerbosity() >= 2) return signatureToString(ind.signature);
			    return "";
		    });

		gagaInstance.addSavePopMethod([this](const GA &ga) {
			if (saveArchiveEnabled) saveArchive(ga);
		});

		gagaInstance.addEnabledObjectivesMethod([this](const GA &, auto &objectives) {
			if (nslc) {
				objectives.clear();
				objectives.insert("novelty");
				objectives.insert("local_score");
			}
		});
	}

	bool isArchived(const Ind_t &ind) {
		bool archived = false;
		for (const auto &archInd : archive) {
			if (archInd.id == ind.id) {
				archived = true;
				break;
			}
		}
		return archived;
	}

	// SQL bindings
	template <typename SQLPlugin, typename SQLStatement>
	void onSQLiteRegister(SQLPlugin &saver) {
		std::vector<std::tuple<
		    std::string, std::string,
		    std::function<void(const Ind_t &, SQLPlugin &, size_t, SQLStatement *)>>>
		    columns;
		columns.emplace_back("archived", "BOOLEAN",
		                     [&](const Ind_t &individual, SQLPlugin &sqlPlugin, size_t index,
		                         SQLStatement *stmt) {
			                     sqlPlugin.sqbind(stmt, index, isArchived(individual));
		                     });
		columns.emplace_back("signature", "TEXT",
		                     [](const Ind_t &individual, SQLPlugin &sqlPlugin, size_t index,
		                        SQLStatement *stmt) {
			                     json jsSig = individual.signature;
			                     sqlPlugin.sqbind(stmt, index, jsSig.dump());
		                     });
		saver.addIndividualColumns(columns);
	}

	// CORE ALGO
	/*********************************************************************************
	 *                          NOVELTY RELATED METHODS
	 ********************************************************************************/
	// Novelty works with signatures. A signature is just a std::vector of std::vector of
	// doubles. It is recommended that those doubles are within a same order of magnitude.
	// Each std::vector<double> is a "snapshot": it represents the state of the evaluation
	// of one individual at a certain time. Thus, a complete signature is a combination of
	// one or more snapshot taken at different points in the simulation (a
	// std::vector<vector<double>>). Snapshot must be of same size accross individuals.
	// Signature must be set in the evaluator (see examples)
	// Options for novelty:
	//  - Local Competition:
	//

	distanceMatrix_t defaultComputeDistanceMatrix(const std::vector<Ind_t> &ar) {
		// this computes both dist(i,j) and dist(j,i), so they can be different.
		distanceMatrix_t dmat(ar.size(), std::vector<double>(ar.size()));
		for (size_t i = 0; i < ar.size(); ++i) {
			for (size_t j = 0; j < ar.size(); ++j) {
				if (i != j)
					dmat[i][j] = computeSignatureDistance(ar[i].signature, ar[j].signature);
			}
		}
		return dmat;
	}

	std::vector<size_t> findKNN(size_t i, size_t knnsize, const distanceMatrix_t &dmat) {
		// returns the K nearest neighbors of i, according to the distance matrix dmat
		if (dmat.size() == 0) return std::vector<size_t>();
		assert(dmat[i].size() == dmat.size());
		assert(i < dmat.size());
		const std::vector<double> &distances = dmat[i];
		std::vector<size_t> indices(distances.size());
		std::iota(indices.begin(), indices.end(), 0);

		size_t k = std::max(std::min(knnsize, distances.size() - 1), (size_t)0u);

		std::nth_element(
		    indices.begin(), indices.begin() + k, indices.end(),
		    [&distances](size_t a, size_t b) { return distances[a] < distances[b]; });

		indices.erase(std::remove(indices.begin(), indices.end(), i),
		              indices.end());  // remove itself from the knn list
		indices.resize(k);
		return indices;
	}

	void updateNovelty(std::vector<Ind_t> &population, GA &ga) {
		// we append the current population to the archive
		auto savedArchiveSize = archive.size();
		for (auto &ind : population) archive.push_back(ind);

		// we compute the distance matrix.
		// distanceMatrix[i][j] == distanceMatrix[j][i] == distance(archive[i], archive[j])
		std::vector<std::vector<double>> distanceMatrix = computeDistanceMatrix(archive);

		// then update the novelty field of every member of the population
		for (size_t p_i = 0; p_i < population.size(); p_i++) {
			size_t i = savedArchiveSize + p_i;  // individuals'id in the archive
			assert(population[p_i].id == archive[i].id);

			std::vector<size_t> knn = findKNN(i, K, distanceMatrix);

			if (nslc) {  // local competition is enabled
				// we put all objectives other than novelty into objs
				auto objs = GA::getAllObjectives(population[p_i]);
				if (objs.count("novelty")) objs.erase("novelty");
				if (objs.count("local_score")) objs.erase("local_score");

				std::vector<Ind_t *> knnPtr;  // pointers to knn individuals
				for (auto k : knn) knnPtr.push_back(&archive[k]);
				knnPtr.push_back(&archive[i]);  // + itself

				// we normalize the rank
				double knnSize = knn.size() > 0 ? static_cast<double>(knn.size()) : 1.0;
				double localScore =
				    static_cast<double>(ga.getParetoRank(knnPtr, knnPtr.size(), objs)) / knnSize;
				population[p_i].fitnesses["local_score"] = 1.0 - localScore;
			}

			// sum = sum of distances between i and its knn
			// novelty = avg dist to knn
			double sum = 0;
			for (auto &j : knn) sum += distanceMatrix[i][j];
			population[p_i].fitnesses["novelty"] = sum / (double)knn.size();

			ga.printLn(2, "Novelty for ind ", population[p_i].id, " -> ",
			           population[p_i].fitnesses["novelty"]);
			ga.printLn(3, "Ind ", population[p_i].id, " signature is ",
			           signatureToString(population[p_i].signature));
		}

		// now we add individuals to the archive

		std::vector<Ind_t> toBeAdded;
		std::uniform_int_distribution<size_t> d(0, population.size() - 1);
		for (size_t i = 0; i < nbOfArchiveAdditionsPerGeneration; ++i)
			toBeAdded.push_back(population[d(GA::globalRand())]);

		// first we erase the entire pop that we had appended to the archive
		archive.erase(archive.begin() + static_cast<long>(savedArchiveSize), archive.end());
		// then we add the newly selected individuals
		archive.insert(std::end(archive), std::begin(toBeAdded), std::end(toBeAdded));
		ga.printLn(2, "Added ", toBeAdded.size(), " new individuals to the archive.");
		ga.printLn(2, "New archive size = ", archive.size());
	}

	template <typename T> static inline std::string signatureToString(const T &f) {
		std::ostringstream res;
		res << "👣  " << json(f).dump();
		return res.str();
	}
	void saveArchive(const GA &ga) {
		json o = Ind_t::popToJSON(archive);
		o["evaluator"] = ga.getEvaluatorName();
		std::stringstream baseName;
		baseName << ga.getSaveFolder() << "/gen" << ga.getCurrentGenerationNumber();
		GA::mkd(baseName.str().c_str());
		std::stringstream fileName;
		fileName << baseName.str() << "/archive" << ga.getCurrentGenerationNumber() << ".pop";
		std::ofstream file;
		file.open(fileName.str());
		file << o.dump();
		file.close();
	}

	// DISTRIBUTED VERSION (works with gagazmq)
	template <typename S> void enableDistributed(S &server) {
		setComputDistanceMatrixFunction(
		    [&](const auto &a) { return distributedComputeDistanceMatrix(a, server); });
	}
	void disableDistributed() {
		setComputDistanceMatrixFunction(
		    [&](const auto &ar) { return defaultComputeDistanceMatrix(ar); });
	}
	//----------------------------------------------------------------------------
	// distributedComputeDistanceMatrix
	//----------------------------------------------------------------------------
	// Computes the distance matrix while avoinding unecessary new recomputations
	//---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---
	// We assume individuals with same id don't change between generations, and that the
	// distance between 2 old inds is stable over time. The top left part of the matrix
	// until the first new individual wont be recomputed. To use that, gaga should always
	// try to append new individuals at the end of the archive vector.

	std::vector<Ind_t> prevArchive;
	distanceMatrix_t prevDistanceMat;

	template <typename SERVER>
	distanceMatrix_t distributedComputeDistanceMatrix(const std::vector<Ind_t> &ar,
	                                                  SERVER &server) {
		// the distance matrix, first filled with zeros.
		distanceMatrix_t dmat(ar.size(), std::vector<double>(ar.size()));

		// finding the id of the first new individual (before which we don't need to
		// recompute distances)
		size_t firstNewId = 0;
		for (size_t i = 0; i < ar.size() && i < prevArchive.size(); ++i) {
			if (ar[i].id == prevArchive[i].id)
				firstNewId = i;
			else
				break;
		}

		// we fill the new distmatrix with the distances we already know
		for (size_t i = 0; i < firstNewId; ++i) {
			for (size_t j = 0; j < firstNewId; ++j) {
				const auto &d = prevDistanceMat[i][j];
				dmat[i][j] = d;
				dmat[j][i] = d;
			}
		}

		// tasks = pairs of ar id for which the workers should compute a distance
		std::vector<std::pair<size_t, size_t>> distancePairs;
		distancePairs.reserve(0.5 * std::pow(ar.size() - firstNewId, 2));
		for (size_t i = 0; i < ar.size(); ++i) {
			for (size_t j = std::max(firstNewId, i + 1); j < ar.size(); ++j) {
				distancePairs.emplace_back(i, j);
			}
		}

		// we send the archive as extra content in the request, to each client.
		auto archive_js = nlohmann::json::array();
		for (const auto &i : ar) archive_js.push_back(i.toJSON());
		json extra_js{{"archive", archive_js}};

		// called whenever results are sent by a worker. We just update the distance matrix
		auto distanceResults = [&](const auto &req) {
			auto distances = req.at("distances");
			for (auto &d : distances) {  // d = [i, j, dist]
				const size_t &i = d[0];
				const size_t &j = d[1];
				assert(i < ar.size());
				assert(j < ar.size());
				double dist = d[2];
				dmat[i][j] = dist;
				dmat[j][i] = dist;
			}
			return distances.size();
		};

		server.taskDispatch("DISTANCE", distancePairs, distanceResults, extra_js);

		prevDistanceMat = dmat;
		prevArchive = ar;

		return dmat;
	}
};
}  // namespace GAGA
