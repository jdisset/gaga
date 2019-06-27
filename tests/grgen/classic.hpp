#ifndef CLASSIC_HPP
#define CLASSIC_HPP
#include <iostream>
#include <array>
#include <vector>
#include <utility>
#include <unordered_map>
#include "common.h"
#include "protein.hpp"

using namespace std;

struct Classic {
	public:
	// we use 3 coordinates proteins (id, enh, inh)
	static constexpr int IDSIZE = 32;
	using Protein_t = Protein<3, int, 0, IDSIZE>;

	// we need 2 parameters (beta, alpha)
	static constexpr unsigned int nbParams = 2;
	// and we produce 2 dimensional signatures (enhnance, inhibit)
	static constexpr unsigned int nbSignatureParams = 2;

	static const array<pair<double, double>, nbParams> paramsLimits() {
		return {{{0.5, 2.0}, {0.5, 2.0}}};
	}

	// helpers for proteins coords
	static inline int& getId(Protein_t& p) { return p.coords[0]; }
	static inline int& getEnh(Protein_t& p) { return p.coords[1]; }
	static inline int& getInh(Protein_t& p) { return p.coords[2]; }

	// aliases for ProteinType
	static constexpr ProteinType pinput = ProteinType::input;
	static constexpr ProteinType pregul = ProteinType::regul;
	static constexpr ProteinType poutput = ProteinType::output;

	double maxEnhance = 0.0, maxInhibit = 0.0;

	Classic() {}

	template <typename GRN> void updateSignatures(GRN& grn) {
		grn.signatures.clear();
		grn.signatures.resize(grn.actualProteins.size());
		for (size_t i = 0; i < grn.actualProteins.size(); ++i) {
			grn.signatures[i].resize(grn.actualProteins.size());
			for (size_t j = 0; j < grn.actualProteins.size(); ++j) {
				auto& p0 = grn.actualProteins[i];
				auto& p1 = grn.actualProteins[j];
				grn.signatures[i][j] = {
				    {static_cast<double>(IDSIZE - abs(getEnh(p0) - getId(p1))),
				     static_cast<double>(IDSIZE - abs(getInh(p0) - getId(p1)))}};
				if (grn.signatures[i][j][0] > maxEnhance) maxEnhance = grn.signatures[i][j][0];
				if (grn.signatures[i][j][1] > maxInhibit) maxInhibit = grn.signatures[i][j][1];
			}
		}
		// std::cerr << "maxEnh = " << maxEnhance << ", maxInh = " << maxInhibit << std::endl;
		for (size_t i = 0; i < grn.actualProteins.size(); ++i) {
			for (size_t j = 0; j < grn.actualProteins.size(); ++j) {
				grn.signatures[i][j] = {
				    {exp(grn.params[0] * grn.signatures[i][j][0] - maxEnhance),
				     exp(grn.params[0] * grn.signatures[i][j][1] - maxInhibit)}};
			}
		}
	}

	template <typename GRN> void step(GRN& grn, unsigned int nbSteps) {
		for (auto s = 0u; s < nbSteps; ++s) {
			std::vector<double> nextProteins;  // only reguls & outputs concentrations
			nextProteins.reserve(grn.getNbProteins() - grn.getProteinSize(ProteinType::input));
			const auto firstOutputId = grn.getFirstOutputIndex();
			for (size_t j = grn.getFirstRegulIndex(); j < grn.getNbProteins(); ++j) {
				double enh = 0.0, inh = 0.0;
				for (size_t k = 0; k < firstOutputId; ++k) {
					enh += grn.actualProteins[k].c * grn.signatures[k][j][0];
					inh += grn.actualProteins[k].c * grn.signatures[k][j][1];
				}
				nextProteins.push_back(
				    max(0.0, grn.actualProteins[j].c +
				                 (grn.params[1] / static_cast<double>(grn.getNbProteins())) *
				                     (enh - inh)));
			}
			// Normalizing regul & output proteins concentrations
			double sumConcentration = 0.0;
			for (auto i : nextProteins) {
				sumConcentration += i;
			}
			if (sumConcentration > 0) {
				for (auto& i : nextProteins) {
					i /= sumConcentration;
				}
			}
			auto firstRegulIndex = grn.getFirstRegulIndex();
			for (size_t i = firstRegulIndex; i < grn.getNbProteins(); ++i) {
				grn.actualProteins[i].c = nextProteins[i - firstRegulIndex];
			}
		}
	}
};
#endif
