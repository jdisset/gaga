#pragma once
#include <random>
#include <vector>
#include "../third_party/json.hpp"


/**
 * @brief a configurable vector DNA with single point crossover and json serialization
 */
namespace GAGA {

template <typename T, typename R>
size_t chooseAction(const std::vector<T> probs, R& randEngine) {
	std::uniform_real_distribution<double> dChoice(0, 1);
	double type = dChoice(randEngine);
	size_t i = 0;
	while (type > 0 && i < probs.size()) {
		type -= probs[i++];
	}
	return i;
}

template <typename T> struct VectorDNA {
	struct Config {
		size_t initialSize = 2;
		size_t minSize = 1;
		size_t maxSize = 100;
		bool mutateSize = false;
		double modifyProba = 0.8;
		double addProba = 0.1;
		double eraseProba = 0.1;
	};

	using json = nlohmann::json;
	std::vector<T> values;
	Config* cfg = nullptr;

	VectorDNA() {}
	explicit VectorDNA(Config* c) : cfg(c) {}

	explicit VectorDNA(const std::string& s) {
		auto o = json::parse(s);
		values = o.at("values");
	}

	std::string serialize() const {
		json o;
		o["values"] = values;
		return o.dump();
	}

	void reset() {}

	void mutate() {
		assert(cfg);
		size_t action = 0;  // default = modify
		if (cfg->mutateSize) {
			action =
			    chooseAction(std::vector<T>{{cfg->modifyProba, cfg->addProba, cfg->eraseProba}},
			                 getRandomEngine());
		}
		std::uniform_real_distribution<T> d(0, 1);  // TODO replace by generic rand
		std::uniform_int_distribution<int> dInt(0, values.size() - 1);
		switch (action) {
			case 0:
				// modify
				values[dInt(getRandomEngine())] = d(getRandomEngine());
				break;
			case 1:
				// add
				values.push_back(d(getRandomEngine()));
				break;

			case 2:
			default:
				// erase
				values.erase(values.begin() + dInt(getRandomEngine()));
				break;
		}
	}

	static VectorDNA random(Config* c) {
		assert(c);
		VectorDNA res(c);
		res.values.resize(c->initialSize);
		std::uniform_real_distribution<T> d(0, 1);  // TODO replace by generic rand
		for (size_t i = 0; i < c->initialSize; ++i) res.values[i] = d(getRandomEngine());
		return res;
	}

	static std::mt19937& getRandomEngine() {
		static std::mt19937 gen;
		return gen;
	}

	VectorDNA crossover(const VectorDNA& other) {  // fixed point crossover
		assert((*(other.cfg)).to_json() == (*cfg).to_json());
		VectorDNA res(cfg);
		for (size_t i = 0; i < values.size() / 2; ++i) res.values.push_back(values[i]);
		for (size_t i = other.values.size() / 2; i < other.values.size(); ++i)
			res.values.push_back(other.values[i]);
		return res;
	}
};
}  // namespace GAGA
