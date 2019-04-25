#include "../../gaga.hpp"
#include "third_party/zmq.hpp"

namespace GAGA {

using json = nlohmann::json;
using request_t = std::pair<std::string, json>;  // identity + request content

// ----------------------------------------------------
//                   helpers for zmq
// ----------------------------------------------------
inline zmq::message_t recvMessage(zmq::socket_t& socket) {
	zmq::message_t message;
	socket.recv(&message);
	return message;
}
inline std::string recvString(zmq::socket_t& socket) {
	auto message = recvMessage(socket);
	return std::string(static_cast<char*>(message.data()), message.size());
}

inline json recvJson(zmq::socket_t& socket) {
	zmq::message_t req;
	socket.recv(&req);
	return json::parse(static_cast<char*>(req.data()),
	                   static_cast<char*>(req.data()) + req.size());
}

inline json recvMsgpack(zmq::socket_t& socket) {
	zmq::message_t req;
	socket.recv(&req);
	return json::from_msgpack(static_cast<char*>(req.data()),
	                          static_cast<char*>(req.data()) + req.size());
}

inline void sendStr(zmq::socket_t& socket, const std::string& identity,
                    const std::string& strg) {
	{
		std::string s = identity;
		zmq::message_t m(s.size());
		memcpy(m.data(), s.data(), s.size());
		socket.send(m, ZMQ_SNDMORE);
	}
	{
		zmq::message_t m(0);
		socket.send(m, ZMQ_SNDMORE);
	}
	{
		std::string s = strg;
		zmq::message_t m(s.size());
		memcpy(m.data(), s.data(), s.size());
		socket.send(m);
	}
}

inline void sendJson(zmq::socket_t& socket, const std::string& identity, const json& j) {
	{
		std::string s = identity;
		zmq::message_t m(s.size());
		memcpy(m.data(), s.data(), s.size());
		socket.send(m, ZMQ_SNDMORE);
	}
	{
		zmq::message_t m(0);
		socket.send(m, ZMQ_SNDMORE);
	}
	{
		std::string s = j.dump(1);
		zmq::message_t m(s.size());
		memcpy(m.data(), s.data(), s.size());
		socket.send(m);
	}
}

inline void sendMsgpack(zmq::socket_t& socket, const std::string& identity,
                        const json& j) {
	{
		std::string s = identity;
		zmq::message_t m(s.size());
		memcpy(m.data(), s.data(), s.size());
		socket.send(m, ZMQ_SNDMORE);
	}
	{
		zmq::message_t m(0);
		socket.send(m, ZMQ_SNDMORE);
	}
	{
		auto s = nlohmann::json::to_msgpack(j);
		zmq::message_t m(s.size());
		memcpy(m.data(), s.data(), s.size());
		socket.send(m);
	}
}

template <typename GA_t> class ZMQWorker {
	std::string addr = "tcp://localhost:4321";
	zmq::context_t context;
	zmq::socket_t socket;
	bool useMsgpack = false;

	void simpleSendJson(const nlohmann::json& js) {
		std::string req = js.dump();
		zmq::message_t m(req.size());
		memcpy(m.data(), req.data(), req.size());
		socket.send(m);
	}
	void simpleSendMsgpack(const nlohmann::json& js) {
		auto req = json::to_msgpack(js);
		zmq::message_t m(req.size());
		memcpy(m.data(), req.data(), req.size());
		socket.send(m);
	}

	std::function<void(const json&)> send = [=](const json& j) { this->simpleSendJson(j); };
	std::function<nlohmann::json(zmq::socket_t&)> recv = recvJson;

 public:
	using ind_t = typename GA_t::Ind_t;
	size_t batchSize = 2;
	bool debug = false;
	std::function<void(ind_t&)> evaluate = [](ind_t&) {};

	ZMQWorker(std::string serverAddr)
	    : addr(serverAddr), context(1), socket(context, ZMQ_REQ) {}

	void setCompression(bool comp) {
		useMsgpack = comp;
		if (useMsgpack) {
			send = [=](const json& j) { this->simpleSendMsgpack(j); };
			recv = recvMsgpack;
		} else {
			send = [=](const json& j) { this->simpleSendJson(j); };
			recv = recvJson;
		}
	}

	void start() {
		if (debug) std::cerr << " Starting worker, connecting to " << addr << std::endl;
		socket.connect(addr);
		bool listening = true;

		while (listening) {
			// send the ready request
			{
				if (debug) std::cerr << "Sending READY" << std::endl;
				nlohmann::json req_json = {{"req", "READY"}, {"qtty", batchSize}};
				send(req_json);
			}

			auto rep_json = recv(socket);

			if (!rep_json.count("req")) {
				std::string errMsg = "couldn't understand server's reply - no req field - ";
				throw std::runtime_error(errMsg);
			}
			if (rep_json["req"] == "EVAL") {
				if (debug) std::cerr << "received EVAL req" << std::endl;
				// we evaluate all individuals
				nlohmann::json evaluatedIndividuals = nlohmann::json::array();
				for (const auto& i : rep_json["tasks"]) {
					ind_t ind(i);
					evaluate(ind);
					evaluatedIndividuals.push_back(ind.toJSON());
				}
				// send the results
				{
					nlohmann::json req_json = {{"req", "RESULT"},
					                           {"individuals", evaluatedIndividuals}};
					if (debug) std::cerr << "WORKER SENDING RESULTS" << std::endl;
					send(req_json);
				}
				recvMessage(socket);  // ACK
			} else if (rep_json["req"] == "STOP")
				listening = false;
			else {
				if (debug)
					std::cerr << "[WARNING] Received unknown request: " << rep_json.dump()
					          << std::endl;
			}
		}
		socket.close();
		context.close();
	}
};

// ----------------------------------------------------
//                    GAGAZMQ
// ----------------------------------------------------

template <typename GA_t> struct ZMQServer {
 protected:
	std::queue<request_t> readyRequests;  // ready requests are stored here when waiting
	std::unordered_set<std::string> workingWorkers;  // list of currently working clients
	zmq::context_t context;
	zmq::socket_t socket;
	std::string port;
	int s_interrupted = 0;
	bool useMsgpack = false;
	std::function<void(zmq::socket_t&, const std::string&, const json&)> send = sendJson;
	std::function<nlohmann::json(zmq::socket_t&)> recv = recvJson;

	inline request_t recvRequest(zmq::socket_t& socket) {
		std::string identity = recvString(socket);
		recvMessage(socket);  // delimiter
		json req = recv(socket);
		return std::make_pair(identity, req);
	}

 public:
	using ind_t = typename GA_t::Ind_t;

	GA_t ga;       // GAGA instance:
	json extra{};  // the json extra is sent to the workers with each EVAL request

	ZMQServer() : context(1), socket(context, ZMQ_ROUTER) {
		ga.setEvaluateFunction([&]() { distributedEvaluate2(); });
	}

	void setCompression(bool comp) {
		useMsgpack = comp;
		if (useMsgpack) {
			send = sendMsgpack;
			recv = recvMsgpack;
		} else {
			send = sendJson;
			recv = recvJson;
		}
	}

	template <typename T, typename F>
	void taskDispatch(std::string commandName, std::vector<T> tasks, F&& processResult) {
		// taks are going to be sent as an array named "tasks" in a request whose "req" value
		// is commandName

		size_t waitingFor = tasks.size();

		while (waitingFor > 0) {
			try {
				if (tasks.size() > 0 && readyRequests.size() > 0) {
					// we have some individuals left to evaluate AND some workers ready
					auto request = readyRequests.front();  // a READY request
					readyRequests.pop();
					auto& req = request.second;  // req = the body of the request
					size_t qtty = 1u;            // default size of task batch is 1
					if (req.count("qtty"))       // a worker can ask for different size via qtty
						qtty = std::min(tasks.size(), req.at("qtty").template get<size_t>());

					json taskArray = json::array();

					for (size_t i = 0; i < qtty; ++i) {
						taskArray.push_back(tasks.back());
						tasks.pop_back();
					}

					json rep = {{"req", commandName}, {"tasks", taskArray}};
					ga.printLn(3, "Sending ", tasks.size(), " ", commandName, " tasks to ",
					           request.first);
					send(socket, request.first, rep);
					workingWorkers.insert(request.first);  // add the worker's to the working list
				} else {
					auto request = recvRequest(socket);
					auto& req = request.second;
					if (req.at("req") == "READY") {
						readyRequests.push(request);
						ga.printLn(3, "Received READY from ", request.first);
					} else if (req.at("req") == "RESULT") {
						if (!workingWorkers.count(request.first)) {
							ga.printWarning("An unknown worker just sent a result (worker = ",
							                request.first, " ; req = ", req.dump(), ")");
							// really, this shouldn't happen...
						} else {
							workingWorkers.erase(request.first);
						}

						size_t numberOfTasksTreated = std::forward<F>(processResult)(req);
						waitingFor -= numberOfTasksTreated;
						sendStr(socket, request.first, "");
					}
				}
			} catch (const std::exception& e) {
				ga.printError("Exception was raised, aborting. Exception :", e.what());
				s_interrupted = 1;
			} catch (...) {
				ga.printError("Unknown exception was raised, aborting");
				s_interrupted = 1;
			}
			if (s_interrupted) {
				terminate();
				exit(0);
			}
		}
	}

	void distributedEvaluate2() {
		std::vector<json> individualsToEvaluate;

		for (size_t i = 0; i < ga.population.size(); ++i) {
			if (ga.getEvaluateAllIndividuals() || !ga.population[i].evaluated) {
				assert(ga.population[i].id.second == i);
				individualsToEvaluate.push_back(ga.population[i].toJSON());
				ga.population[i].wasAlreadyEvaluated = false;
			} else {
				ga.population[i].evalTime = 0.0;
				ga.population[i].wasAlreadyEvaluated = true;
			}
		}

		auto evalResults = [&](const auto& req) {
			// results is a json reply containing the evaluated individuals
			auto individuals = req.at("individuals");
			for (auto& i : individuals) {
				auto id = i.at("id").template get<std::pair<size_t, size_t>>();
				assert(id.first == ga.currentGeneration);
				ind_t ind(i);
				// we only write fitnesses, footprint, infos and evalTime
				assert(id.second == ga.population[id.second].id.second);
				ga.population[id.second].fitnesses = ind.fitnesses;
				ga.population[id.second].footprint = ind.footprint;
				ga.population[id.second].infos = ind.infos;
				ga.population[id.second].evalTime = ind.evalTime;
				ga.population[id.second].evaluated = true;
				if (ga.getVerbosity() >= 2) ga.printIndividualStats(ga.population[id.second]);
			}
			return individuals.size();
		};

		taskDispatch("EVAL", individualsToEvaluate, evalResults);
	}

	// distanceMap_t distributedComputeDistanceMap() {
	//// TODO
	//// in GAGA:
	//// - change updateNovelty to first compute a distance map (pair of ind ->
	/// distance)
	//// before calling computeAvgDist
	//// - the computeDistanceMap should be a lambda that takes an archive (vec of ind)
	//// - change computeAvgDist to take distance map as argument (and stop directly
	//// computing distances)
	////
	//// this function distributes unique pairs from the archive in defined chunk sizes.
	//// Very similar to distributedEvaluate
	// std::vector<signaturePair_t> signaturePairs;
	// json j = {{"req", "DIST"}, {"pairs", signaturePairs}};
	//	}

	void terminate() {
		ga.printLn(2, "Terminating server, sending STOP signal to all workers");
		json stop = {{"req", "STOP"}};
		// we send a stop to all workers
		int timeout = 400;
		zmq_setsockopt(socket, ZMQ_RCVTIMEO, &timeout, sizeof(int));
		// same for all working workers
		for (const auto& w : workingWorkers) {
			send(socket, w, stop);
			send(socket, w, stop);
		}
		workingWorkers.clear();
		// first we check if some workers are still sending stuff
		zmq::message_t message;
		while (socket.recv(&message)) {
			// a worker has sent its identity
			std::string id(static_cast<char*>(message.data()), message.size());
			recvMessage(socket);
			json j = recv(socket);
			// we add it to the readyRequests (even if it's not a ready request...)
			readyRequests.push(std::make_pair(id, j));
		}

		// then we go through all readyRequests and tell the senders to stop working
		while (readyRequests.size() > 0) {
			auto r = readyRequests.front();
			readyRequests.pop();
			send(socket, r.first, stop);
			recvMessage(socket);
			send(socket, r.first, stop);
		}
		socket.close();
		context.close();
	}

	void bind(std::string serverPort = "tcp://*:4321") {
		port = serverPort;
		socket.bind(port);
		// s_catch_signals();
	}
};  // namespace GAGA

}  // namespace GAGA
