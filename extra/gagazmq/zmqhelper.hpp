#include <utility>
#include <vector>

template <typename T>
void distributedTasks(std::vector<T> tasks) {
	size_t waitingFor = tasks.size();

	while (waitingFor > 0) {
		if (tasks.size() > 0 && readyRequests.size() > 0) {
			// sendwork to last ready request
		} else {
			req = recvReq();
			if (req == READY) {
				readyRequest.push(req);
			} else if (req == RESULT) {
				processResult(req);
				workingWorkers.erase(req.origin);
				waitingFor -= req.numberOfTasks;
			}
		}
	}
}
