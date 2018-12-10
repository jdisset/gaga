#pragma once
#include <mpi.h>
#include <cstring>
#include "extra/mpigaga.hpp"

// WARNING: UNTESTED !!!

template <typename G> inline initMPI(G &ga) {
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nbProcs);
	MPI_Comm_rank(MPI_COMM_WORLD, &procId);
	if (procId == 0) {
		if (verbosity >= 3) {
			std::cout << "   -------------------" << endl;
			std::cout << GAGA_COLOR_CYAN << " MPI STARTED WITH " << GAGA_COLOR_NORMAL << nbProcs
			          << GAGA_COLOR_CYAN << " PROCS " << GAGA_COLOR_NORMAL << endl;
			std::cout << "   -------------------" << endl;
			std::cout << "Initialising population in master process" << endl;
		}
	}
}

void MPI_distributePopulation() {
	if (procId == 0) {
		// if we're in the master process, we send batches to the others.
		// master will have the remaining
		size_t batchSize = population.size() / nbProcs;
		for (size_t dest = 1; dest < (size_t)nbProcs; ++dest) {
			vector<Ind_t> batch;
			for (size_t ind = 0; ind < batchSize; ++ind) {
				batch.push_back(population.back());
				population.pop_back();
			}
			string batchStr = Ind_t::popToJSON(batch).dump();
			std::vector<char> tmp(batchStr.begin(), batchStr.end());
			tmp.push_back('\0');
			MPI_Send(tmp.data(), tmp.size(), MPI_BYTE, dest, 0, MPI_COMM_WORLD);
		}
	} else {
		// we're in a slave process, we welcome our local population !
		int strLength;
		MPI_Status status;
		MPI_Probe(0, 0, MPI_COMM_WORLD, &status);  // we want to know its size
		MPI_Get_count(&status, MPI_CHAR, &strLength);
		char *popChar = new char[strLength + 1];
		MPI_Recv(popChar, strLength, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		// and we dejsonize !
		auto o = json::parse(popChar);
		population = Ind_t::loadPopFromJSON(o);  // welcome
		delete[] popChar;
		if (verbosity >= 3) {
			std::ostringstream buf;
			buf << endl
			    << "Proc " << GAGA_COLOR_PURPLE << procId << GAGA_COLOR_NORMAL
			    << " : reception of " << population.size() << " new individuals !" << endl;
			cout << buf.str();
		}
	}
}

void MPI_receivePopulation() {
	if (procId != 0) {  // if slave process we send our population to our mighty leader
		string batchStr = Ind_t::popToJSON(population).dump();
		std::vector<char> tmp(batchStr.begin(), batchStr.end());
		tmp.push_back('\0');
		MPI_Send(tmp.data(), tmp.size(), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
	} else {
		// master process receives all other batches
		for (size_t source = 1; source < (size_t)nbProcs; ++source) {
			int strLength;
			MPI_Status status;
			MPI_Probe(source, 0, MPI_COMM_WORLD, &status);  // determining batch size
			MPI_Get_count(&status, MPI_CHAR, &strLength);
			char *popChar = new char[strLength + 1];
			MPI_Recv(popChar, strLength + 1, MPI_BYTE, source, 0, MPI_COMM_WORLD,
			         MPI_STATUS_IGNORE);
			// and we dejsonize!
			auto o = json::parse(popChar);
			vector<Ind_t> batch = Ind_t::loadPopFromJSON(o);
			population.insert(population.end(), batch.begin(), batch.end());
			delete[] popChar;
			if (verbosity >= 3) {
				cout << endl
				     << "Proc " << procId << " : reception of " << batch.size()
				     << " treated individuals from proc " << source << endl;
			}
		}
	}
}
