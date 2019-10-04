#pragma once
#include <sqlite3.h>

#include <cassert>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

template <typename GA> struct SQLiteSaver {
	using Ind_t = typename GA::Ind_t;

	int currentRunId = -1;
	sqlite3 *db = nullptr;

	std::unordered_map<std::string, size_t> objectivesId;  // objs ids in db

	std::vector<std::vector<size_t>> gagaToSQLiteIds;  // individuals ids in db

	std::vector<size_t> generationIds;  // generation ids in db

	template <typename E> void useExtension(E &e) {
		e.template onSQLiteRegister<SQLiteSaver, sqlite3_stmt>(*this);
	}

	std::vector<std::function<void(SQLiteSaver &)>> newRunExtras;
	std::vector<std::function<void(SQLiteSaver &, const GA &)>> newGenExtras;

	template <typename F1, typename F2>
	void addExtraTableInstructions(const F1 &onNewRun, const F2 &onNewGen) {
		newRunExtras.push_back(onNewRun);
		newGenExtras.push_back(onNewGen);
	}

	std::vector<std::tuple<
	    std::string, std::string,
	    std::function<void(const Ind_t &, SQLiteSaver &, size_t, sqlite3_stmt *)>>>
	    extraIndividualColumns;

	template <typename C> void addIndividualColumns(const C &c) {
		extraIndividualColumns.insert(extraIndividualColumns.end(), c.begin(), c.end());
	}

	SQLiteSaver(std::string dbfile) {
		if (sqlite3_open(dbfile.c_str(), &db)) throw std::runtime_error(sqlite3_errmsg(db));
	}

	void createTables() {
		if (!db) throw std::invalid_argument("db pointer is null");
		std::string sql =
		    "CREATE TABLE IF NOT EXISTS run("
		    "id INTEGER PRIMARY KEY ,"
		    "config TEXT,"
		    "start DATETIME,"
		    "duration REAL);"
		    "CREATE TABLE IF NOT EXISTS individual("
		    "id INTEGER PRIMARY KEY,"
		    "dna TEXT,"
		    "eval_time REAL,"
		    "already_evaluated BOOLEAN,"
		    "infos TEXT,";
		for (auto &c : extraIndividualColumns) {
			sql += std::get<0>(c) + " " + std::get<1>(c) + ",";
		}
		sql +=
		    "original_id INTEGER,"
		    "id_generation INTEGER);"
		    "CREATE TABLE IF NOT EXISTS generation("
		    "id INTEGER PRIMARY KEY,"
		    "number INTEGER,"
		    "duration REAL,"
		    "id_run INTEGER);"
		    "CREATE TABLE IF NOT EXISTS objective("
		    "id INTEGER PRIMARY KEY,"
		    "name TEXT,"
		    "type TEXT CHECK(type IN ('MAX','MIN')) NOT NULL DEFAULT 'MAX',"
		    "id_run INTEGER);"
		    "CREATE TABLE IF NOT EXISTS inheritance("
		    "id_parent INTEGER,"
		    "id_child INTEGER,"
		    "type TEXT CHECK(type IN ('M','X','C','')) DEFAULT '',"
		    "PRIMARY KEY (id_parent, id_child));"
		    "CREATE TABLE IF NOT EXISTS evaluation("
		    "id_objective INTEGER,"
		    "id_individual INTEGER,"
		    "value REAL,"
		    "PRIMARY KEY (id_objective, id_individual));";

		exec(sql);
	}

	void newRun(const std::string &conf = "") {
		createTables();
		std::ostringstream runReq;
		runReq << "INSERT INTO run(config,start,duration) VALUES ('" << conf
		       << "',datetime('now'),0);";
		exec(runReq.str());
		currentRunId = sqlite3_last_insert_rowid(db);
		assert(currentRunId >= 0);
		for (auto &f : newRunExtras) f(*this);
	}

	void insertAllIndividuals(size_t idGeneration, const GA &ga) {
		assert(idGeneration >= 0);
		auto &population = ga.previousGenerations.back();
		std::string sql =
		    "INSERT INTO individual "
		    "(dna,eval_time,already_evaluated,infos,";
		for (auto &c : extraIndividualColumns) sql += std::get<0>(c) + ",";
		sql +=
		    "original_id, id_generation) "
		    "VALUES "
		    "(?1, ?2, ?3, ?4,";
		const size_t FIRST_ADDED_COL = 5;
		size_t i = FIRST_ADDED_COL;

		for (size_t c = 0; c < extraIndividualColumns.size() + 1; c++)
			sql += " ?" + std::to_string(i++) + ",";
		sql += " ?" + std::to_string(i) + ");";
		sqlite3_stmt *stmt;
		prepare(sql, &stmt);

		for (const auto &ind : population) {
			bind(stmt, ind.dna.serialize(), ind.evalTime, ind.wasAlreadyEvaluated, ind.infos);

			// adding extra columns values (from extensions)
			size_t i = FIRST_ADDED_COL;
			for (const auto &c : extraIndividualColumns) std::get<2>(c)(ind, *this, i++, stmt);

			sqbind(stmt, i++, ind.id.second);  // original id
			sqbind(stmt, i, idGeneration);
			step(stmt);
			size_t idInd = sqlite3_last_insert_rowid(db);
			gagaToSQLiteIds.back().push_back(idInd);  // we save the id of this new individual
		}
	}

	size_t getLastInsertId() { return sqlite3_last_insert_rowid(db); }

	void insertNewGeneration(const GA &ga) {
		int idGeneration = -1;
		int generationNumber = ga.getCurrentGenerationNumber() - 1;
		{
			std::string sql =
			    "INSERT INTO generation"
			    "(number, duration, id_run) "
			    "VALUES "
			    "(?1, ?2, ?3);";
			sqlite3_stmt *stmt;
			prepare(sql, &stmt);
			bind(stmt, generationNumber, ga.genStats.back().at("global").at("genTotalTime"),
			     currentRunId);
			step(stmt);
			idGeneration = sqlite3_last_insert_rowid(db);
			generationIds.push_back(idGeneration);
			gagaToSQLiteIds.push_back(std::vector<size_t>());
		}

		if (generationNumber == 0) {  // insert objectives
			std::string sql =
			    "INSERT INTO objective"
			    "(name, type, id_run) "
			    "VALUES "
			    "(?1, ?2, ?3);";
			sqlite3_stmt *stmt;
			prepare(sql, &stmt);
			assert(ga.previousGenerations.back().size() > 0);
			for (const auto &o : ga.previousGenerations.back()[0].fitnesses) {
				std::string type = "MAX";
				bind(stmt, o.first, type, currentRunId);
				step(stmt);
				int idObj = getLastInsertId();
				objectivesId[o.first] = idObj;
			}
		}
	}

	void prepare(const std::string &sql, sqlite3_stmt **stmt) {
		sqlite3_prepare_v2(db, sql.c_str(), -1, stmt, 0);
	}

	void newGen(const GA &ga) {
		auto t0 = std::chrono::high_resolution_clock::now();
		assert(currentRunId >= 0);

		insertNewGeneration(ga);
		size_t idGeneration = generationIds.back();
		insertAllIndividuals(idGeneration, ga);
		assert(gagaToSQLiteIds.size() == ga.getCurrentGenerationNumber());

		{  // insert evaluations
			std::string sql =
			    "INSERT INTO evaluation"
			    "(id_objective, id_individual, value) "
			    "VALUES "
			    "(?1, ?2, ?3);";
			sqlite3_stmt *stmt;
			sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, 0);
			for (const auto &ind : ga.previousGenerations.back()) {
				for (const auto &o : ind.fitnesses) {
					bind(stmt, objectivesId.at(o.first), getIndId(ind.id), o.second);
					step(stmt);
				}
			}
		}

		{  // insert inheritance relations
			std::string sql =
			    "INSERT INTO inheritance"
			    "(id_parent, id_child, type) "
			    "VALUES "
			    "(?1, ?2, ?3);";
			sqlite3_stmt *stmt;
			sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, 0);
			for (const auto &ind : ga.previousGenerations.back()) {
				for (const auto &p : ind.parents) {
					std::string type = "";
					if (ind.inheritanceType == "mutation")
						type = 'M';
					else if (ind.inheritanceType == "crossover")
						type = 'X';
					else if (ind.inheritanceType == "copy")
						type = 'C';
					bind(stmt, getIndId(p), getIndId(ind.id), type);
					step(stmt);
				}
			}
		}
		assert(gagaToSQLiteIds.back().size() == ga.previousGenerations.back().size());
		for (auto &f : newGenExtras) f(*this, ga);
		auto t1 = std::chrono::high_resolution_clock::now();
		double t = std::chrono::duration<double>(t1 - t0).count();
		ga.printInfos("Time for SQLite newGen operations = ", t, "s");
	}

	size_t getIndId(std::pair<size_t, size_t> id) {
		assert(gagaToSQLiteIds.size() > id.first);
		assert(gagaToSQLiteIds[id.first].size() > id.second);
		return gagaToSQLiteIds[id.first][id.second];
	}

	void endRun() {}

	static int callbackHandler(void *actualCallback, int argc, char **argv,
	                           char **azColName) {
		auto ptr =
		    (static_cast<std::function<void(int, char **, char **)> *>(actualCallback));
		(*ptr)(argc, argv, azColName);
		return 0;
	}

	void exec(std::string sql) {
		if (!db) throw std::invalid_argument("db pointer is null");
		char *err_msg = nullptr;
		int rc = sqlite3_exec(db, sql.c_str(), 0, 0, &err_msg);
		if (rc != SQLITE_OK) {
			std::ostringstream errorMsg;
			errorMsg << "SQL error: " << err_msg << ". \n\nREQ = " << sql << std::endl;
			sqlite3_free(err_msg);
			throw std::invalid_argument(errorMsg.str());
		} else
			sqlite3_free(err_msg);
	}
	template <typename C> void exec(std::string sql, const C &callback) {
		if (!db) throw std::invalid_argument("db pointer is null");
		char *err_msg = nullptr;
		std::function<void(int, char **, char **)> cbackFunc = callback;
		int rc = sqlite3_exec(db, sql.c_str(), callbackHandler, &cbackFunc, &err_msg);
		if (rc != SQLITE_OK) {
			std::ostringstream errorMsg;
			errorMsg << "SQL error: " << err_msg << ". \n\nREQ = " << sql << std::endl;
			sqlite3_free(err_msg);
			throw std::invalid_argument(errorMsg.str());
		} else
			sqlite3_free(err_msg);
	}

	template <typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value>>
	int sqbind(sqlite3_stmt *stmt, int index, T value) {
		if constexpr (std::is_integral<T>::value) {
			if constexpr (sizeof(T) <= sizeof(int))
				return sqlite3_bind_int(stmt, index, value);
			else
				return sqlite3_bind_int64(stmt, index, value);
		}
		return sqlite3_bind_double(stmt, index, static_cast<double>(value));
	}

	int sqbind(sqlite3_stmt *stmt, int index, const std::string &value) {
		return sqlite3_bind_text(stmt, index, value.c_str(), -1, SQLITE_TRANSIENT);
	}

	template <class... Args, size_t... Is>
	int bind_impl(sqlite3_stmt *stmt, std::index_sequence<Is...>, Args &&... args) {
		return (sqbind(stmt, Is + 1, std::forward<Args>(args)) + ...);
	}

	template <class... Args> int bind(sqlite3_stmt *stmt, Args &&... args) {
		int res =
		    bind_impl(stmt, std::index_sequence_for<Args...>{}, std::forward<Args>(args)...);

		return res;
	}
	void step(sqlite3_stmt *stmt) {
		sqlite3_step(stmt);
		sqlite3_clear_bindings(stmt);
		sqlite3_reset(stmt);
	}

	~SQLiteSaver() { sqlite3_close(db); }
};
