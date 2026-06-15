// ============================================================
// DAPG Dynamic Vector Index Maintenance Experiments
// Author: Solmaz Seyed Monir
// Project: Distance-Aware Pruned Graph (DAPG)
// Description:
//   Implements DAPG search, insert/delete maintenance,
//   DAPG+Hybrid candidate augmentation, and comparison
//   utilities for update-aware ANN evaluation.
// ============================================================

#include "alg.h"
#include <iomanip> // for std::setprecision
#include <map>
#include <tuple>
#include <filesystem>
#include <random>
#include <deque>
#include <string>
#include <cstring>
#include <cmath>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <sstream>
#include <cctype>

#include <hnswlib/hnswlib.h>
#include <hnswlib/hnswalg.h>
int _lsh_UB=0;
//double _chi2inv;
//double _chi2invSqr;
//double _coeff;

// Struct to hold bests for each metric
struct MultiBest {
	SearchResult bestRecall;
	SearchResult bestCost;
	SearchResult bestTime;
	SearchResult bestPruning;
	bool hasRecall = false, hasCost = false, hasTime = false, hasPruning = false;
};

// Solmaz Seyed Monir
// This replicates graphSearch() logic but does NOT write to result.txt.
template <class Graph>
static SearchResult solmaz_graphSearch_no_write(float c, int k, Graph* myGraph, Preprocess& prep, float beta, int qType, int Qnum)
{
	if (!myGraph) return SearchResult{ -1.0f, -1.0f, -1.0f, -1.0f, "" };

	Performance perform;
	for (unsigned j = 0; j < (unsigned)Qnum; j++) {
		queryN* q = new queryN(j, c, k, prep, beta);
		switch (qType % 2) {
		case 0:
			myGraph->knn(q);
			break;
		case 1:
			myGraph->knnHNSW(q);
			break;
		}
		perform.update(q, prep);
		delete q;
	}

	float mean_time = (float)perform.timeTotal / perform.num;
	float cost = ((float)perform.cost) / ((float)perform.num);
	float ratio = (perform.cost == 0) ? 0.0f : ((float)perform.prunings) / (perform.cost);
	float cost_total = myGraph->S + cost / (1 - ratio) * (((float)myGraph->lowDim) / myGraph->dim) + cost;
	float cpq = myGraph->L * myGraph->K + _lsh_UB + cost / (1 - ratio) * (((float)myGraph->lowDim) / myGraph->dim);

	SearchResult result;
	result.recall = ((float)perform.NN_num) / (perform.num * k);
	result.cost = ((float)perform.cost) / ((float)perform.num);
	result.time = mean_time * 1000; // ms
	result.pruning = (perform.cost == 0) ? 0.0f : ((float)perform.prunings) / (perform.cost);

	
	std::stringstream ss;
	ss << std::setw(_lspace) << "Solmaz_NoWrite"
	   << std::setw(_sspace) << k
	   << std::setw(_sspace) << myGraph->ef
	   << std::setw(_lspace) << result.time
	   << std::setw(_lspace) << result.recall
	   << std::setw(_lspace) << result.cost
	   << std::setw(_lspace) << cpq
	   << std::setw(_lspace) << cost_total
	   << std::setw(_lspace) << result.pruning
	   << std::endl;
	result.rowString = ss.str();
	return result;
}

// HNSW (hnswlib) baseline runner for curve/table logging.

static SearchResult solmaz_hnsw_no_write(
	int k,
	hnswlib::HierarchicalNSW<float>* index,
	Preprocess& prep,
	int ef_search,
	int Qnum
) {
	if (!index) return SearchResult{ -1.0f, -1.0f, -1.0f, -1.0f, "" };
	index->setEf((size_t)std::max(ef_search, k));

	Performance perform;
	for (int j = 0; j < Qnum; ++j) {
		queryN q(j, /*c=*/1.5f, k, prep, /*beta=*/0.1f);
		const auto t0 = std::chrono::high_resolution_clock::now();
		auto pq = index->searchKnn((const void*)q.queryPoint, (size_t)k);
		const auto t1 = std::chrono::high_resolution_clock::now();
		q.timeTotal = (float)std::chrono::duration<double>(t1 - t0).count();
		q.timeHash = 0.0f;
		q.timeSift = q.timeTotal;
		q.cost = 0;
		q.prunings = 0;

		std::vector<Res> res;
		res.reserve((size_t)k);
		while (!pq.empty()) {
			const auto& top = pq.top();
			Res r;
			r.dist = (float)top.first;
			r.id = (int)top.second;
			res.emplace_back(r);
			pq.pop();
		}
		std::sort(res.begin(), res.end(), [](const Res& a, const Res& b) { return a.dist < b.dist; });
		if ((int)res.size() > k) res.resize((size_t)k);
		q.res = std::move(res);

		perform.update(&q, prep);
	}

	SearchResult result;
	result.recall = (perform.num > 0) ? ((float)perform.NN_num) / (perform.num * k) : 0.0f;
	result.cost = 0.0f;
	result.pruning = 0.0f;
	result.time = (perform.num > 0) ? ((float)perform.timeTotal / perform.num) * 1000.0f : 0.0f; // ms
	return result;
}

// Solmaz Seyed Monir
// Wolverine-style dynamic recall. Unlike the static bench_graph path,
// this computes exact top-k over the current active FIFO set for each query.
template <class Graph>
static SearchResult solmaz_graphSearch_active_no_write(
	float c,
	int k,
	Graph* myGraph,
	Preprocess& prep,
	float beta,
	int qType,
	int Qnum,
	const std::deque<int>& active_ids
) {
	if (!myGraph) return SearchResult{ -1.0f, -1.0f, -1.0f, -1.0f, "" };
	if (active_ids.empty()) return SearchResult{ 0.0f, 0.0f, 0.0f, 0.0f, "" };

	double time_total = 0.0;
	size_t hits = 0;
	int qnum_eff = std::min(Qnum, (int)prep.benchmark.N);

	for (int j = 0; j < qnum_eff; ++j) {
		queryN q(j, c, k, prep, beta);
		switch (qType % 2) {
		case 0:
			myGraph->knn(&q);
			break;
		case 1:
			myGraph->knnHNSW(&q);
			break;
		}
		time_total += q.timeTotal;

		std::priority_queue<std::pair<float, int>> exact_heap;
		for (int id : active_ids) {
			if (id < 0 || id >= (int)prep.data.N) continue;
			float dist = cal_dist(q.queryPoint, prep.data.val[id], prep.data.dim);
			if ((int)exact_heap.size() < k) {
				exact_heap.emplace(dist, id);
			}
			else if (dist < exact_heap.top().first) {
				exact_heap.pop();
				exact_heap.emplace(dist, id);
			}
		}

		std::unordered_set<int> exact_ids;
		exact_ids.reserve((size_t)k * 2);
		while (!exact_heap.empty()) {
			exact_ids.insert(exact_heap.top().second);
			exact_heap.pop();
		}

		std::unordered_set<int> seen_res;
		seen_res.reserve((size_t)q.res.size() * 2);
		for (const auto& r : q.res) {
			if ((int)seen_res.size() >= k) break;
			if (r.id < 0 || r.id >= (int)prep.data.N) continue;
			if (!seen_res.insert(r.id).second) continue;
			if (exact_ids.find(r.id) != exact_ids.end()) ++hits;
		}
	}

	SearchResult result;
	result.recall = (qnum_eff > 0) ? (float)((double)hits / ((double)qnum_eff * (double)k)) : 0.0f;
	result.cost = 0.0f;
	result.pruning = 0.0f;
	result.time = (qnum_eff > 0) ? (float)((time_total / (double)qnum_eff) * 1000.0) : 0.0f;
	return result;
}

// Solmaz Seyed Monir
// NEW hybrid experiment: DAPG remains the main locally maintained graph,

template <class Graph>
static SearchResult solmaz_graphSearch_active_hnsw_hybrid_no_write(
	float c,
	int k,
	Graph* myGraph,
	hnswlib::HierarchicalNSW<float>* hnsw_index,
	Preprocess& prep,
	float beta,
	int qType,
	int Qnum,
	const std::deque<int>& active_ids,
	int hnsw_ef,
	int hnsw_candidates
) {
	if (!myGraph || !hnsw_index) return SearchResult{ -1.0f, -1.0f, -1.0f, -1.0f, "" };
	if (active_ids.empty()) return SearchResult{ 0.0f, 0.0f, 0.0f, 0.0f, "" };

	hnsw_candidates = std::max(k, hnsw_candidates);
	hnsw_index->setEf((size_t)std::max(hnsw_ef, hnsw_candidates));

	double time_total = 0.0;
	size_t hits = 0;
	int qnum_eff = std::min(Qnum, (int)prep.benchmark.N);

	for (int j = 0; j < qnum_eff; ++j) {
		queryN q(j, c, k, prep, beta);
		const auto t0 = std::chrono::high_resolution_clock::now();
		switch (qType % 2) {
		case 0:
			myGraph->knn(&q);
			break;
		case 1:
			myGraph->knnHNSW(&q);
			break;
		}
		auto hpq = hnsw_index->searchKnn((const void*)q.queryPoint, (size_t)hnsw_candidates);
		const auto t1 = std::chrono::high_resolution_clock::now();
		time_total += std::chrono::duration<double>(t1 - t0).count();

		std::priority_queue<std::pair<float, int>> exact_heap;
		for (int id : active_ids) {
			if (id < 0 || id >= (int)prep.data.N) continue;
			float dist = cal_dist(q.queryPoint, prep.data.val[id], prep.data.dim);
			if ((int)exact_heap.size() < k) {
				exact_heap.emplace(dist, id);
			}
			else if (dist < exact_heap.top().first) {
				exact_heap.pop();
				exact_heap.emplace(dist, id);
			}
		}

		std::unordered_set<int> exact_ids;
		exact_ids.reserve((size_t)k * 2);
		while (!exact_heap.empty()) {
			exact_ids.insert(exact_heap.top().second);
			exact_heap.pop();
		}

		std::unordered_map<int, float> cand;
		cand.reserve((size_t)(q.res.size() + hnsw_candidates) * 2);
		for (const auto& r : q.res) {
			if (r.id < 0 || r.id >= (int)prep.data.N) continue;
			float dist = cal_dist(q.queryPoint, prep.data.val[r.id], prep.data.dim);
			auto it = cand.find(r.id);
			if (it == cand.end() || dist < it->second) cand[r.id] = dist;
		}
		while (!hpq.empty()) {
			int id = (int)hpq.top().second;
			hpq.pop();
			if (id < 0 || id >= (int)prep.data.N) continue;
			float dist = cal_dist(q.queryPoint, prep.data.val[id], prep.data.dim);
			auto it = cand.find(id);
			if (it == cand.end() || dist < it->second) cand[id] = dist;
		}

		std::vector<Res> merged;
		merged.reserve(cand.size());
		for (const auto& kv : cand) merged.emplace_back(kv.first, kv.second);
		std::sort(merged.begin(), merged.end(), [](const Res& a, const Res& b) { return a.dist < b.dist; });

		const int out_k = std::min<int>(k, (int)merged.size());
		for (int i = 0; i < out_k; ++i) {
			if (exact_ids.find(merged[(size_t)i].id) != exact_ids.end()) ++hits;
		}
	}

	SearchResult result;
	result.recall = (qnum_eff > 0) ? (float)((double)hits / ((double)qnum_eff * (double)k)) : 0.0f;
	result.cost = 0.0f;
	result.pruning = 0.0f;
	result.time = (qnum_eff > 0) ? (float)((time_total / (double)qnum_eff) * 1000.0) : 0.0f;
	return result;
}

// Solmaz Seyed Monir 
// hybrid experiment

static SearchResult solmaz_graphSearch_active_faiss_flat_no_write(
	int k,
	Preprocess& prep,
	int Qnum,
	const std::deque<int>& active_ids,
	int flat_candidates
) {
	if (active_ids.empty()) return SearchResult{ 0.0f, 0.0f, 0.0f, 0.0f, "" };
	flat_candidates = std::max(k, flat_candidates);

	double time_total = 0.0;
	size_t hits = 0;
	int qnum_eff = std::min(Qnum, (int)prep.benchmark.N);

	for (int j = 0; j < qnum_eff; ++j) {
		queryN q(j, /*c=*/1.5f, k, prep, /*beta=*/0.1f);

		const auto t0 = std::chrono::high_resolution_clock::now();
		std::priority_queue<std::pair<float, int>> cand_heap;
		for (int id : active_ids) {
			if (id < 0 || id >= (int)prep.data.N) continue;
			float dist = cal_dist(q.queryPoint, prep.data.val[id], prep.data.dim);
			if ((int)cand_heap.size() < flat_candidates) {
				cand_heap.emplace(dist, id);
			}
			else if (dist < cand_heap.top().first) {
				cand_heap.pop();
				cand_heap.emplace(dist, id);
			}
		}

		std::vector<Res> ranked;
		ranked.reserve(cand_heap.size());
		while (!cand_heap.empty()) {
			ranked.emplace_back(cand_heap.top().second, cand_heap.top().first);
			cand_heap.pop();
		}
		std::sort(ranked.begin(), ranked.end(), [](const Res& a, const Res& b) { return a.dist < b.dist; });
		const auto t1 = std::chrono::high_resolution_clock::now();
		time_total += std::chrono::duration<double>(t1 - t0).count();

		// Flat-L2 top-k over the active set is exact for this query, so the top-k
		// candidate list is also the exact active-window ground truth.
		const int out_k = std::min<int>(k, (int)ranked.size());
		hits += (size_t)out_k;
	}

	SearchResult result;
	result.recall = (qnum_eff > 0) ? (float)((double)hits / ((double)qnum_eff * (double)k)) : 0.0f;
	result.cost = 0.0f;
	result.pruning = 0.0f;
	result.time = (qnum_eff > 0) ? (float)((time_total / (double)qnum_eff) * 1000.0) : 0.0f;
	return result;
}

int main(int argc, char const* argv[])
{
#if (__cplusplus >= 201703L) || (defined(_MSVC_LANG) && (_MSVC_LANG >= 1913))
	std::cout<<"C++17!\n";
#endif

	// Solmaz: Optional "table mode" to generate a paper-style CSV for Jupyter.
	// Usage example (WSL):
	//   ./lgo mnist 0 2 18 24 80 0.95 0.9 0 solmaz_table 5000
	// Args:
	//   argv[10] = "solmaz_table" enables this mode (target-recall table)
	//   argv[10] = "solmaz_curve" enables this mode (full ef sweep curve: ef, recall, qps)
	//   argv[10] = "solmaz_ncurve" enables this mode (LSH budget sweep: n_frac/n_abs vs recall/qps)
	//   argv[11] = update_count (default 5000)
	const std::string solmaz_mode = (argc > 10) ? std::string(argv[10]) : std::string();
	bool solmaz_table_mode = (solmaz_mode == "solmaz_table");
	bool solmaz_curve_mode = (solmaz_mode == "solmaz_curve");
	bool solmaz_ncurve_mode = (solmaz_mode == "solmaz_ncurve");
	bool solmaz_wolverine_mode = (solmaz_mode == "solmaz_wolverine");
	int solmaz_update_count = (argc > 11) ? std::atoi(argv[11]) : 5000;

	// Solmaz Seyed Monir
	// NEW separate experiment path:
	// Outputs and cached indexes are written under indexes/wolverine_exp by default,
	// Optional CLI override for W.
	
	//   ./lgo sift 1 2 18 24 80 0.95 0.9 0 solmaz_table 200 W=600
	float w_override = -1.0f;
	// Solmaz Seyed Monir: FIFO sliding-window workload controls.
	// - FIFO_INIT: number of initially active points in the index (prefix if FIFO_SHUFFLE=0)
	// - FIFO_BLOCK: number of insert/delete operations per iteration
	// - FIFO_SHUFFLE: if 1, still shuffle the insertion order for the initial build (default 0 for chronological prefix)
	// - UPDATE=fifo: sliding-window delete-oldest/insert-next workload
	// - UPDATE=random_reinsert: Wolverine-style random delete + reinsert same IDs
	// - UPDATE=random_new: delete random active IDs and insert next unseen IDs
	// - KLIST=100: run only Recall@100 instead of the default Wolverine-mode k=10
	// - VARIANT=default: eager delete/insert maintenance
	// - VARIANT=hnsw_hybrid: DAPG search augmented with an auxiliary HNSW candidate layer
#if 0
	// Disabled experimental maintenance variants.
	// - VARIANT=lazy: mark deletes inactive and skip immediate graph repair
	// - VARIANT=batch: lazy deletes plus one cleanup/prune pass after the update block
	// - VARIANT=hnsw_lazy: HNSW candidate layer plus lazy DAPG-side deletion
	// - VARIANT=hnsw_batch: HNSW candidate layer plus batch DAPG-side cleanup
	// - DELETE_P=<0..1>: percentile used by eager delete repair
	//
	// To re-enable, restore the parser for DELETE_P, allow these variants in the
	// validation block, use hnsw candidates for hnsw_lazy/hnsw_batch, and switch
	// deleteNode(id, 0.8f) to lazyDeleteNode(id) for lazy variants.
#endif
	int fifo_init = -1;
	int fifo_block = 1;
	int fifo_shuffle = 0;
	std::string update_mode = "fifo";
	int random_seed = 100;
	std::vector<unsigned> k_override_values;
	std::string dynamic_variant = "default";
	int ef_override = -1;
	int hnsw_hybrid_candidates = -1;
	// Experiment label for the isolated Wolverine-style dynamic runs.
	// Use EXP=... to keep different ablations separate, e.g. EXP=wolv_t80_efc800.
	std::string exp_tag = "exp";
	// Separate output root for new Wolverine-style experiments.
	// Do not use the normal indexes/ CSV files for this mode.
	std::string dynamic_out_dir = "indexes/wolverine_exp";
	for (int ai = 1; ai < argc; ++ai) {
		const char* a = argv[ai];
		if (!a) continue;
		if (std::strncmp(a, "W=", 2) == 0) {
			w_override = (float)std::atof(a + 2);
		}
		else if (std::strncmp(a, "FIFO_INIT=", 10) == 0) {
			fifo_init = std::atoi(a + 10);
		}
		else if (std::strncmp(a, "FIFO_BLOCK=", 11) == 0) {
			fifo_block = std::atoi(a + 11);
		}
		else if (std::strncmp(a, "FIFO_SHUFFLE=", 13) == 0) {
			fifo_shuffle = std::atoi(a + 13);
		}
		else if (std::strncmp(a, "UPDATE=", 7) == 0) {
			update_mode = std::string(a + 7);
		}
		else if (std::strncmp(a, "RANDOM_SEED=", 12) == 0) {
			random_seed = std::atoi(a + 12);
		}
		else if (std::strncmp(a, "KLIST=", 6) == 0) {
			k_override_values.clear();
			std::stringstream ss(std::string(a + 6));
			std::string item;
			while (std::getline(ss, item, ',')) {
				int kv = std::atoi(item.c_str());
				if (kv > 0) k_override_values.push_back((unsigned)kv);
			}
		}
		else if (std::strncmp(a, "VARIANT=", 8) == 0) {
			dynamic_variant = std::string(a + 8);
		}
		else if (std::strncmp(a, "EF=", 3) == 0) {
			ef_override = std::atoi(a + 3);
		}
		else if (std::strncmp(a, "HNSW_CAND=", 10) == 0) {
			hnsw_hybrid_candidates = std::atoi(a + 10);
		}
		else if (std::strncmp(a, "EXP=", 4) == 0) {
			exp_tag = std::string(a + 4);
			for (char& ch : exp_tag) {
				if (!(std::isalnum((unsigned char)ch) || ch == '_' || ch == '-' || ch == '.')) ch = '_';
			}
		}
		else if (std::strncmp(a, "OUTDIR=", 7) == 0) {
			dynamic_out_dir = std::string(a + 7);
			while (!dynamic_out_dir.empty() && (dynamic_out_dir.back() == '/' || dynamic_out_dir.back() == '\\')) {
				dynamic_out_dir.pop_back();
			}
		}
	}

	float c = 1.5;
	std::vector<float> W_values = (w_override > 0.0f)
		? std::vector<float>{w_override}
		: std::vector<float>{0.1f, 0.3f, 0.5f, 1.0f};
	for (float W_run : W_values) {
		// Solmaz Seyed Monir: Cache the table CSV path per W run so we append all k rows into ONE file
		// (prevents repeated headers / schema corruption).
		std::string solmaz_table_out_csv;
		bool solmaz_table_out_csv_set = false;
        // Solmaz: include k=1 to enable direct comparison to papers that report Recall@1 curves (e.g., GATE).
        std::vector<unsigned> k_values = !k_override_values.empty()
			? k_override_values
			: (solmaz_wolverine_mode
				? std::vector<unsigned>{10}
				: std::vector<unsigned>{1, 10, 20, 50, 100});
		unsigned k = 0;
		for (unsigned k_val : k_values) {
			k = k_val;

            unsigned L = 8, K = 10;
			float beta = 0.1;
			unsigned Qnum = 100;
			// Solmaz: Use W from the outer sweep / override (do NOT shadow it).
			float W = W_run;
			int T = 48;
			int efC = 200;
			L = 16;
			K = 24;
			double pC = 0.95, pQ = 0.9;
			std::string datasetName;
			bool isbuilt = 0;
			_lsh_UB=0;
			if (argc > 1) datasetName = argv[1];
			if (argc > 2) isbuilt = std::atoi(argv[2]);
			if (argc > 3) L = std::atoi(argv[3]);
			if (argc > 4) K = std::atoi(argv[4]);
			if (argc > 5) T = std::atoi(argv[5]);
			if (argc > 6) efC = std::atoi(argv[6]);
			if (argc > 7) pC = std::atof(argv[7]);
			if (argc > 8) pQ = std::atof(argv[8]);
			if (argc > 9) _lsh_UB = std::atoi(argv[9]);
			if (argc == 1) {
				const std::string datas[] = { "audio","mnist","cifar","NUS","Trevi","gist","deep1m","skew_10M_8d","gauss_8d","gauss_25w_128" };
				datasetName = datas[0];
				// Solmaz Seyed Monir: Keep legacy dataset-specific W defaults.
				if (w_override <= 0.0f) setW(datasetName, W);
				std::cout << "Using the default configuration!\n\n";
			}

			#if defined(unix) || defined(__unix__)
          
            std::string data_fold = "/mnt/c/Users/Soli1/SIGMOD/Solmaz/LSH-APG/dataset/", index_fold = "./indexes/";
			#else
			std::string data_fold = "/home/solmaz/", index_fold = "./indexes/";
			#endif
			std::cout << "DEBUG: data_fold = " << data_fold << std::endl;

			std::cout << "Using LSH-Graph for " << datasetName << " ..." << std::endl;
			std::cout << "c=        " << c << std::endl;
			std::cout << "k=        " << k << std::endl;
			std::cout << "L=        " << L << std::endl;
			std::cout << "K=        " << K << std::endl;
			std::cout << "T=        " << T << std::endl;
			std::cout << "lsh_UB=   " << _lsh_UB << std::endl;
			Preprocess prep(data_fold + datasetName + ".data", data_fold + "ANN/" + datasetName + ".bench_graph");
			
			if (k > (unsigned)prep.data.N) {
				std::cout << "Solmaz WARNING: skipping k=" << k
					<< " because base N=" << prep.data.N << " is smaller.\n";
				continue;
			}

			showMemoryInfo();

			// Solmaz Seyed Monir
			// Build/load two indexes
			// DAPG index (DAP pruning enabled during construction)
			// LSH-APG baseline index (DAP pruning disabled during construction)
			auto w_tag = [&]() -> std::string {
				// stable filename component for W (e.g., 0.3 -> "0p300")
				std::ostringstream os;
				os << std::fixed << std::setprecision(3) << W_run;
				std::string s = os.str();
				for (char& ch : s) if (ch == '.') ch = 'p';
				return s;
			}();

			std::string path_dapg = index_fold + datasetName + "_W" + w_tag + ".index";
			std::string path_base = index_fold + datasetName + "_W" + w_tag + "_BASE.index";
			if (solmaz_wolverine_mode && fifo_init > 0) {
				// Solmaz NEW: keep Wolverine-style dynamic indexes separate from the
				// original APG/DAPG indexes. Include all construction parameters in
				// the filename so T=48/efC=200 and T=80/efC=800 cannot share a cache.
				std::filesystem::create_directories(dynamic_out_dir);
				const std::string dyn_suffix = "_WOLV_" + exp_tag
					+ "_T" + std::to_string(T)
					+ "_efC" + std::to_string(efC)
					+ "_L" + std::to_string(L)
					+ "_K" + std::to_string(K)
					+ "_FI" + std::to_string(fifo_init)
					+ "_FB" + std::to_string(std::max(fifo_block, 1));
				path_dapg = dynamic_out_dir + "/" + datasetName + "_W" + w_tag + dyn_suffix + ".index";
				path_base = dynamic_out_dir + "/" + datasetName + "_W" + w_tag + dyn_suffix + "_BASE.index";
			}
			Parameter param1(prep, L, K, 1.0f);
			param1.W = W;
			zlsh* gLsh= nullptr;
			divGraph* divG = nullptr;      // DAPG
			divGraph* divG_base = nullptr; // baseline (no DAP pruning)
				if (!GenericTool::CheckPathExistence(index_fold.c_str())) {
					GenericTool::EnsurePathExistence(index_fold.c_str());
				}

			// DAPG index 
			
			if (find_file(path_dapg + "_divGraph")) {
				divG = new divGraph(&prep, path_dapg + "_divGraph", pQ);
				
				// (overriding can silently break search and benchmarking).
				divG->efC = efC;
			}
			else {
				if (fifo_init > 0) {
					divG = new divGraph(prep, param1, path_dapg + "_divGraph", T, efC,
						/*use_dap_pruning=*/true, /*initial_active_N=*/fifo_init, /*shuffle_insert=*/(fifo_shuffle != 0), pC, pQ);
				}
				else {
					divG = new divGraph(prep, param1, path_dapg + "_divGraph", T, efC, /*use_dap_pruning=*/true, pC, pQ);
				}
			}

			// --- Baseline index (no DAP pruning during construction) ---
			if (!solmaz_wolverine_mode) {
				if (find_file(path_base + "_divGraph")) {
					divG_base = new divGraph(&prep, path_base + "_divGraph", pQ);
					// Solmaz: Do NOT override L/K after loading; they are part of the serialized index.
					divG_base->efC = efC;
				}
				else {
					if (fifo_init > 0) {
						divG_base = new divGraph(prep, param1, path_base + "_divGraph", T, efC,
							/*use_dap_pruning=*/false, /*initial_active_N=*/fifo_init, /*shuffle_insert=*/(fifo_shuffle != 0), pC, pQ);
					}
					else {
						divG_base = new divGraph(prep, param1, path_base + "_divGraph", T, efC, /*use_dap_pruning=*/false, pC, pQ);
					}
				}
			}

			if (solmaz_wolverine_mode) {
				if (fifo_init <= 0) {
					throw std::runtime_error("Solmaz ERROR: solmaz_wolverine requires FIFO_INIT=<initial active points>.");
				}
				if (update_mode != "fifo" && update_mode != "random_reinsert" && update_mode != "random_new") {
					throw std::runtime_error("Solmaz ERROR: UPDATE must be fifo, random_reinsert, or random_new.");
				}
				if (dynamic_variant != "default" && dynamic_variant != "pseudo" && dynamic_variant != "aware"
					&& dynamic_variant != "hnsw_hybrid" && dynamic_variant != "faiss_flat") {
					throw std::runtime_error("Solmaz ERROR: VARIANT must be default, pseudo, aware, hnsw_hybrid, or faiss_flat.");
				}
				const int Nbase = (int)prep.data.N;
				const int activeN = std::min(std::max(fifo_init, 1), Nbase);
				const int block = std::max(fifo_block, 1);
				int next_id = activeN;
				std::deque<int> active;
				for (int i = 0; i < activeN; ++i) active.push_back(i);
				std::mt19937 update_rng((unsigned)random_seed);

				// Solmaz Seyed Monir 
				// indexes/wolverine_exp  
				
				std::filesystem::create_directories(dynamic_out_dir);
				const std::string out_csv = dynamic_out_dir + "/dapg_wolverine_" + exp_tag
					+ "_" + datasetName
					+ "_k" + std::to_string(k)
					+ "_rounds" + std::to_string(solmaz_update_count)
					+ "_T" + std::to_string(T)
					+ "_efC" + std::to_string(efC)
					+ "_L" + std::to_string(L)
					+ "_K" + std::to_string(K)
					+ "_fi" + std::to_string(activeN)
					+ "_fb" + std::to_string(block)
					+ "_upd" + update_mode
					+ "_var" + dynamic_variant
					+ "_ef" + std::to_string(ef_override > 0 ? ef_override : ((prep.data.N > 500000) ? ((int)k + 2000) : ((int)k + 300)))
					+ ".csv";
				std::ofstream wout(out_csv);
				if (!wout.is_open()) {
					throw std::runtime_error("Solmaz ERROR: could not open Wolverine-style CSV: " + out_csv);
				}
				wout << "round,recall,search_OPS,delete_OPS,insert_OPS,search_ms,delete_s,insert_s\n";

				const int QnumW = 100;
				divG->use_pseudo_search = false;
				divG->use_ligs_collision_search = false;
				divG->use_collision_aware_start = false;
				divG->use_anchor_start = false;
				divG->use_learned_anchor_start = false;
				divG->pseudo_entry_count = (k >= 100) ? 8 : ((k >= 50) ? 4 : 2);
				if (dynamic_variant == "pseudo" || dynamic_variant == "aware") {
					divG->use_pseudo_search = true;
				}
				if (dynamic_variant == "aware") {
					divG->use_collision_aware_start = true;
					if (k <= 10) { divG->aware_entry_count = 1; divG->aware_rerank_topN_by_dist = 128; divG->aware_bucket_cap_per_table = 512; }
					else if (k <= 20) { divG->aware_entry_count = 2; divG->aware_rerank_topN_by_dist = 256; divG->aware_bucket_cap_per_table = 512; }
					else if (k <= 50) { divG->aware_entry_count = 3; divG->aware_rerank_topN_by_dist = 384; divG->aware_bucket_cap_per_table = 1024; }
					else { divG->aware_entry_count = 4; divG->aware_rerank_topN_by_dist = 512; divG->aware_bucket_cap_per_table = 1024; }
				}
				divG->ef = (ef_override > 0) ? ef_override : ((prep.data.N > 500000) ? ((int)k + 2000) : ((int)k + 300));

				std::unique_ptr<hnswlib::L2Space> hybrid_hnsw_space;
				std::unique_ptr<hnswlib::HierarchicalNSW<float>> hybrid_hnsw;
				const int hnsw_cand = (hnsw_hybrid_candidates > 0)
					? hnsw_hybrid_candidates
					: std::max<int>((int)k * 4, 200);
				if (dynamic_variant == "hnsw_hybrid") {
					hybrid_hnsw_space.reset(new hnswlib::L2Space((size_t)prep.data.dim));
					hybrid_hnsw.reset(new hnswlib::HierarchicalNSW<float>(
						hybrid_hnsw_space.get(),
						(size_t)prep.data.N + 1,
						/*M=*/32,
						/*ef_construction=*/std::max(200, efC),
						/*random_seed=*/100,
						/*allow_replace_deleted=*/true));
					for (int id : active) {
						if (id >= 0 && id < (int)prep.data.N) {
							hybrid_hnsw->addPoint((const void*)prep.data.val[id], (hnswlib::labeltype)id);
						}
					}
					std::cout << "Solmaz: built HNSW hybrid candidate layer activeN=" << activeN
						<< " hnsw_cand=" << hnsw_cand << "\n";
				}

				for (int round = 0; round < solmaz_update_count; ++round) {
					SearchResult sr = (dynamic_variant == "hnsw_hybrid")
						? solmaz_graphSearch_active_hnsw_hybrid_no_write(c, (int)k, divG, hybrid_hnsw.get(), prep, beta, /*qType=*/0, QnumW, active, divG->ef, hnsw_cand)
						: ((dynamic_variant == "faiss_flat")
							? solmaz_graphSearch_active_faiss_flat_no_write((int)k, prep, QnumW, active, hnsw_cand)
							: solmaz_graphSearch_active_no_write(c, (int)k, divG, prep, beta, /*qType=*/0, QnumW, active));
					const double search_ops = (sr.time > 0.0f) ? (1000.0 / (double)sr.time) : 0.0;

					std::vector<int> del_ids;
					std::vector<int> ins_ids;
					del_ids.reserve((size_t)block);
					ins_ids.reserve((size_t)block);
					if (update_mode == "random_reinsert") {
						const int draw_n = std::min(block, (int)active.size());
						std::uniform_int_distribution<int> draw_pos(0, (int)active.size() - 1);
						std::unordered_set<int> chosen;
						chosen.reserve((size_t)draw_n * 2);
						while ((int)chosen.size() < draw_n) {
							const int id = active[(size_t)draw_pos(update_rng)];
							if (chosen.insert(id).second) {
								del_ids.push_back(id);
								ins_ids.push_back(id);
							}
						}
					}
					else if (update_mode == "random_new") {
						const int draw_n = std::min(block, std::min((int)active.size(), Nbase - next_id));
						if (draw_n > 0) {
							std::uniform_int_distribution<int> draw_pos(0, (int)active.size() - 1);
							std::unordered_set<int> chosen_pos;
							chosen_pos.reserve((size_t)draw_n * 2);
							while ((int)chosen_pos.size() < draw_n) {
								const int pos = draw_pos(update_rng);
								if (!chosen_pos.insert(pos).second) continue;
								const int did = active[(size_t)pos];
								const int iid = next_id++;
								active[(size_t)pos] = iid;
								del_ids.push_back(did);
								ins_ids.push_back(iid);
							}
						}
					}
					else {
						for (int b = 0; b < block && !active.empty() && next_id < Nbase; ++b) {
							int did = active.front();
							active.pop_front();
							int iid = next_id++;
							active.push_back(iid);
							del_ids.push_back(did);
							ins_ids.push_back(iid);
						}
					}

					lsh::timer tdel;
					tdel.restart();
					for (int id : del_ids) {
						divG->deleteNode(id, 0.8f);
						if (hybrid_hnsw) {
							try { hybrid_hnsw->markDelete((hnswlib::labeltype)id); }
							catch (...) {}
						}
					}
					const double delete_s = tdel.elapsed();
					const double delete_ops = (delete_s > 0.0 && !del_ids.empty()) ? ((double)del_ids.size() / delete_s) : 0.0;

					lsh::timer tins;
					tins.restart();
					for (int id : ins_ids) {
						divG->insertNode(id);
						if (hybrid_hnsw) {
							if (update_mode == "random_reinsert") {
								try { hybrid_hnsw->unmarkDelete((hnswlib::labeltype)id); }
								catch (...) {}
							}
							else {
								hybrid_hnsw->addPoint((const void*)prep.data.val[id], (hnswlib::labeltype)id);
							}
						}
					}
					const double insert_s = tins.elapsed();
					const double insert_ops = (insert_s > 0.0 && !ins_ids.empty()) ? ((double)ins_ids.size() / insert_s) : 0.0;

					wout << round << ","
						<< sr.recall << ","
						<< search_ops << ","
						<< delete_ops << ","
						<< insert_ops << ","
						<< sr.time << ","
						<< delete_s << ","
						<< insert_s << "\n";

					std::cout << "Solmaz Wolverine-style round=" << round
						<< " update=" << update_mode
						<< " variant=" << dynamic_variant
						<< " ef=" << divG->ef
						<< " recall=" << sr.recall
						<< " search_OPS=" << search_ops
						<< " delete_OPS=" << delete_ops
						<< " insert_OPS=" << insert_ops
						<< std::endl;
					if ((update_mode == "fifo" || update_mode == "random_new") && ((int)del_ids.size() < block || (int)ins_ids.size() < block)) break;
				}
				wout.close();
				std::cout << "Solmaz: wrote Wolverine-style DAPG CSV to " << out_csv << std::endl;
				return 0;
			}

            // Compare original vs pseudo-graph search using the existing benchmark (graphSearch).
            // graphSearch() computes average recall/time/cost over Qnum queries and writes results to files.
            std::filesystem::create_directories("indexes");

            // Use one ef setting for quick A/B comparison (adjust if you want a sweep).
            divG->ef = (int)k + 150;

            // Name includes the learned threshold (from your DAP pruning in insertLSHRefine).
            std::ostringstream algNameBaseStream;
            algNameBaseStream << "DAP_k" << k << "_th" << std::fixed << std::setprecision(3) << divG->last_threshold;
            std::string algNameBase = algNameBaseStream.str();

            // Print a clear header so it's obvious what each result column means.
            std::cout << std::setw(_lspace) << "algName"
				<< std::setw(_sspace) << "k"
				<< std::setw(_sspace) << "ef"
                      << std::setw(_lspace) << "Time(ms)"
				<< std::setw(_lspace) << "Recall"
				<< std::setw(_lspace) << "Cost"
				<< std::setw(_lspace) << "CPQ1"
				<< std::setw(_lspace) << "CPQ2"
				<< std::setw(_lspace) << "Pruning"
				<< std::endl;

           
            // DAPG: graph traversal on the DAPG-built graph (Alg-6 updates + DAP pruning)
            // PSEUDO_DAPG: neighbor expansion without hash pruning on the same DAPG graph
            // LIGA_COL: collision-set expansion as an additional baseline
            enum class SolmazSearchMode { ORIG, PSEUDO_NEI, LIGS_COL };
            const std::vector<SolmazSearchMode> modes = {
                SolmazSearchMode::ORIG,
                SolmazSearchMode::PSEUDO_NEI,
                SolmazSearchMode::LIGS_COL
            };
            for (auto mode : modes) {
                divG->use_pseudo_search = false;
                divG->use_ligs_collision_search = false;

                std::string suffix;
                if (mode == SolmazSearchMode::ORIG) {
                    suffix = "_DAPG";
                } else if (mode == SolmazSearchMode::PSEUDO_NEI) {
                    suffix = "_PSEUDO_DAPG";
                    divG->use_pseudo_search = true;
                } else {
                    suffix = "_LIGA_COL";
                    divG->use_ligs_collision_search = true;
                }

                std::string algName = algNameBase + suffix;
                std::cout << "\n=== " << algName << " (ef=" << divG->ef << ") ===\n";
                graphSearch(c, (int)k, divG, prep, beta, datasetName, data_fold, /*qType=*/0, algName);
            }

            // Solmaz Seyed Monir
			//benchmark the original LSH-APG baseline graph (same LSH params, but without DAP local pruning during construction).
          
            if (divG_base) {
                divG_base->ef = divG->ef; // keep same ef for comparable runtime
                divG_base->use_pseudo_search = false;
                divG_base->use_ligs_collision_search = false;
                std::ostringstream baseName;
                baseName << "LSHAPG_BASE_k" << k;
                std::string algName = baseName.str();
                std::cout << "\n=== " << algName << " (ef=" << divG_base->ef << ") ===\n";
                graphSearch(c, (int)k, divG_base, prep, beta, datasetName, data_fold, /*qType=*/0, algName);
            }

			
			if (solmaz_table_mode || solmaz_curve_mode || solmaz_ncurve_mode) {
				// Solmaz: Include higher-recall targets to make curves comparable with NSG/GATE (which often operate at 0.99+).
				// Avoid 1.0 (often unattainable/noisy); use 0.99/0.995/0.999 instead.
				const std::vector<float> recall_targets = { 0.8f, 0.9f, 0.95f, 0.97f, 0.99f, 0.995f, 0.999f };
				const int QnumTable = 100; // match alg.h graphSearch() default

				// Sweep ef to find minimal time achieving each recall target.
				std::vector<int> ef_sweep;
				// Solmaz: For large datasets (e.g., SIFT ~1M), ef=k+300 is often too small to reach high recall.
				// Use a larger default sweep, but stop early once all targets are met.
				int ef_max = (prep.data.N > 500000) ? ((int)k + 2000) : ((int)k + 300);
				for (int efv = (int)k; efv <= ef_max; efv += 20) ef_sweep.push_back(efv);

				const std::string expected_header =
					"SolmazDataset,SolmazMethod,k,T,efC,UpdateCount,IndexingTime_s,InsertAvg_ms,DeleteAvg_ms,"
					"SearchRt0p8_ms,SearchRt0p9_ms,SearchRt0p95_ms,SearchRt0p97_ms,SearchRt0p99_ms,SearchRt0p995_ms,SearchRt0p999_ms,"
					"MaxRecall,MaxRecallEf";

				
				if (!solmaz_table_out_csv_set) {
					const std::string stem = "indexes/solmaz_table_" + datasetName + "_W" + w_tag;
					const std::string base_csv = stem + ".csv";

					auto read_first_line = [&](const std::string& path) -> std::string {
						std::ifstream in(path);
						std::string line;
						std::getline(in, line);
						return line;
					};
					auto versioned = [&](int v) -> std::string {
						return stem + "_v" + std::to_string(v) + ".csv";
					};

					// Find the latest existing CSV that matches the current header.
					int latest_matching_v = 0; // 0 means none
					const bool base_exists = std::filesystem::exists(base_csv);
					const bool base_matches = base_exists && (read_first_line(base_csv) == expected_header);
					if (base_matches) {
						latest_matching_v = 1; // treat base as v1
					}
					for (int v = 2; v <= 99; ++v) {
						const std::string p = versioned(v);
						if (!std::filesystem::exists(p)) break;
						const std::string h = read_first_line(p);
						if (!h.empty() && h == expected_header) latest_matching_v = v;
					}

					
					if (!base_exists) {
						// Base doesn't exist: start with base.
						solmaz_table_out_csv = base_csv;
					} else if (!base_matches && latest_matching_v <= 0) {
						// Base exists but header mismatches: NEVER append. Create a fresh v2/v3/...
						int v = 2;
						while (std::filesystem::exists(versioned(v))) ++v;
						solmaz_table_out_csv = versioned(v);
					} else if (latest_matching_v == 1) {
						// Base exists and matches: write to v2.
						solmaz_table_out_csv = versioned(2);
					} else {
						// Latest matching is vN: write to v(N+1).
						solmaz_table_out_csv = versioned(latest_matching_v + 1);
					}

					solmaz_table_out_csv_set = true;
				}

				const std::string out_csv = solmaz_table_out_csv;
				const bool write_header = !std::filesystem::exists(out_csv);
				std::ofstream out(out_csv, std::ios::app);
				if (!out.is_open()) {
					throw std::runtime_error("Solmaz ERROR: could not open CSV for writing: " + out_csv);
				}
				if (write_header) {
					out << expected_header << "\n";
				}

				// Solmaz Seyed Monir 
				// Update schedule
				// Default: random unique IDs (legacy)
				// FIFO mode:  sliding window (insert new, delete oldest), controlled by FIFO_INIT/FIFO_BLOCK.
				std::vector<int> del_ids;
				std::vector<int> ins_ids;
				int solmaz_effective_update_ops = 0;
				if (fifo_init > 0) {
					int Nbase = (int)prep.data.N;
					int initN = std::min(std::max(fifo_init, 1), Nbase);
					int block = std::max(fifo_block, 1);

					std::deque<int> active;
					active.resize(initN);
					for (int i = 0; i < initN; ++i) active[i] = i;
					int next_id = initN;

					del_ids.reserve((size_t)solmaz_update_count * (size_t)block);
					ins_ids.reserve((size_t)solmaz_update_count * (size_t)block);
					for (int t = 0; t < solmaz_update_count; ++t) {
						for (int b = 0; b < block; ++b) {
							if (active.empty()) break;
							if (next_id >= Nbase) break;
							int did = active.front(); active.pop_front();
							int iid = next_id++;
							active.push_back(iid);
							del_ids.push_back(did);
							ins_ids.push_back(iid);
						}
						if (next_id >= Nbase) break;
					}
					solmaz_effective_update_ops = (int)del_ids.size();
			}

			
			
			if (solmaz_curve_mode) {
				const int QnumCurve = 100; // keep consistent with solmaz_table defaults (and alg.h graphSearch default)

				// ef sweep (same heuristic as table mode)
				std::vector<int> ef_sweep;
				int ef_max = (prep.data.N > 500000) ? ((int)k + 2000) : ((int)k + 300);
				for (int efv = (int)k; efv <= ef_max; efv += 20) ef_sweep.push_back(efv);

				const std::string curve_header =
					"SolmazDataset,SolmazMethod,k,T,efC,W,UpdateCount,ef,Recall,QPS,Time_ms";

				
				static std::string solmaz_curve_out_csv;
				static bool solmaz_curve_out_csv_set = false;
				if (!solmaz_curve_out_csv_set) {
					const std::string stem = "indexes/solmaz_curve_" + datasetName + "_W" + w_tag;
					const std::string base_csv = stem + ".csv";
					auto read_first_line = [&](const std::string& path) -> std::string {
						std::ifstream in(path);
						std::string line;
						std::getline(in, line);
						return line;
					};
					auto versioned = [&](int v) -> std::string {
						return stem + "_v" + std::to_string(v) + ".csv";
					};

					int latest_matching_v = 0;
					const bool base_exists = std::filesystem::exists(base_csv);
					const bool base_matches = base_exists && (read_first_line(base_csv) == curve_header);
					if (base_matches) latest_matching_v = 1;
					for (int v = 2; v <= 99; ++v) {
						const std::string p = versioned(v);
						if (!std::filesystem::exists(p)) break;
						const std::string h = read_first_line(p);
						if (!h.empty() && h == curve_header) latest_matching_v = v;
					}
					if (!base_exists) solmaz_curve_out_csv = base_csv;
					else if (!base_matches && latest_matching_v <= 0) {
						int v = 2;
						while (std::filesystem::exists(versioned(v))) ++v;
						solmaz_curve_out_csv = versioned(v);
					}
					else if (latest_matching_v == 1) solmaz_curve_out_csv = versioned(2);
					else solmaz_curve_out_csv = versioned(latest_matching_v + 1);

					solmaz_curve_out_csv_set = true;
				}

				const std::string out_csv = solmaz_curve_out_csv;
				const bool write_header = !std::filesystem::exists(out_csv);
				std::ofstream out(out_csv, std::ios::app);
				if (!out.is_open()) {
					throw std::runtime_error("Solmaz ERROR: could not open curve CSV for writing: " + out_csv);
				}
				if (write_header) out << curve_header << "\n";

			
				// (delete/insert), so search is measured on the maintained index state.

			
				auto write_curve_row = [&](const std::string& method, int efv, const SearchResult& r) {
					const double qps = (r.time > 0.0f) ? (1000.0 / (double)r.time) : 0.0;
					out << datasetName << ","
						<< method << ","
						<< k << ","
						<< divG->T << ","
						<< efC << ","
						<< w_tag << ","
						<< solmaz_update_count << ","
						<< efv << ","
						<< r.recall << ","
						<< qps << ","
						<< r.time << "\n";
				};

				
				enum class SolmazSearchMode { ORIG, PSEUDO_NEI, LIGS_COL };
				const std::vector<SolmazSearchMode> modes = {
					SolmazSearchMode::ORIG,
					SolmazSearchMode::PSEUDO_NEI,
					SolmazSearchMode::LIGS_COL
				};
				for (auto mode : modes) {
					divG->use_pseudo_search = false;
					divG->use_ligs_collision_search = false;
					divG->use_collision_aware_start = false;
					divG->use_anchor_start = false;
					divG->use_learned_anchor_start = false;
					divG->pseudo_entry_count = 1;

					std::string method;
					if (mode == SolmazSearchMode::ORIG) method = "DAPG";
					else if (mode == SolmazSearchMode::PSEUDO_NEI) {
						method = "PSEUDO_DAPG";
						divG->use_pseudo_search = true;
						// Multi-start for pseudo traversal (beam start).
						divG->pseudo_entry_count = (k >= 100) ? 8 : ((k >= 50) ? 4 : 2);
					}
					else { method = "LIGA_COL"; divG->use_ligs_collision_search = true; }

					const bool run_collision_aware_variant = (mode == SolmazSearchMode::PSEUDO_NEI);
					const bool run_anchor_variant = (mode == SolmazSearchMode::PSEUDO_NEI);
					const bool run_learned_anchor_variant = (mode == SolmazSearchMode::PSEUDO_NEI);

					const std::string method_orig = method;
					const std::string method_aware = "AWARE_" + method_orig;
					const std::string method_anchor = "ANCHOR_" + method_orig;
					const std::string method_learned_anchor = "LEARNED_ANCHOR_" + method_orig;

					const int variant_loops = (run_collision_aware_variant || run_anchor_variant || run_learned_anchor_variant) ? 4 : 1;
					for (int variant = 0; variant < variant_loops; ++variant) {
						method = method_orig;
						divG->use_collision_aware_start = false;
						divG->use_anchor_start = false;
						divG->use_learned_anchor_start = false;

						if (variant == 1 && run_collision_aware_variant) {
							method = method_aware;
							divG->use_collision_aware_start = true;
							// Reuse the same per-k tuning already used in solmaz_table.
							if (k <= 10) { divG->aware_entry_count = 1; divG->aware_rerank_topN_by_dist = 128; divG->aware_bucket_cap_per_table = 512; }
							else if (k <= 20) { divG->aware_entry_count = 2; divG->aware_rerank_topN_by_dist = 256; divG->aware_bucket_cap_per_table = 512; }
							else if (k <= 50) { divG->aware_entry_count = 3; divG->aware_rerank_topN_by_dist = 384; divG->aware_bucket_cap_per_table = 1024; }
							else { divG->aware_entry_count = 4; divG->aware_rerank_topN_by_dist = 512; divG->aware_bucket_cap_per_table = 1024; }
						}
						else if (variant == 2 && run_anchor_variant) {
							method = method_anchor;
							const std::string anchor_path = "indexes/anchors_" + datasetName + ".txt";
							if (!std::filesystem::exists(anchor_path) || !divG->load_anchor_ids(anchor_path)) continue;
							divG->use_anchor_start = true;
							// Stronger warm-start for hard/high-recall settings.
							divG->anchor_warm_start = true;
							divG->anchor_local_refine = true;
							if (k >= 100) { divG->anchor_warmup_expansions = 128; divG->anchor_warmup_degree_cap = 64; }
							else { divG->anchor_warmup_expansions = 64; divG->anchor_warmup_degree_cap = 32; }
						}
						else if (variant == 3 && run_learned_anchor_variant) {
							method = method_learned_anchor;
							const std::string anchor_path = "indexes/anchors_" + datasetName + ".txt";
							const std::string model_path = "indexes/anchors_" + datasetName + "_softmax.bin";
							if (!std::filesystem::exists(anchor_path) || !divG->load_anchor_ids(anchor_path)) continue;
							if (!std::filesystem::exists(model_path) || !divG->load_learned_anchor_model(model_path)) continue;
							divG->use_learned_anchor_start = true;
							// Stronger warm-start for hard/high-recall settings.
							divG->anchor_warm_start = true;
							divG->anchor_local_refine = true;
							if (k >= 100) { divG->anchor_warmup_expansions = 128; divG->anchor_warmup_degree_cap = 64; }
							else { divG->anchor_warmup_expansions = 64; divG->anchor_warmup_degree_cap = 32; }
						}

						for (int efv : ef_sweep) {
							divG->ef = efv;
							SearchResult r = solmaz_graphSearch_no_write(c, (int)k, divG, prep, beta, /*qType=*/0, QnumCurve);
							write_curve_row(method, efv, r);
						}
					}
				}

				// log the LSH-APG baseline construction
				if (divG_base) {
					divG_base->use_pseudo_search = false;
					divG_base->use_ligs_collision_search = false;
					divG_base->use_collision_aware_start = false;
					divG_base->use_anchor_start = false;
					divG_base->use_learned_anchor_start = false;
					for (int efv : ef_sweep) {
						divG_base->ef = efv;
						SearchResult r = solmaz_graphSearch_no_write(c, (int)k, divG_base, prep, beta, /*qType=*/0, QnumCurve);
						
						const double qps = (r.time > 0.0f) ? (1000.0 / (double)r.time) : 0.0;
						out << datasetName << ","
							<< "LSHAPG_BASE" << ","
							<< k << ","
							<< divG_base->T << ","
							<< efC << ","
							<< w_tag << ","
							<< solmaz_update_count << ","
							<< efv << ","
							<< r.recall << ","
							<< qps << ","
							<< r.time << "\n";
					}
				}

				// HNSW baseline: build once per dataset (per process) and sweep ef_search.
				
				static std::string hnsw_built_for_dataset;
				static std::unique_ptr<hnswlib::L2Space> hnsw_space;
				static std::unique_ptr<hnswlib::HierarchicalNSW<float>> hnsw_index;
				if (hnsw_built_for_dataset != datasetName) {
					try {
						hnsw_space.reset(new hnswlib::L2Space((size_t)prep.data.dim));
						const size_t max_elements = (size_t)prep.data.N;
						const size_t M = 16;
						const size_t efC_hnsw = 200;
						hnsw_index.reset(new hnswlib::HierarchicalNSW<float>(hnsw_space.get(), max_elements, M, efC_hnsw));
						for (size_t i = 0; i < max_elements; ++i) {
							hnsw_index->addPoint((const void*)prep.data.val[i], (hnswlib::labeltype)i);
						}
						hnsw_built_for_dataset = datasetName;
						std::cout << "Built HNSW baseline: N=" << prep.data.N << " dim=" << prep.data.dim << "\n";
					}
					catch (const std::exception& e) {
						std::cerr << "HNSW build failed: " << e.what() << "\n";
						hnsw_index.reset();
					}
				}

				if (hnsw_index) {
					for (int efv : ef_sweep) {
						SearchResult r = solmaz_hnsw_no_write((int)k, hnsw_index.get(), prep, /*ef_search=*/efv, QnumCurve);
						write_curve_row("HNSW", efv, r);
					}
				}
			}
			
			// measure Recall/QPS 
			else if (solmaz_ncurve_mode) {
				const int QnumCurve = 100;
				const int ef_fixed = (prep.data.N > 500000) ? ((int)k + 2000) : ((int)k + 300);
				divG->ef = ef_fixed;
				if (divG_base) divG_base->ef = ef_fixed;

				const std::vector<double> n_fracs = {0.2, 0.4, 0.6, 0.8, 1.0};
				const double baseUB = 4.0 * (double)L * std::log((double)prep.data.N + 1.0);

				const std::string ncurve_header =
					"SolmazDataset,SolmazMethod,k,T,efC,W,UpdateCount,n_frac,n_abs,ef,Recall,QPS,Time_ms";

				static std::string solmaz_ncurve_out_csv;
				static bool solmaz_ncurve_out_csv_set = false;
				if (!solmaz_ncurve_out_csv_set) {
					const std::string stem = "indexes/solmaz_ncurve_" + datasetName + "_W" + w_tag;
					const std::string base_csv = stem + ".csv";
					auto read_first_line = [&](const std::string& path) -> std::string {
						std::ifstream in(path);
						std::string line;
						std::getline(in, line);
						return line;
					};
					auto versioned = [&](int v) -> std::string {
						return stem + "_v" + std::to_string(v) + ".csv";
					};
					int latest_matching_v = 0;
					const bool base_exists = std::filesystem::exists(base_csv);
					const bool base_matches = base_exists && (read_first_line(base_csv) == ncurve_header);
					if (base_matches) latest_matching_v = 1;
					for (int v = 2; v <= 99; ++v) {
						const std::string p = versioned(v);
						if (!std::filesystem::exists(p)) break;
						const std::string h = read_first_line(p);
						if (!h.empty() && h == ncurve_header) latest_matching_v = v;
					}
					if (!base_exists) solmaz_ncurve_out_csv = base_csv;
					else if (!base_matches && latest_matching_v <= 0) {
						int v = 2;
						while (std::filesystem::exists(versioned(v))) ++v;
						solmaz_ncurve_out_csv = versioned(v);
					}
					else if (latest_matching_v == 1) solmaz_ncurve_out_csv = versioned(2);
					else solmaz_ncurve_out_csv = versioned(latest_matching_v + 1);
					solmaz_ncurve_out_csv_set = true;
				}

				const std::string out_csv = solmaz_ncurve_out_csv;
				const bool write_header = !std::filesystem::exists(out_csv);
				std::ofstream out(out_csv, std::ios::app);
				if (!out.is_open()) throw std::runtime_error("Solmaz ERROR: could not open ncurve CSV: " + out_csv);
				if (write_header) out << ncurve_header << "\n";

				auto run_one = [&](divGraph* g, const std::string& method, double n_frac, int n_abs) {
					_lsh_UB = n_abs;
					SearchResult r = solmaz_graphSearch_no_write(c, (int)k, g, prep, beta, /*qType=*/0, QnumCurve);
					const double qps = (r.time > 0.0f) ? (1000.0 / (double)r.time) : 0.0;
					out << datasetName << "," << method << "," << k << ","
						<< g->T << "," << efC << "," << w_tag << ","
						<< solmaz_update_count << ","
						<< n_frac << "," << n_abs << ","
						<< g->ef << ","
						<< r.recall << "," << qps << "," << r.time << "\n";
				};

				// DAPG and PSEUDO_DAPG under varying n.
				for (double f : n_fracs) {
					const int n_abs = std::max(1, (int)std::llround(f * baseUB));

					divG->use_pseudo_search = false;
					divG->use_ligs_collision_search = false;
					divG->use_collision_aware_start = false;
					divG->use_anchor_start = false;
					divG->use_learned_anchor_start = false;
					run_one(divG, "DAPG", f, n_abs);

					divG->use_pseudo_search = true;
					divG->use_ligs_collision_search = false;
					divG->use_collision_aware_start = false;
					divG->use_anchor_start = false;
					divG->use_learned_anchor_start = false;
					run_one(divG, "PSEUDO_DAPG", f, n_abs);
				}
				// LSH-APG construction baseline (no DAP pruning) if present.
				if (divG_base) {
					for (double f : n_fracs) {
						const int n_abs = std::max(1, (int)std::llround(f * baseUB));
						divG_base->use_pseudo_search = false;
						divG_base->use_ligs_collision_search = false;
						divG_base->use_collision_aware_start = false;
						divG_base->use_anchor_start = false;
						divG_base->use_learned_anchor_start = false;
						run_one(divG_base, "LSHAPG_BASE", f, n_abs);
					}
				}

				std::cout << "Solmaz: wrote ncurve rows to " << out_csv << std::endl;
			}
			else {
					std::mt19937 rng(42);
					std::uniform_int_distribution<int> dist_id(0, prep.data.N - 1);
					std::unordered_set<int> upd_seen;
					upd_seen.reserve((size_t)solmaz_update_count * 2);
					del_ids.reserve(solmaz_update_count);
					ins_ids.reserve(solmaz_update_count);
					while ((int)del_ids.size() < solmaz_update_count) {
						int id = dist_id(rng);
						if (id == divG->first_id) continue;
						if (upd_seen.insert(id).second) {
							del_ids.push_back(id);
							ins_ids.push_back(id); // re-insert same IDs
						}
					}
					solmaz_effective_update_ops = (int)del_ids.size();
				}

				enum class SolmazSearchMode { ORIG, PSEUDO_NEI, LIGS_COL };
				const std::vector<SolmazSearchMode> modes = {
					SolmazSearchMode::ORIG,
					SolmazSearchMode::PSEUDO_NEI,
					SolmazSearchMode::LIGS_COL
				};
				for (auto mode : modes) {
					divG->use_pseudo_search = false;
					divG->use_ligs_collision_search = false;
					divG->use_collision_aware_start = false;
				divG->use_anchor_start = false;
					divG->use_learned_anchor_start = false;
					divG->pseudo_entry_count = 1;

					std::string method;
					if (mode == SolmazSearchMode::ORIG) {
						method = "DAPG";
					} else if (mode == SolmazSearchMode::PSEUDO_NEI) {
						method = "PSEUDO_DAPG";
						divG->use_pseudo_search = true;
						// Multi-start for pseudo traversal (beam start).
						// Heuristic: use more starts for larger k (harder, higher recall).
						divG->pseudo_entry_count = (k >= 100) ? 8 : ((k >= 50) ? 4 : 2);
					} else {
						method = "LIGA_COL";
						divG->use_ligs_collision_search = true;
					}

					// a collision-aware start variant (best effort) on top of the strongest traversal.
					// enable it for the pseudo traversal to isolate the entry-point effect cleanly.
					const bool run_collision_aware_variant = (mode == SolmazSearchMode::PSEUDO_NEI);
					const std::string method_orig = method;
					const std::string method_aware = "AWARE_" + method_orig;
					// Solmaz (Idea1A): Anchor-based entry points (no learning), also only for pseudo traversal.
					const bool run_anchor_variant = (mode == SolmazSearchMode::PSEUDO_NEI);
					const std::string method_anchor = "ANCHOR_" + method_orig;
					// Solmaz (Idea1B-B1): Learned anchors (predict nearest anchor label), also only for pseudo traversal.
					const bool run_learned_anchor_variant = (mode == SolmazSearchMode::PSEUDO_NEI);
					const std::string method_learned_anchor = "LEARNED_ANCHOR_" + method_orig;
					const int variant_loops = (run_collision_aware_variant || run_anchor_variant || run_learned_anchor_variant) ? 4 : 1;
					for (int variant = 0; variant < variant_loops; ++variant) {
						method = method_orig;
						divG->use_collision_aware_start = false;
						divG->use_anchor_start = false;
						divG->use_learned_anchor_start = false;

						if (variant == 1 && run_collision_aware_variant) {
							method = method_aware;
							divG->use_collision_aware_start = true;

							// Tune AWARE parameters per-k (entry policy only).
							// Heuristic defaults (can be adjusted):
							// - k small: single-entry start (avoid overhead)
							// - k large: small multi-entry start helps reach high recall faster
							if (k <= 10) {
								divG->aware_entry_count = 1;
								divG->aware_rerank_topN_by_dist = 128;
								divG->aware_bucket_cap_per_table = 512;
							}
							else if (k <= 20) {
								divG->aware_entry_count = 2;
								divG->aware_rerank_topN_by_dist = 256;
								divG->aware_bucket_cap_per_table = 512;
							}
							else if (k <= 50) {
								divG->aware_entry_count = 3;
								divG->aware_rerank_topN_by_dist = 384;
								divG->aware_bucket_cap_per_table = 1024;
							}
							else { // k >= 100
								divG->aware_entry_count = 4;
								divG->aware_rerank_topN_by_dist = 512;
								divG->aware_bucket_cap_per_table = 1024;
							}

							// Hybrid-aware entry scoring
							// collisions vs distance-rank tradeoff
							
							divG->aware_alpha = 1.0f;
							divG->aware_beta = (k >= 100) ? 0.10f : 0.15f;
							// Keep hard filter disabled by default (hybrid scoring is smoother).
							divG->aware_min_collisions = 0;
						}
						else if (variant == 2 && run_anchor_variant) {
							method = method_anchor;
							// Load anchors (if present); otherwise skip this variant.
							const std::string anchor_path = "indexes/anchors_" + datasetName + ".txt";
							if (!std::filesystem::exists(anchor_path) || !divG->load_anchor_ids(anchor_path)) {
								std::cout << "Solmaz WARNING: missing/empty anchors file '" << anchor_path
									<< "'; skipping " << method_anchor << std::endl;
								continue;
							}
							divG->use_anchor_start = true;
							// Stronger warm-start for hard/high-recall settings.
							divG->anchor_warm_start = true;
							divG->anchor_local_refine = true;
							if (k >= 100) {
								divG->anchor_warmup_expansions = 128;
								divG->anchor_warmup_degree_cap = 64;
							} else {
								divG->anchor_warmup_expansions = 64;
								divG->anchor_warmup_degree_cap = 32;
							}
						}
						else if (variant == 3 && run_learned_anchor_variant) {
							method = method_learned_anchor;
							const std::string anchor_path = "indexes/anchors_" + datasetName + ".txt";
							const std::string model_path = "indexes/anchors_" + datasetName + "_softmax.bin";
							if (!std::filesystem::exists(anchor_path) || !divG->load_anchor_ids(anchor_path)) {
								std::cout << "Solmaz WARNING: missing/empty anchors file '" << anchor_path
									<< "'; skipping " << method_learned_anchor << std::endl;
								continue;
							}
							if (!std::filesystem::exists(model_path) || !divG->load_learned_anchor_model(model_path)) {
								std::cout << "Solmaz WARNING: missing/invalid learned-anchor model '" << model_path
									<< "'; skipping " << method_learned_anchor << std::endl;
								continue;
							}
							divG->use_learned_anchor_start = true;
							// Stronger warm-start for hard/high-recall settings.
							divG->anchor_warm_start = true;
							divG->anchor_local_refine = true;
							if (k >= 100) {
								divG->anchor_warmup_expansions = 128;
								divG->anchor_warmup_degree_cap = 64;
							} else {
								divG->anchor_warmup_expansions = 64;
								divG->anchor_warmup_degree_cap = 32;
							}
						}

					//Delete timing 
					double delete_ms = -1.0;
					int solmaz_cur_delete_id = -1;
					try {
						lsh::timer tdel;
						tdel.restart();
						for (int id : del_ids) {
							solmaz_cur_delete_id = id;
							divG->deleteNode(id, 0.8f);
						}
						delete_ms = del_ids.empty() ? -1.0 : (tdel.elapsed() * 1000.0) / (double)del_ids.size();
					}
					catch (const std::system_error& e) {
						std::cerr << "Solmaz ERROR: deleteNode threw system_error for method=" << method
							<< " id=" << solmaz_cur_delete_id
							<< " what=" << e.what() << std::endl;
						throw;
					}

					//Insert timing (re-activate)
					double insert_ms = -1.0;
					int solmaz_cur_insert_id = -1;
					try {
						lsh::timer tins;
						tins.restart();
						for (int id : ins_ids) {
							solmaz_cur_insert_id = id;
							divG->insertNode(id);
						}
						insert_ms = ins_ids.empty() ? -1.0 : (tins.elapsed() * 1000.0) / (double)ins_ids.size();
					}
					catch (const std::system_error& e) {
						std::cerr << "Solmaz ERROR: insertNode threw system_error for method=" << method
							<< " id=" << solmaz_cur_insert_id
							<< " what=" << e.what() << std::endl;
						throw;
					}

					//  Search timing at recall targets 
					std::vector<double> best_ms(recall_targets.size(), -1.0);
					float max_recall = -1.0f;
					int max_recall_ef = -1;
					for (int efv : ef_sweep) {
						divG->ef = efv;
						SearchResult r = solmaz_graphSearch_no_write(c, (int)k, divG, prep, beta, /*qType=*/0, QnumTable);
						if (r.recall > max_recall) {
							max_recall = r.recall;
							max_recall_ef = efv;
						}
						for (size_t ti = 0; ti < recall_targets.size(); ++ti) {
							if (r.recall >= recall_targets[ti]) {
								if (best_ms[ti] < 0 || r.time < best_ms[ti]) best_ms[ti] = r.time;
							}
						}
						bool all_met = true;
						for (double v : best_ms) {
							if (v < 0) { all_met = false; break; }
						}
						if (all_met) break;
					}
					if (best_ms[0] < 0) {
						std::cout << "Solmaz WARNING: recall targets not met for " << method
							<< " (k=" << k << "). MaxRecall=" << max_recall
							<< " at ef=" << max_recall_ef << std::endl;
					}

					out << datasetName << ","
					    << method << ","
						<< k << ","
					    << divG->T << ","
					    << efC << ","
					    << solmaz_effective_update_ops << ","
						<< (divG->indexingTime > 0 ? divG->indexingTime : -1.0f) << ","
					    << insert_ms << ","
					    << delete_ms << ",";

					
					auto write_ms_or_blank = [&](double v) {
						if (v < 0) out << "";
						else out << v;
					};
					write_ms_or_blank(best_ms.size() > 0 ? best_ms[0] : -1.0); out << ",";
					write_ms_or_blank(best_ms.size() > 1 ? best_ms[1] : -1.0); out << ",";
					write_ms_or_blank(best_ms.size() > 2 ? best_ms[2] : -1.0); out << ",";
					write_ms_or_blank(best_ms.size() > 3 ? best_ms[3] : -1.0); out << ",";

					out << max_recall << "," << max_recall_ef << "\n";

					std::cout << "Solmaz: wrote table row to " << out_csv << " for " << method << std::endl;
					} // end variant loop
				}

				// Solmaz Seyed monir
		       // LSH-APG baseline (no DAP pruning during construction)
				
				if (divG_base) {
					divG_base->ef = divG->ef;
					divG_base->use_pseudo_search = false;
					divG_base->use_ligs_collision_search = false;

					const std::string method = "LSHAPG_BASE";

					// Delete/Insert timings on the baseline graph are still supported by our implementation,
					
					
					const double insert_ms = -1.0;
					const double delete_ms = -1.0;

					// Search timing at recall targets
					std::vector<double> best_ms(recall_targets.size(), -1.0);
					float max_recall = -1.0f;
					int max_recall_ef = -1;
					for (int efv : ef_sweep) {
						divG_base->ef = efv;
						SearchResult r = solmaz_graphSearch_no_write(c, (int)k, divG_base, prep, beta, /*qType=*/0, QnumTable);
						if (r.recall > max_recall) { max_recall = r.recall; max_recall_ef = efv; }
						for (size_t ti = 0; ti < recall_targets.size(); ++ti) {
							if (r.recall >= recall_targets[ti]) {
								if (best_ms[ti] < 0 || r.time < best_ms[ti]) best_ms[ti] = r.time;
							}
						}
						bool all_met = true;
						for (double v : best_ms) { if (v < 0) { all_met = false; break; } }
						if (all_met) break;
					}

					out << datasetName << ","
						<< method << ","
						<< k << ","
						<< divG_base->T << ","
						<< efC << ","
						<< solmaz_effective_update_ops << ","
						<< (divG_base->indexingTime > 0 ? divG_base->indexingTime : -1.0f) << ",";

					// Insert/Delete blank
					out << "," << ",";

					auto write_ms_or_blank = [&](double v) {
						if (v < 0) out << "";
						else out << v;
					};
					write_ms_or_blank(best_ms.size() > 0 ? best_ms[0] : -1.0); out << ",";
					write_ms_or_blank(best_ms.size() > 1 ? best_ms[1] : -1.0); out << ",";
					write_ms_or_blank(best_ms.size() > 2 ? best_ms[2] : -1.0); out << ",";
					write_ms_or_blank(best_ms.size() > 3 ? best_ms[3] : -1.0); out << ",";
					out << max_recall << "," << max_recall_ef << "\n";

					std::cout << "Solmaz: wrote table row to " << out_csv << " for " << method << std::endl;
				}
			}
		}
	}
	return 0;
}
