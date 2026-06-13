
/**  // Purpose:  Distance-Aware Pruning for Graph-based ANN Index Updates
// # Author: Solmaz Seyed Monir
 * 
 * Description:
 * - Computes a local threshold (τ_q) based on the 80th percentile of distances.
 * - Only retains neighbors that are within τ_q of the query node.
 * 
 * Benefits:
 * - Promotes sparser graph construction.
 * - Filters out weak or far neighbors.
 * - Stores τ_q in `last_threshold` for traceability.
 */
// Purpose:
//   Implements a local distance-based pruning strategy.
//   1. Collects all candidate neighbors from the LSH table.
//   2. Sorts them by distance to the query point.
//   3. Computes τ_q as the 80th percentile of all distances.
//   4. Filters and inserts only neighbors where dist < τ_q.
// Benefit:
//   - Reduces graph redundancy
//   - Enhances sparsity and locality-awareness
//   - Improves efficiency during query-time traversals
// =====================
//// Purpose:
// ===================== ANN Index Updates
// # Author: Solmaz Seyed Monir
// Solmaz/DAPG:
// - Switchable DAP local percentile pruning during construction (use_dap_pruning),
//   enabling comparison between full DAPG and baseline.
// - Dynamic insert/delete maintenance using flagStates, deleteNode(), and insertNode(),
//   including partial initial builds for sliding-window/random-new workloads.
// - Query-time ablation variants: pseudo traversal, collision-aware starts, anchor
//   starts, and learned-anchor starts.
// - Search/build safety for deleted nodes and serialization of active/deleted state.


#pragma once
#include "e2lsh.h"
#include "space_l2.h"
#include <algorithm>
#include <random>
#include <mutex>
#include <boost/math/distributions/chi_squared.hpp>

#if (__cplusplus >= 201703L) || (defined(_MSVC_LANG) && (_MSVC_LANG >= 201703L) && (_MSC_VER >= 1913))
#include <shared_mutex>
typedef std::shared_mutex mp_mutex;
//In C++17 format, read_lock can be shared
typedef std::shared_lock<std::shared_mutex> read_lock;
typedef std::unique_lock<std::shared_mutex> write_lock;
#else
typedef std::mutex mp_mutex;
//Not in C++17 format, read_lock is the same as write_lock and can not be shared
typedef std::unique_lock<std::mutex> read_lock;
typedef std::unique_lock<std::mutex> write_lock;
#endif // _HAS_CXX17

struct Node2
{
private:
public:
	int id = 0;
	Res* neighbors = nullptr;
	int in = 0;
	int out = 0;
	//int nextFill = -1;
public:
	bool* idxs = nullptr;
	std::unordered_set<int> remainings;

	Node2() {}
	Node2(int pId) :id(pId) {}
	Node2(int pId, Res* ptr) :id(pId), neighbors(ptr) {}

	void increaseIn() { ++in; }
	void decreaseIn() { --in; }
	void setOut(int out_) { out = out_; }
	int size() { return out; }
	void insertSafe(int pId, float dist_, int idx) {
		neighbors[idx] = Res(dist_, pId);
	}
	bool findSmaller(float dist_) {
		return dist_ < neighbors[0].dist;
	}

	bool findGreater(float dist_) {
		return dist_ > neighbors[0].dist;
	}

	inline void insert(float dist_, int pId)
	{
		neighbors[out++] = Res(dist_, pId);
		std::push_heap(neighbors, neighbors + out);
	}

	inline void insert(int pId, float dist_)
	{
		neighbors[out++] = Res(dist_, pId);
		std::push_heap(neighbors, neighbors + out);
	}

	inline int& erase()
	{
		std::pop_heap(neighbors, neighbors + out);
		--out;
		return neighbors[out].id;
	}


	inline bool isFull(int maxT_) {
		return out > maxT_;
	}
	inline void reset(int T_) {
		out = 0;
		in = 0;
	}

	int& operator[](int i) const
	{
		return neighbors[i].id;
	}
	Res& getNeighbor(int i) {
		return neighbors[i];
	}

	inline void readFromFile(std::ifstream& in_)
	{
		in_.read((char*)&id, sizeof(int));
		int nnSize = -1;
		in_.read((char*)&nnSize, sizeof(int));
		in_.read((char*)neighbors, sizeof(Res) * nnSize);
		in_.read((char*)&in, sizeof(int));
		in_.read((char*)&out, sizeof(int));
		out = nnSize;
	}

	inline void writeToFile(std::ofstream& out_)
	{
		out_.write((char*)&id, sizeof(int));
		int nnSize = out;
		out_.write((char*)&(nnSize), sizeof(int));
		out_.write((char*)neighbors, sizeof(Res) * nnSize);
		out_.write((char*)&(in), sizeof(int));
		out_.write((char*)&(out), sizeof(int));
	}
};

using minTopResHeap = std::vector<std::priority_queue<Res, std::vector<Res>, std::greater<Res>>>;
typedef std::priority_queue<std::pair<Res, int>, std::vector<std::pair<Res, int>>, std::greater<std::pair<Res, int>>> entryHeap;

//using namespace threadPoollib;


class divGraph :public zlsh
{
private:

	std::string file;
	size_t edgeTotal = 0;

	// Solmaz Seyed Monir: optional "partial build" controls for DAPG sliding window experiments.
	// If initial_active_N_build_ > 0, we only insert that many points initially, leaving the rest deleted ('D')
	// and available for later insertNode() calls. If shuffle_insert_build_ is false, insertion order is 0..N-1.
	int initial_active_N_build_ = -1;
	bool shuffle_insert_build_ = true;

	std::default_random_engine ng;
	std::uniform_int_distribution<uint64_t> rnd = std::uniform_int_distribution<uint64_t>(0, (uint64_t)-1);
	std::vector<int> records;
	int clusterFlag = 0;

	void oneByOneInsert();
	void refine();
	void buildExact(Preprocess* prep);
	void buildExactLikeHNSW(Preprocess* prep);
	void buildChunks();
	void insertPart(int pId, int ep, int mT, int mC, std::vector<std::vector<Res>>& partEdges);
	// Solmaz: helpers for dynamic delete/insert updates (Algorithm 6 style)
	inline bool isDeleted(int id) const {
		return (id >= 0 && id < (int)flagStates.size() && flagStates[id] == 'D');
	}
	inline void clearUnusedNeighborSlots(Node2* node) {
		// Ensure unused slots are marked as invalid (-1) because search loops over maxT.
		for (int i = node->out; i < maxT; ++i) {
			node->neighbors[i] = Res();
		}
	}
	inline void removeFromHashTables(int pId, const std::vector<zint>& keys) {
		for (int j = 0; j < L; ++j) {
			write_lock lock_h(hash_locks_[j]);
			auto range = hashTables[j].equal_range(keys[j]);
			for (auto it = range.first; it != range.second; ) {
				if (it->second == pId) it = hashTables[j].erase(it);
				else ++it;
			}
		}
	}
	inline void removeEdgeFromNodeUnsafe(Node2* node, int targetId) {
		// Removes all occurrences of targetId from node's neighbor list.
		int write = 0;
		for (int i = 0; i < node->out; ++i) {
			if (node->neighbors[i].id == targetId) continue;
			if (write != i) node->neighbors[write] = node->neighbors[i];
			++write;
		}
		node->out = write;
		if (node->out > 0) std::make_heap(node->neighbors, node->neighbors + node->out);
		clearUnusedNeighborSlots(node);
	}

public:
	//Only for construction, not saved
	int maxT = -1;
	std::atomic<size_t> compCostConstruction{ 0 };
	std::atomic<size_t> pruningConstruction{ 0 };
	float indexingTime = 0.0f;
	std::unordered_set<uint64_t> foundEdges;
	//std::vector<int> checkedArrs;
	int efC = 40;
	float coeff = 0.0f;
	float coeffq = 0.0f;
	std::vector<Res> linkListBase;
	//
	int T = -1;
	int step = 10;
	int nnD = 0;
	int lowDim = -1;
	float** myData = nullptr;
	std::string flagStates;
	std::vector<Node2*> linkLists;
	// Solmaz Seyed Monir: Runtime switch for comparing original graph search vs pseudo-graph search.
	// Default is the original behavior.
	bool use_pseudo_search = false;
	// Solmaz: Multi-start for pseudo traversal (beam start).
	// If >1 and use_pseudo_search==true, we keep the top-m closest seed entries as start points
	// instead of collapsing to a single entry. This can reduce local-optimum failures on hard datasets (e.g., SIFT k=100).
	int pseudo_entry_count = 1;
	// Solmaz: Enable DAPG Algorithm-1  collision-set expansion (pseudo-graph search).
	// This expands LSH collision sets (hashTables equal_range) instead of graph neighbors.
	// If true, it takes priority over use_pseudo_search.
	bool use_ligs_collision_search = false;

	// Solmaz Seyed Monir: Collision-weighted multi-entry start.
	// If true, seed the initial candidate heap using IDs that collide with the query across LSH tables
	// (i.e., appear in hashTables[j].equal_range(key_j(query))) and rank them by collision frequency.
	// This only changes the *start policy*; the rest of the traversal is unchanged.
	bool use_collision_aware_start = false;
	// Upper bound on how many candidates we consider from each table's exact bucket (prevents huge buckets from dominating).
	int aware_bucket_cap_per_table = 4096;
	// Minimum number of exact-bucket collisions (across tables) required for a seed to be considered "strong".
	// If <= 0, we auto-pick a reasonable default: for small L (e.g., L=2) we often want >=2 collisions for k>=100.
	// If not enough strong-collision seeds exist, we fall back to the legacy (collision desc, dist asc) ranking.
	int aware_min_collisions = 0;
	// Hybrid-aware scoring for entry selection (smooths out discrete switching):
	//   score = aware_alpha * (collisions/L) - aware_beta * distRankNorm
	// where distRankNorm is the distance rank within the top-N closest seeds, normalized to [0,1].
	// If aware_beta <= 0, selection degrades to collision-only; if aware_alpha <= 0, it becomes distance-rank only.
	float aware_alpha = 1.0f;
	float aware_beta = 0.15f;
	// For collision-aware start: we pick start nodes from the top-N closest LSH candidates,
	// breaking ties / prioritizing by collision frequency (higher is better).
	int aware_rerank_topN_by_dist = 256;
	// How many starting nodes to push into pqEntries (multi-entry start). 1 means classic single-entry.
	int aware_entry_count = 8;

	// Solmaz Seyed Monir: Anchor-based entry points (no learning).
	// If enabled and anchors are loaded, we choose the closest anchors to the query as entry points.
	// Anchors are point IDs in [0, N) (base IDs, not including the 200 query prefix in .data_new).
	bool use_anchor_start = false;
	int anchor_entry_count = 8;
	// Optional local refinement: after selecting an anchor, do a cheap 1-hop check in the current graph
	// and move the entry to the best (closest-to-query) node among {anchor} ∪ N_out(anchor).
	// This is a lightweight GATE-like idea (anchor routing + local refinement) without any learned model.
	bool anchor_local_refine = true;
	int anchor_refine_degree_cap = 32; // number of outgoing neighbors to inspect per anchor (0 = all)
	// Anchor warm-start (2-phase): run a tiny local best-first expansion starting from the anchor,
	// then use the best node reached as the entry point for the full search.
	// This is stronger than 1-hop refinement and often helps hard datasets (e.g., SIFT k=100).
	bool anchor_warm_start = true;
	int anchor_warmup_expansions = 32;   // number of pops/expansions during warm-start (0 disables)
	int anchor_warmup_degree_cap = 32;   // neighbors to inspect per expansion (0 = all)
	std::vector<int> anchor_ids;
	bool load_anchor_ids(const std::string& path) {
		anchor_ids.clear();
		std::ifstream in(path);
		if (!in.is_open()) return false;
		std::string line;
		while (std::getline(in, line)) {
			if (line.empty()) continue;
			try {
				int id = std::stoi(line);
				if (id >= 0 && id < N) anchor_ids.push_back(id);
			}
			catch (...) {
				// ignore bad lines
			}
		}
		return !anchor_ids.empty();
	}

	// Solmaz Seyed Monir: Learned anchor start (predict best anchors by L2).
	// We train a lightweight softmax regression model offline to predict the nearest anchor label.
	// At query time, we pick top-m anchors by model score (logits) and use their point IDs as entry points.
	bool use_learned_anchor_start = false;
	int learned_anchor_entry_count = 8;
	int learned_anchor_num_classes = 0; // should equal anchor_ids.size()
	int learned_anchor_dim = 0;         // should equal dim
	std::vector<float> learned_anchor_W; // row-major: [C][dim]
	std::vector<float> learned_anchor_b; // [C]
	bool load_learned_anchor_model(const std::string& path) {
		learned_anchor_num_classes = 0;
		learned_anchor_dim = 0;
		learned_anchor_W.clear();
		learned_anchor_b.clear();
		std::ifstream in(path, std::ios::binary);
		if (!in.is_open()) return false;
		int32_t C = 0, D = 0;
		in.read((char*)&C, sizeof(int32_t));
		in.read((char*)&D, sizeof(int32_t));
		if (!in.good() || C <= 0 || D <= 0) return false;
		learned_anchor_num_classes = (int)C;
		learned_anchor_dim = (int)D;
		learned_anchor_W.resize((size_t)C * (size_t)D);
		learned_anchor_b.resize((size_t)C);
		in.read((char*)learned_anchor_W.data(), sizeof(float) * learned_anchor_W.size());
		in.read((char*)learned_anchor_b.data(), sizeof(float) * learned_anchor_b.size());
		return in.good();
	}

	threadPoollib::VisitedListPool* visited_list_pool_ = nullptr;
	// Solmaz (Idea3): fast visited marking for query-time search.
	// Avoids O(N) per-query flag initialization by using an epoch/tag array.
	std::vector<uint32_t> q_visit_marks_;
	std::vector<uint32_t> q_cand_marks_;
	uint32_t q_visit_tag_ = 1;
	uint32_t q_cand_tag_ = 1;

	inline void ensure_query_marks() {
		if (q_visit_marks_.size() != (size_t)N) q_visit_marks_.assign((size_t)N, 0u);
		if (q_cand_marks_.size() != (size_t)N) q_cand_marks_.assign((size_t)N, 0u);
	}
	inline uint32_t next_tag(uint32_t& tag, std::vector<uint32_t>& marks) {
		++tag;
		if (tag == 0u) { // overflow: rare
			std::fill(marks.begin(), marks.end(), 0u);
			tag = 1u;
		}
		return tag;
	}
	std::vector<mp_mutex> link_list_locks_;
	std::vector<mp_mutex> hash_locks_;
	mp_mutex hash_lock;
	int ef = -1;
	int first_id = 0;
	uint64_t getKey(int u, int v);
	inline constexpr uint64_t getKey(tPoints& tp) const noexcept { return *(uint64_t*)&tp; }
public:
	float last_threshold = -1.0f; // Store the last used threshold for DAP
	// Solmaz: Toggle DAP local percentile pruning during construction.
	// If false, insertLSHRefine will behave closer to the original LSH-APG edge insertion (no local percentile filter).
	bool use_dap_pruning = true;
	std::string getFilename() const { return file; }
	void knn(queryN* q) override;
	//void knn(queryN* q);
	void knnHNSW(queryN* q);
	void insertHNSW(int pId);
	//int searchLSH(int pId, std::vector<zint>& keys, std::priority_queue<Res>& candTable, threadPoollib::vl_type* checkedArrs_local, threadPoollib::vl_type tag);
	int searchLSH(int pId, std::vector<zint>& keys, std::priority_queue<Res>& candTable, std::unordered_set<int>& checkedArrs_local, threadPoollib::vl_type tag);
	//int searchLSH(std::vector<zint>& keys, std::priority_queue<Res>& candTable, threadPoollib::vl_type* checkedArrs_local, threadPoollib::vl_type tag);
	//int searchLSH(std::vector<zint>& keys, std::priority_queue<Res>& candTable);
	void insertLSHRefine(int pId);
	// Solmaz Seyed Monir: Dynamic update APIs (Algorithm 6 DAPG). These assume the point's vector already exists in myData/hashval.
	// deleteNode(): removes the node from hash tables and graph, and prunes affected neighbors.
	void deleteNode(int pId, float pruning_percentile = 0.8f);
#if 0
	// Disabled experimental maintenance APIs kept for recovery.
	// lazyDeleteNode(): marks the node deleted and removes it from hash tables, but defers edge repair/pruning.
	void lazyDeleteNode(int pId);
	// cleanupLazyDeletedBatch(): repairs stale edges once after a batch of lazy deletes/inserts.
	void cleanupLazyDeletedBatch(const std::vector<int>& deleted_ids, const std::vector<int>& inserted_ids, float pruning_percentile = 0.8f);
#endif
	// insertNode(): re-inserts a previously deleted node using the existing insertLSHRefine pipeline.
	void insertNode(int pId);
	//int searchInBuilding(int pId, int ep, Res* arr, int& size_res);
	int searchInBuilding(int p, std::priority_queue<Res, std::vector<Res>, std::greater<Res>>& eps, Res* arr, int& size_res, std::unordered_set<int>& checkedArrs_local, threadPoollib::vl_type tag);
	void chooseNN_simple(Res* arr, int& size_res);
	void chooseNN_div(Res* arr, int& size_res);
	void chooseNN(Res* arr, int& size_res);
	void chooseNN_simple(Res* arr, int& size_res, Res new_res);
	void chooseNN_div(Res* arr, int& size_res, Res new_res);
	void chooseNN(Res* arr, int& size_res, Res new_res);
	void bestFirstSearchInGraph(queryN* q, std::vector<uint32_t>& visited, uint32_t vtag, entryHeap& pqEntries);
	// Solmaz: Pseudo-graph best-first search (Algorithm-1 style)
	void bestFirstSearchPseudo(queryN* q, std::vector<uint32_t>& visited, uint32_t vtag, entryHeap& pqEntries);
	// Solmaz: LIGS Algorithm-1 exact collision-set expansion (no graph neighbor expansion).
	void bestFirstSearchLIGSCollision(queryN* q, std::vector<uint32_t>& visited, uint32_t vtag, entryHeap& pqEntries);
	
	void showInfo(Preprocess* prep);
	void traverse();
	void save(const std::string& file) override;
public:
	divGraph(Preprocess& prep, Parameter& param_, const std::string& file_, int T_,int efC_, double probC = 0.95, double probQ = 0.99);
	// Solmaz: Build constructor with explicit DAP pruning toggle.
	divGraph(Preprocess& prep, Parameter& param_, const std::string& file_, int T_,int efC_, bool use_dap_pruning_, double probC = 0.95, double probQ = 0.99);
	// Solmaz: Build constructor with DAP pruning toggle + partial initial build controls (for FIFO sliding-window workloads).
	divGraph(Preprocess& prep, Parameter& param_, const std::string& file_, int T_, int efC_, bool use_dap_pruning_,
		int initial_active_N_, bool shuffle_insert_, double probC = 0.95, double probQ = 0.99);
	divGraph(Preprocess* prep, const std::string& path, double probQ = 0.99);
};

#include "basis.h"
#include <queue>
#include <functional>
#include <unordered_set>
#include <stdio.h>
#include <fstream>

divGraph::divGraph(Preprocess& prep_, Parameter& param_, const std::string& file_, int T_, int efC_, double probC,double probQ) :zlsh(prep_, param_, ""), link_list_locks_(prep_.data.N)
{
	// Default: DAP pruning enabled (backward compatible).
	use_dap_pruning = true;
	myData = prep_.data.val;
	T = T_;
	dim = prep_.data.dim;
	file = file_;
	lowDim = K;
	if (L == 0) lowDim = 0;
	maxT = 2 * T;
	//maxT = T;
	efC = 5 * T / 2;
	efC = efC_;
	visited_list_pool_ = new threadPoollib::VisitedListPool(1, N);

	normalizeHash();
	double _coeff = 1.0, _coeffq = 1.0;
	if (lowDim) {
		boost::math::chi_squared chi(lowDim);
		_coeff = sqrt(boost::math::quantile(chi, probC));
		if (probQ == 1.0) _coeffq = DBL_MAX;
		else _coeffq = sqrt(boost::math::quantile(chi, probQ));
	}
	
#ifdef USE_SQRDIST
	_coeff = _coeff * _coeff;
	coeff = W * W / _coeff;
	_coeffq = _coeffq * _coeffq;
	coeffq = W * W / _coeffq;
#else
	coeff = W / _coeff;
	coeffq = W / _coeffq;
#endif

	lsh::timer timer;
	std::cout << "CONSTRUCTING GRAPH..." << std::endl;
	timer.restart();
	oneByOneInsert();
	std::cout << "CONSTRUCTING TIME: " << timer.elapsed() << "s." << std::endl << std::endl;
	indexingTime = timer.elapsed();

	std::cout << "SAVING GRAPH..." << std::endl;
	timer.restart();
	save(file);
	std::cout << "SAVING TIME: " << timer.elapsed() << "s." << std::endl << std::endl;

	showInfo(&prep_);
}

divGraph::divGraph(Preprocess& prep_, Parameter& param_, const std::string& file_, int T_, int efC_, bool use_dap_pruning_, double probC,double probQ)
	: zlsh(prep_, param_, ""), link_list_locks_(prep_.data.N)
{
	// Solmaz Seyed Monir: Configure DAP pruning BEFORE building.
	use_dap_pruning = use_dap_pruning_;
	// Default: build all points (backward compatible).
	initial_active_N_build_ = -1;
	shuffle_insert_build_ = true;
	myData = prep_.data.val;
	T = T_;
	dim = prep_.data.dim;
	file = file_;
	lowDim = K;
	if (L == 0) lowDim = 0;
	maxT = 2 * T;
	efC = 5 * T / 2;
	efC = efC_;
	visited_list_pool_ = new threadPoollib::VisitedListPool(1, N);

	normalizeHash();
	double _coeff = 1.0, _coeffq = 1.0;
	if (lowDim) {
		boost::math::chi_squared chi(lowDim);
		_coeff = sqrt(boost::math::quantile(chi, probC));
		if (probQ == 1.0) _coeffq = DBL_MAX;
		else _coeffq = sqrt(boost::math::quantile(chi, probQ));
	}

#ifdef USE_SQRDIST
	_coeff = _coeff * _coeff;
	coeff = W * W / _coeff;
	_coeffq = _coeffq * _coeffq;
	coeffq = W * W / _coeffq;
#else
	coeff = W / _coeff;
	coeffq = W / _coeffq;
#endif

	lsh::timer timer;
	std::cout << "CONSTRUCTING GRAPH..." << std::endl;
	timer.restart();
	oneByOneInsert();
	std::cout << "CONSTRUCTING TIME: " << timer.elapsed() << "s." << std::endl << std::endl;
	indexingTime = timer.elapsed();

	std::cout << "SAVING GRAPH..." << std::endl;
	timer.restart();
	save(file);
	std::cout << "SAVING TIME: " << timer.elapsed() << "s." << std::endl << std::endl;

	showInfo(&prep_);
}

divGraph::divGraph(Preprocess& prep_, Parameter& param_, const std::string& file_, int T_, int efC_, bool use_dap_pruning_,
	int initial_active_N_, bool shuffle_insert_, double probC, double probQ)
	: zlsh(prep_, param_, ""), link_list_locks_(prep_.data.N)
{
	// Solmaz: Configure DAP pruning BEFORE building.
	use_dap_pruning = use_dap_pruning_;
	initial_active_N_build_ = initial_active_N_;
	shuffle_insert_build_ = shuffle_insert_;

	myData = prep_.data.val;
	T = T_;
	dim = prep_.data.dim;
	file = file_;
	lowDim = K;
	if (L == 0) lowDim = 0;
	maxT = 2 * T;
	efC = 5 * T / 2;
	efC = efC_;
	visited_list_pool_ = new threadPoollib::VisitedListPool(1, N);

	normalizeHash();
	double _coeff = 1.0, _coeffq = 1.0;
	if (lowDim) {
		boost::math::chi_squared chi(lowDim);
		_coeff = sqrt(boost::math::quantile(chi, probC));
		if (probQ == 1.0) _coeffq = DBL_MAX;
		else _coeffq = sqrt(boost::math::quantile(chi, probQ));
	}

#ifdef USE_SQRDIST
	_coeff = _coeff * _coeff;
	coeff = W * W / _coeff;
	_coeffq = _coeffq * _coeffq;
	coeffq = W * W / _coeffq;
#else
	coeff = W / _coeff;
	coeffq = W / _coeffq;
#endif

	lsh::timer timer;
	std::cout << "CONSTRUCTING GRAPH..." << std::endl;
	timer.restart();
	oneByOneInsert();
	std::cout << "CONSTRUCTING TIME: " << timer.elapsed() << "s." << std::endl << std::endl;
	indexingTime = timer.elapsed();

	std::cout << "SAVING GRAPH..." << std::endl;
	timer.restart();
	save(file);
	std::cout << "SAVING TIME: " << timer.elapsed() << "s." << std::endl << std::endl;

	showInfo(&prep_);
}

divGraph::divGraph(Preprocess* prep, const std::string& path, double probQ):link_list_locks_(prep->data.N)
{
	myData = prep->data.val;
	file = path;
	// Solmaz: If this index file predates persistence of indexingTime, we will keep -1.
	indexingTime = -1.0f;

	std::ifstream in(file, std::ios::binary);
	if (!in.good()) {
		std::cout << BOLDGREEN << "WARNING:\n" << GREEN << "Could not find the divGraph index file. \n"
			<< "Filename: " << file.c_str() << RESET;
		exit(-1);
	}

	lsh::timer timer;
	std::cout << "LOADING GRAPH..." << std::endl;
	/***********************************************************************************/
	in.read((char*)&N, sizeof(int));
	in.read((char*)&dim, sizeof(int));
	in.read((char*)&L, sizeof(int));
	in.read((char*)&K, sizeof(int));
	in.read((char*)&W, sizeof(float));
	in.read((char*)&u, sizeof(int));

	S = L * K;

	//hashval
	hashval = new float* [N];
	for (int i = 0; i < N; ++i) {
		hashval[i] = new float[S];
		in.read((char*)(hashval[i]), sizeof(float) * S);
	}

	//hashpar,hashmin,hashmax
	hashPar.rndBs = new float[S];
	hashMaxs.resize(S);
	hashMins.resize(S);
	hashPar.rndAs = new float* [S];

	in.read((char*)hashPar.rndBs, sizeof(float) * S);
	in.read((char*)&hashMins[0], sizeof(float) * S);
	in.read((char*)&hashMaxs[0], sizeof(float) * S);
	for (int i = 0; i != S; ++i) {
		hashPar.rndAs[i] = new float[dim];
		in.read((char*)hashPar.rndAs[i], sizeof(float) * dim);
	}

	//Index
	hashTables.resize(L);
	// Solmaz: When loading an existing index (isbuilt=1 path), we still need runtime locks.
	// These are initialized during oneByOneInsert() for freshly-built indexes, but not here.
	std::vector<mp_mutex>(L).swap(hash_locks_);
	zint key;
	int pointId;
	std::cout << "Loading hash..." << std::endl;
	lsh::progress_display pd(0);
	// Solmaz: Backward-compatible hash table serialization:
	// - Old format: exactly N entries per table (no counts)
	// - New format: magic + per-table counts (supports partial builds)
	{
		const uint32_t MAGIC_HTBL = 0x4C425448; // "HTBL" in little-endian
		uint32_t maybe_magic = 0;
		in.read((char*)&maybe_magic, sizeof(uint32_t));
		if (in && maybe_magic == MAGIC_HTBL) {
			// New format: per-table size + entries
			std::vector<uint64_t> sizes(L, 0);
			uint64_t total = 0;
			for (int i = 0; i < L; ++i) {
				in.read((char*)&sizes[i], sizeof(uint64_t));
				total += sizes[i];
			}
			pd.restart(total);
			for (int i = 0; i < L; ++i) {
				for (uint64_t j = 0; j < sizes[i]; ++j) {
					in.read((char*)&(key), sizeof(zint));
					in.read((char*)&((pointId)), sizeof(int));
					hashTables[i].insert({ key,pointId });
					++pd;
				}
			}
		}
		else {
			// Old format: rewind and read N entries per table
			in.clear();
			in.seekg(-((std::streamoff)sizeof(uint32_t)), std::ios::cur);
			pd.restart((size_t)N * (size_t)L);
			for (int i = 0; i != L; ++i) {
				for (int j = 0; j < N; ++j) {
					in.read((char*)&(key), sizeof(zint));
					in.read((char*)&((pointId)), sizeof(int));
					hashTables[i].insert({ key,pointId });
					++pd;
				}
			}
		}
	}

	/**********************************************************************/

	in.read((char*)&T, sizeof(int));
	in.read((char*)&step, sizeof(int));
	in.read((char*)&nnD, sizeof(int));
	in.read((char*)&edgeTotal, sizeof(size_t));
	in.read((char*)&lowDim, sizeof(int));
	maxT = 2 * T;
	// Solmaz: Required for insertLSHRefine()/dynamic insert/delete operations.
	// (During fresh builds this is created in the other constructor.)
	if (!visited_list_pool_) visited_list_pool_ = new threadPoollib::VisitedListPool(1, N);

	double _coeffq = 1.0;
	if (lowDim) {
		boost::math::chi_squared chi(lowDim);
		if (probQ == 1.0) _coeffq = DBL_MAX;
		else _coeffq = sqrt(boost::math::quantile(chi, probQ));
	}
#ifdef USE_SQRDIST
	_coeffq = _coeffq * _coeffq;
	coeffq = W * W / _coeffq;
#else
	coeffq = W / _coeffq;
#endif

	int len = -1;
	in.read((char*)&len, sizeof(int));
	char* buf = new char[len];
	in.read((char*)buf, sizeof(char) * len);
	flagStates.assign(buf);
	delete[] buf;

	in.read((char*)&len, sizeof(int));
	assert(len == N);
	//linkLists.resize(N, nullptr);
	//How to quickly initialize?
	linkLists.resize(N, nullptr);
	std::cout << "Loading graph..." << std::endl;
	linkListBase.resize((size_t)N * (size_t)maxT);
	//std::swap(pd,lsh::progress_display(N));
	pd.restart(N);
	for (size_t i = 0; i < N; ++i) {
		linkLists[i] = new Node2(i, (Res*)(&(linkListBase[i * (size_t)maxT])));
		linkLists[i]->readFromFile(in);
	}

	// Solmaz: Load persisted indexingTime if present (newer index files).
	// If not present, keep indexingTime = -1.0f.
	{
		float idx_time_tmp = -1.0f;
		in.read((char*)&idx_time_tmp, sizeof(float));
		if (in.gcount() == (std::streamsize)sizeof(float)) {
			indexingTime = idx_time_tmp;
		}
		else {
			// old file format: no trailing indexingTime
			indexingTime = -1.0f;
			in.clear();
		}
	}

	in.close();

	std::cout << "LOADING TIME: " << timer.elapsed() << "s." << std::endl << std::endl;

	showInfo(prep);
}

int  divGraph::searchLSH(int pId, std::vector<zint>& keys, std::priority_queue<Res>& candTable, std::unordered_set<int>& checkedArrs_local, threadPoollib::vl_type tag)
{
	read_lock lock_hr(hash_lock);
	std::vector<read_lock> lock_hs;
	for (int i = 0; i < L; ++i) {
		lock_hs.push_back(read_lock(hash_locks_[i]));
	}

	Res res_pair;

	int lshUB = N / 200;
	lshUB = L * log(pId + 1);
	int step = 2;

	std::vector<int> numAccess(L);
	std::vector<std::multimap<zint, int>::iterator> lpos(L), rpos(L), qpos(L);

	std::priority_queue<posInfo> lEntries, rEntries;


	for (int j = 0; j < L; j++) {
		qpos[j] = hashTables[j].lower_bound(keys[j]);
		if (qpos[j] != hashTables[j].begin()) {
			lpos[j] = qpos[j];
			--lpos[j];
#ifdef USE_LCCP
			lEntries.push(posInfo(j, getLLCP(lpos[j]->first, keys[j])));
#else
			lEntries.push(posInfo(j, getLevel(lpos[j]->first, qpos[j]->first)));
#endif // USE_LCCP

		}
		//
		rpos[j] = qpos[j];
		if (rpos[j] != hashTables[j].end()) {
#ifdef USE_LCCP
			rEntries.push(posInfo(j, getLLCP(rpos[j]->first, keys[j])));
#else
			rEntries.push(posInfo(j, getLevel(rpos[j]->first, qpos[j]->first)));
#endif // USE_LCCP
		}
	}

	while (!(lEntries.empty() && rEntries.empty())) {
		posInfo t;
		bool f = true;//TRUE:left; FALSE:right
		if (lEntries.empty()) f = false;
		else if (rEntries.empty()) f = true;
		else if (rEntries.top().dist > lEntries.top().dist) f = false;

		if (f) {
			t = lEntries.top();
			lEntries.pop();
			for (int i = 0; i < step; ++i) {
				++numAccess[t.id];
				res_pair.id = lpos[t.id]->second;
				if (checkedArrs_local.find(res_pair.id)==checkedArrs_local.end()) {
					res_pair.dist = cal_dist(myData[pId], myData[res_pair.id], dim);
					candTable.push(res_pair);
					//checkedArrs_local[res_pair.id] = tag;
					checkedArrs_local.emplace(res_pair.id);
				}

				if (lpos[t.id] != hashTables[t.id].begin()) {
					--lpos[t.id];
				}
				else {
					break;
				}
			}
			if (lpos[t.id] != hashTables[t.id].begin()) {
#ifdef USE_LCCP
				t.dist = getLLCP(lpos[t.id]->first, keys[t.id]);
#else
				t.dist = getLevel(lpos[t.id]->first, qpos[t.id]->first);
#endif // USE_LCCP
				lEntries.push(t);
			}

		}
		else {
			t = rEntries.top();
			rEntries.pop();
			//read_lock lock_h(hash_locks_[t.id]);
			for (int i = 0; i < step; ++i) {
				++numAccess[t.id];
				res_pair.id = rpos[t.id]->second;
				if (checkedArrs_local.find(res_pair.id)==checkedArrs_local.end()) {
					res_pair.dist = cal_dist(myData[pId], myData[res_pair.id], dim);
					candTable.push(res_pair);
					//checkedArrs_local[res_pair.id] = tag;
					checkedArrs_local.emplace(res_pair.id);
				}
				if (++rpos[t.id] == hashTables[t.id].end()) {
					break;
				}
			}
			if (rpos[t.id] != hashTables[t.id].end()) {
#ifdef USE_LCCP
				t.dist = getLLCP(rpos[t.id]->first, keys[t.id]);
#else
				t.dist = getLevel(rpos[t.id]->first, qpos[t.id]->first);
#endif // USE_LCCP
				rEntries.push(t);
			}
		}
		if (candTable.size() >= lshUB) break;
	}
	
	return 0;
}

void divGraph::insertLSHRefine(int pId)
{
	std::priority_queue<Res> candTable;
	std::vector<zint> keys(L);
	threadPoollib::VisitedList* vl = visited_list_pool_->getFreeVisitedList();
	auto checkedArrs_local = vl->mass;
	//checkedArrs_local.reserve(N);
	threadPoollib::vl_type tag = vl->curV;
	for (int j = 0; j < L; j++) {
		keys[j] = getZ(hashval[pId] + j * K);
	}
	// Solmaz: If this point was previously inserted (dynamic update), remove it first to avoid duplicates.
	removeFromHashTables(pId, keys);
	
	searchLSH(pId, keys, candTable, checkedArrs_local, tag);
	compCostConstruction += candTable.size();

	if (pId != first_id && candTable.empty()) {
		candTable.emplace(first_id, cal_dist(myData[pId], myData[first_id], dim));
		//checkedArrs_local[first_id] = tag;
		checkedArrs_local.emplace(first_id);
	}

	write_lock lock(link_list_locks_[pId]);
	std::priority_queue<Res, std::vector<Res>, std::greater<Res>> eps;

	// 1. Collect all candidate distances
	std::vector<Res> candidates;
	while (!candTable.empty()) {
		candidates.push_back(candTable.top());
		candTable.pop();
	}

	// 2. Sort by distance
	std::sort(candidates.begin(), candidates.end(), [](const Res& a, const Res& b) {
		return a.dist < b.dist;
	});

	// 3. (Optional) DAP local percentile pruning (e.g., 80th percentile).
	// If disabled, we keep all candidates (subject to efC cap and chooseNN()).
	float threshold = -1.0f;
	if (use_dap_pruning && !candidates.empty()) {
		size_t percentile_idx = candidates.size() * 0.8;
		threshold = candidates[std::min(percentile_idx, candidates.size() - 1)].dist;
	}
	this->last_threshold = threshold; // Store the threshold for output (-1 when disabled/empty)

	// 4. Insert edges (filtered if DAP pruning is enabled)
	for (const auto& u : candidates) {
		if (!use_dap_pruning || u.dist < threshold) {
			linkLists[pId]->insert(u.dist, u.id);
			if (linkLists[pId]->size() > efC) linkLists[pId]->erase();
			eps.emplace(u.dist, u.id);
		}
	}
	compCostConstruction += searchInBuilding(pId, eps, linkLists[pId]->neighbors, linkLists[pId]->out, checkedArrs_local, tag);
	chooseNN(linkLists[pId]->neighbors, linkLists[pId]->out);
	clearUnusedNeighborSlots(linkLists[pId]);

	visited_list_pool_->releaseVisitedList(vl);

	int len = linkLists[pId]->size();
	//Res* arr = new Res[len];
	//memcpy(arr, linkLists[pId]->neighbors, len * sizeof(Res));
	lock.unlock();
	for (int pos = 0; pos < len; ++pos) {
		auto& x = linkLists[pId]->neighbors[pos];
		int& qId = x.id;
		float& dist = x.dist;

		write_lock lock_q(link_list_locks_[qId]);

		chooseNN(linkLists[qId]->neighbors, linkLists[qId]->out, Res(pId, dist));
		clearUnusedNeighborSlots(linkLists[qId]);
	}

	for (int j = 0; j < L; j++) {
		write_lock lock_h(hash_locks_[j]);
		hashTables[j].insert({ keys[j],pId });
	}
}

int divGraph::searchInBuilding(int p, std::priority_queue<Res, std::vector<Res>, std::greater<Res>>& eps, Res* arr, int& size_res,
	std::unordered_set<int>& checkedArrs_local, threadPoollib::vl_type tag)
{
	//size_res = 0;
	Res res_pair;
	int cost = 0;
	while (!eps.empty()) {
		auto u = eps.top();
		read_lock lock_e(link_list_locks_[u.id]);
		if (u > arr[0]) break;
		eps.pop();
		for (int pos = 0; pos < linkLists[u.id]->size(); ++pos) {
			res_pair.id = (*(linkLists[u.id]))[pos];
			// Solmaz: avoid revisiting the point being inserted (can appear via stale incoming edges),
			// and skip deleted nodes during dynamic update experiments.
			if (res_pair.id == p) continue;
			if (isDeleted(res_pair.id)) continue;
			if (checkedArrs_local.find(res_pair.id)==checkedArrs_local.end()) {
				//checkedArrs_local[res_pair.id] = tag;
				checkedArrs_local.emplace(res_pair.id);
				if (0 || arr[0].dist> cal_dist(hashval[p], hashval[res_pair.id], lowDim) * coeff) {
					res_pair.dist = cal_dist(myData[p], myData[res_pair.id], dim);
					++cost;
					if(arr[0]> res_pair||size_res<efC){
						arr[size_res++] = res_pair;
						std::push_heap(arr, arr + size_res);
						if (size_res >= efC) {
							std::pop_heap(arr, arr + size_res);
							size_res--;
						}
						eps.emplace(res_pair);
					}
				}
				else {
					++pruningConstruction;
				}

			}
		}
	}

	return cost;
}

void divGraph::chooseNN_simple(Res* arr, int& size_res)
{
	while (size_res > T) {
		std::pop_heap(arr, arr + size_res);
		size_res--;
	}
}

void divGraph::chooseNN_div(Res* arr, int& size_res)
{
	if (size_res <= T) return;

	int old_res = size_res;

	int choose_num = 0;
	std::sort(arr, arr + size_res);
	//std::priority_queue<Res, std::vector<Res>, std::greater<Res>> res;
	for (int i = 0; i < size_res; ++i) {
		if (choose_num >= T) break;

		auto& curRes = arr[i];
		bool flag = true;
		for (int j = 0; j < choose_num; ++j) {
			++compCostConstruction;
			float dist = cal_dist(myData[curRes.id], myData[arr[j].id], dim);
			if ( dist < curRes.dist) {
				flag = false;
				break;
			}
		}
		if (flag) {
			if (choose_num < i) {
				Res temp = arr[i];
				arr[i] = arr[choose_num];
				arr[choose_num] = temp;
			}
			choose_num++;
		}
	}

	size_res = choose_num;
	std::swap(arr[size_res - 1], arr[0]);
	
	bool f = false;
	for (int i = 0; i < size_res - 1; ++i) {
		if (arr[i] == arr[i + 1]) {
			f = true;
			break;
		}
	}
	if (f) {
		int pId = (arr - linkLists[0]->neighbors) / (linkLists[1]->neighbors - linkLists[0]->neighbors);
		printf("Error in %d:\n", pId);
		for (int j = 0; j < size_res; ++j) {
			printf("%2d: dist=%f, id=%d\n", j, arr[j].dist, arr[j].id);
		}
#ifdef _MSC_VER
		system("pause");
#endif
	}
}

void divGraph::chooseNN(Res* arr, int& size_res)
{
#ifdef DIV
	chooseNN_div(arr, size_res);
	//
#else
	chooseNN_simple(arr, size_res);
#endif
}

void divGraph::chooseNN_simple(Res* arr, int& size_res, Res new_res)
{
	if (myFind(arr, arr + size_res, new_res)) return;

	if (size_res < maxT) {
		arr[size_res++] = new_res;
		std::push_heap(arr, arr + size_res);
		//linkLists[qId]->insert(dist, pId);
		//linkLists[pId]->increaseIn();
	}
	else if (arr[0]>new_res) {
		//linkLists[linkLists[qId]->erase()]->decreaseIn();
		std::pop_heap(arr, arr + size_res);
		size_res--;
		arr[size_res++] = new_res;
		std::push_heap(arr, arr + size_res);
	}

	//while (size_res > T) {
	//	std::pop_heap(arr, arr + size_res);
	//	size_res--;
	//}
}

void divGraph::chooseNN_div(Res* arr, int& size_res, Res new_res)
{
	if (myFind(arr, arr + size_res, new_res)) return;

	if (size_res < maxT) {
		arr[size_res++] = new_res;
		std::sort(arr, arr + size_res);

		bool f = false;
		for (int i = 0; i < size_res - 1; ++i) {
			if (arr[i] == arr[i + 1]) {
				f = true;
				break;
			}
		}
		if (f) {
			int pId = (arr - linkLists[0]->neighbors) / (linkLists[1]->neighbors - linkLists[0]->neighbors);
			printf("Error in %d:\n", pId);
			for (int j = 0; j < size_res; ++j) {
				printf("%2d: dist=%f, id=%d\n", j, arr[j].dist, arr[j].id);
			}
#ifdef _MSC_VER
			system("pause");
#endif
		}
		std::swap(arr[size_res - 1], arr[0]);
		//arr[size_res] = arr[size_res - 1];
		//arr[size_res - 1] = arr[0];
		//arr[0] = arr[size_res];
	}
	else {
		arr[size_res++] = new_res;
		chooseNN_div(arr, size_res);

		/*if (arr[0] > new_res) {
			arr[size_res] = arr[size_res - 1];
			arr[size_res - 1] = arr[0];
			arr[0] = arr[size_res];

			auto idx = std::upper_bound(arr, arr + size_res, new_res) - arr;
			bool flag = true;
			for (int j = 0; j < idx; ++j) {
				++compCostConstruction;
				if (cal_dist(myData[new_res.id], myData[arr[j].id], dim) < new_res.dist) {
					flag = false;
					break;
				}
			}
			if (flag) {
				memmove(arr + idx + 1, arr + idx, (size_res - idx + 1) * sizeof(Res));
				arr[idx] = new_res;
				size_res++;
				int choose_num = idx + 1;
				std::sort(arr, arr + size_res);
				std::priority_queue<Res, std::vector<Res>, std::greater<Res>> res;
				for (int i = idx + 1; i < size_res; ++i) {
					if (choose_num >= maxT) break;
					auto& curRes = arr[i];
					++compCostConstruction;
					if (cal_dist(myData[curRes.id], myData[new_res.id], dim) < curRes.dist) {
						flag = false;
						break;
					}
					if (flag) {
						if (choose_num < i) {
							Res temp = arr[i];
							arr[i] = arr[choose_num];
							arr[choose_num] = temp;
						}
						choose_num++;
					}
					else flag = true;
				}

				size_res = choose_num;

				if (size_res < T) {
					size_res = T;
					std::sort(arr, arr + size_res);
				}
			}

			arr[size_res] = arr[size_res - 1];
			arr[size_res - 1] = arr[0];
			arr[0] = arr[size_res];
		}*/
	}
	
}

void divGraph::chooseNN(Res* arr, int& size_res, Res new_res)
{
#ifdef DIV
	chooseNN_div(arr, size_res, new_res);
#else
	chooseNN_simple(arr, size_res, new_res);
#endif

}

// Solmaz: Dynamic delete (Algorithm 6 style).
// Removes node pId from the hash tables and removes incident edges from the graph.
// Then applies a light local pruning (percentile) + existing chooseNN() to affected neighbors.
void divGraph::deleteNode(int pId, float pruning_percentile)
{
	if (pId < 0 || pId >= N) return;
	if (isDeleted(pId)) return;

	// 1) Mark deleted
	if ((int)flagStates.size() == N) flagStates[pId] = 'D';

	// 2) Remove from hash tables
	std::vector<zint> keys(L);
	for (int j = 0; j < L; ++j) keys[j] = getZ(hashval[pId] + j * K);
	removeFromHashTables(pId, keys);

	// 3) Collect current neighbors, then clear this node's adjacency list
	std::vector<int> affected;
	{
		write_lock lock_p(link_list_locks_[pId]);
		for (int i = 0; i < linkLists[pId]->out; ++i) {
			int nb = linkLists[pId]->neighbors[i].id;
			if (nb >= 0) affected.push_back(nb);
		}
		linkLists[pId]->out = 0;
		clearUnusedNeighborSlots(linkLists[pId]);
	}

	// 4) Remove incident edges and re-prune affected nodes
	for (int u : affected) {
		if (u < 0 || u >= N) continue;
		write_lock lock_u(link_list_locks_[u]);

		// Remove pId from u's neighbor list
		removeEdgeFromNodeUnsafe(linkLists[u], pId);

		// Apply local percentile pruning on current neighbor list (optional)
		if (linkLists[u]->out > 0 && pruning_percentile > 0.0f && pruning_percentile < 1.0f) {
			std::vector<Res> cur(linkLists[u]->neighbors, linkLists[u]->neighbors + linkLists[u]->out);
			std::sort(cur.begin(), cur.end(), [](const Res& a, const Res& b) { return a.dist < b.dist; });
			size_t idx = (size_t)(cur.size() * pruning_percentile);
			float thr = cur[std::min(idx, cur.size() - 1)].dist;
			int write = 0;
			for (auto& r : cur) {
				if (r.dist < thr && r.id >= 0 && !isDeleted(r.id)) {
					linkLists[u]->neighbors[write++] = r;
				}
			}
			linkLists[u]->out = write;
			if (linkLists[u]->out > 0) std::make_heap(linkLists[u]->neighbors, linkLists[u]->neighbors + linkLists[u]->out);
			clearUnusedNeighborSlots(linkLists[u]);
		}

		// Existing global pruning to degree T (or DIV variant)
		chooseNN(linkLists[u]->neighbors, linkLists[u]->out);
		clearUnusedNeighborSlots(linkLists[u]);
	}
}

#if 0
// Disabled experimental lazy delete for high-throughput update experiments.
// Search and insertion paths already skip deleted nodes, so this fast path only
// marks the node inactive and removes it from LSH buckets. Incident graph edges
// are left in place and ignored by traversal until a later cleanup/rebuild.
void divGraph::lazyDeleteNode(int pId)
{
	if (pId < 0 || pId >= N) return;
	if (isDeleted(pId)) return;

	if ((int)flagStates.size() == N) flagStates[pId] = 'D';

	std::vector<zint> keys(L);
	for (int j = 0; j < L; ++j) keys[j] = getZ(hashval[pId] + j * K);
	removeFromHashTables(pId, keys);
}

// Disabled experimental batch cleanup after lazy deletes/inserts.
// This removes stale deleted neighbors and reapplies pruning once per affected
// adjacency list, avoiding repeated repair for overlapping neighborhoods.
void divGraph::cleanupLazyDeletedBatch(const std::vector<int>& deleted_ids, const std::vector<int>& inserted_ids, float pruning_percentile)
{
	std::unordered_set<int> affected;
	affected.reserve((deleted_ids.size() + inserted_ids.size()) * (size_t)std::max(4, maxT));

	for (int pId : deleted_ids) {
		if (pId < 0 || pId >= N) continue;
		write_lock lock_p(link_list_locks_[pId]);
		for (int i = 0; i < linkLists[pId]->out; ++i) {
			int nb = linkLists[pId]->neighbors[i].id;
			if (nb >= 0 && nb < N && !isDeleted(nb)) affected.insert(nb);
		}
		linkLists[pId]->out = 0;
		clearUnusedNeighborSlots(linkLists[pId]);
	}

	for (int pId : inserted_ids) {
		if (pId < 0 || pId >= N || isDeleted(pId)) continue;
		affected.insert(pId);
		read_lock lock_p(link_list_locks_[pId]);
		for (int i = 0; i < linkLists[pId]->out; ++i) {
			int nb = linkLists[pId]->neighbors[i].id;
			if (nb >= 0 && nb < N && !isDeleted(nb)) affected.insert(nb);
		}
	}

	for (int u : affected) {
		if (u < 0 || u >= N || isDeleted(u)) continue;
		write_lock lock_u(link_list_locks_[u]);

		std::vector<Res> cur;
		cur.reserve((size_t)linkLists[u]->out);
		for (int i = 0; i < linkLists[u]->out; ++i) {
			const Res& r = linkLists[u]->neighbors[i];
			if (r.id >= 0 && r.id < N && !isDeleted(r.id)) cur.push_back(r);
		}

		if (!cur.empty() && pruning_percentile > 0.0f && pruning_percentile < 1.0f) {
			std::sort(cur.begin(), cur.end(), [](const Res& a, const Res& b) { return a.dist < b.dist; });
			size_t idx = (size_t)(cur.size() * pruning_percentile);
			float thr = cur[std::min(idx, cur.size() - 1)].dist;
			std::vector<Res> kept;
			kept.reserve(cur.size());
			for (const auto& r : cur) {
				if (r.dist < thr) kept.push_back(r);
			}
			cur.swap(kept);
		}

		linkLists[u]->out = 0;
		for (const auto& r : cur) {
			if (linkLists[u]->out >= maxT) break;
			linkLists[u]->neighbors[linkLists[u]->out++] = r;
		}
		if (linkLists[u]->out > 0) std::make_heap(linkLists[u]->neighbors, linkLists[u]->neighbors + linkLists[u]->out);
		chooseNN(linkLists[u]->neighbors, linkLists[u]->out);
		clearUnusedNeighborSlots(linkLists[u]);
	}
}
#endif

// Solmaz: Dynamic insert/re-activate (Algorithm 6 style).
// This assumes the data vector for pId already exists; we re-run insertLSHRefine to rebuild its edges.
void divGraph::insertNode(int pId)
{
	if (pId < 0 || pId >= N) return;

	// Mark active
	if ((int)flagStates.size() == N) flagStates[pId] = 'E';

	// Ensure its adjacency list is clean before reinsertion
	{
		write_lock lock_p(link_list_locks_[pId]);
		linkLists[pId]->out = 0;
		clearUnusedNeighborSlots(linkLists[pId]);
	}

	// Rebuild neighborhood + symmetric edges + hash insertion
	insertLSHRefine(pId);
}

void divGraph::oneByOneInsert()
{
	linkLists.resize(N, nullptr);
	int unitL = max(efC, maxT);

	linkListBase.resize((size_t)N * (size_t)unitL + efC);
	for (int i = 0; i < N; ++i) {
		linkLists[i] = new Node2(i, (Res*)(&(linkListBase[i * unitL])));
	}

	// Solmaz: if partial-build is enabled, mark all as deleted first, and only activate/insert the initial prefix.
	flagStates.assign(N, 'D');

	hashTables.resize(L);
	std::vector<mp_mutex>(L).swap(hash_locks_);

	int* idx = new int[N];
	for (int j = 0; j < N; ++j) {
		idx[j] = j;
	}

	if (shuffle_insert_build_) {
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::shuffle(idx, idx + N, std::default_random_engine(seed));
	}

	int activeN = N;
	if (initial_active_N_build_ > 0) activeN = std::min(initial_active_N_build_, N);
	if (activeN <= 0) activeN = 1;

	// Activate the chosen initial set.
	for (int i = 0; i < activeN; ++i) {
		flagStates[idx[i]] = 'E';
	}

	first_id = idx[0];
	insertLSHRefine(idx[0]);//Ensure there is at least one point in the graph before parallelizing
	lsh::progress_display pd(activeN - 1);

	// Add data to index
	ParallelFor(1, activeN, 96, [&](size_t i, size_t threadId) {
		insertLSHRefine(idx[i]);
		++pd;
	});

// #pragma omp parallel for //num_threads(32) 
// 	for (int i = 1; i < N; i++) {
// 		insertLSHRefine(idx[i]);
// 		++pd;
// 	}
	//std::cout << "count: " << pd.count() << std::endl;

//#pragma omp parallel for
//	for (int i = N - 1; i >= 0; i--) {
//		insertHNSW(i);
//		++pd;
//	}

	//refine();
}

void divGraph::refine()
{
	Res* rnns = new Res[N * maxT + 1];
	std::vector<int> rnnSize(N, 0);
	for (int i = 0; i < N; ++i) {
		const auto& nns = linkLists[i]->neighbors;
		for (int j = 0; j < linkLists[i]->size(); ++j) {
			auto& qId = nns[j].id;
			const auto& rnn = rnns + qId * maxT;
			if (rnnSize[qId] < maxT) {
				rnn[rnnSize[qId]++] = Res(nns[j].dist, i);
				std::push_heap(rnn, rnn + rnnSize[qId]);
			}
			else if (nns[j].dist < rnn[0].dist) {
				std::pop_heap(rnn, rnn + rnnSize[qId]);
				rnn[rnnSize[qId] - 1] = Res(nns[j].dist, i);
				std::push_heap(rnn, rnn + rnnSize[qId]);
			}

		}
	}
	for (int i = 0; i < N; ++i) {
		const auto& nns = linkLists[i]->neighbors;
		const auto& rnn = rnns + i * maxT;
		for (int j = 0; j < rnnSize[i]; ++j) {
			//if (rnn[j].dist > nns[0].dist) break;
			if (!myFind(nns, nns + linkLists[i]->size(), rnn[j])) {
				if (linkLists[i]->size() < maxT) {
					linkLists[i]->insert(rnn[j].dist, rnn[j].id);
				}
				else if (rnn[j].dist < nns[0].dist) {
					linkLists[i]->erase();
					linkLists[i]->insert(rnn[j].dist, rnn[j].id);
					//printf("Find one!\n");
				}
			}
		}
	}
}

void divGraph::buildExact(Preprocess* prep)
{
	getIndexes();
	flagStates.resize(N, 'E');
	////How to quickly initialize?
	//linkLists.resize(N, NULL);
	//for (auto& pt : linkLists) {
	//	pt = new Node2();
	//}
	for (int i = 0; i < N; ++i) {
		int numEdges = T;
		linkLists[i]->setOut(numEdges);
		for (int l = 1; l <= numEdges; ++l) {
			int v = prep->benchmark.indice[i + 200][l];
			linkLists[i]->insertSafe(v, prep->benchmark.dist[i + 200][l], l - 1);
			linkLists[v]->increaseIn();
		}
	}
}

void divGraph::buildExactLikeHNSW(Preprocess* prep)
{
	getIndexes();
	flagStates.resize(N, 'E');
	std::vector<float> minTDists(N);
	////How to quickly initialize?
	//linkLists.resize(N, NULL);
	//for (auto& pt : linkLists) {
	//	pt = new Node2();
	//}
	for (int i = 0; i < N; ++i) {
		int numEdges = T;
		linkLists[i]->setOut(numEdges);
		for (int l = 1; l <= numEdges; ++l) {
			int v = prep->benchmark.indice[i + 200][l];
			linkLists[i]->insertSafe(v, prep->benchmark.dist[i + 200][l], l - 1);
			linkLists[v]->increaseIn();
		}
		std::make_heap(&(linkListBase[0]) + i * maxT, &(linkListBase[0]) + i * maxT + T);
		minTDists[i] = linkLists[i]->getNeighbor(0).dist;
	}

	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < T; ++j) {
			auto& v = linkLists[i]->getNeighbor(j);
			if (v.dist > minTDists[v.id]) {// avoid inserting the appeared point
				if (linkLists[v.id]->findSmaller(v.dist)) {
					if (linkLists[v.id]->isFull(maxT)) linkLists[v.id]->erase();
					linkLists[v.id]->insert(v.dist, i);
				}

			}
		}
	}

}

inline uint64_t divGraph::getKey(int u, int v)
{
	if (u > v) {
		return (((uint64_t)u) << 32) | (uint64_t)v;
	}
	else {
		return getKey(v, u);
	}
}

extern int _lsh_UB;

void divGraph::knn(queryN* q)
{
	lsh::timer timer;
	timer.restart();
	q->hashval = calHash(q->queryPoint);

	ensure_query_marks();
	const uint32_t cand_tag = next_tag(q_cand_tag_, q_cand_marks_);
	const uint32_t visit_tag = next_tag(q_visit_tag_, q_visit_marks_);
	entryHeap pqEntries;
	std::priority_queue<Res> candTable;
	Res res_pair;

	q->UB = (int)N / 10;
	int lshUB = N / 200;
	lshUB = 4 * L * log(N);
	int step = 1;
	if(_lsh_UB>0) lshUB=_lsh_UB;
	std::vector<int> numAccess(L);
	std::vector<std::multimap<zint, int>::iterator> lpos(L), rpos(L), qpos(L);
	std::priority_queue<posInfo> lEntries, rEntries;
	std::vector<zint> keys(L);
	for (int j = 0; j < L; j++) {
		keys[j] = getZ(q->hashval + j * K);
		qpos[j] = hashTables[j].lower_bound(keys[j]);
		if (qpos[j] != hashTables[j].begin()) {
			lpos[j] = qpos[j];
			--lpos[j];
#ifdef USE_LCCP
			lEntries.push(posInfo(j, getLLCP(lpos[j]->first, keys[j])));
#else
			lEntries.push(posInfo(j, getLevel(lpos[j]->first, qpos[j]->first)));
#endif // USE_LCCP

		}
		//
		rpos[j] = qpos[j];
		if (rpos[j] != hashTables[j].end()) {
#ifdef USE_LCCP
			rEntries.push(posInfo(j, getLLCP(rpos[j]->first, keys[j])));
#else
			rEntries.push(posInfo(j, getLevel(rpos[j]->first, qpos[j]->first)));
#endif // USE_LCCP
		}
	}

	// Solmaz (Idea2): collision-aware start based on exact bucket collisions.
	// We collect IDs colliding with the query across tables and use the counts to choose better entry points
	// FROM the normal LSH candidates (hybrid), rather than replacing the candidate generator.
	std::unordered_map<int, int> aware_cnt;
	if (use_collision_aware_start) {
		// Lock hash tables for safe concurrent reads under dynamic updates.
		read_lock lock_hr(hash_lock);
		std::vector<read_lock> lock_hs;
		lock_hs.reserve((size_t)L);
		for (int i = 0; i < L; ++i) {
			lock_hs.emplace_back(read_lock(hash_locks_[i]));
		}

		aware_cnt.reserve((size_t)aware_rerank_topN_by_dist * 2);
		for (int j = 0; j < L; ++j) {
			auto range = hashTables[j].equal_range(keys[j]);
			int scanned = 0;
			for (auto it = range.first; it != range.second; ++it) {
				if (aware_bucket_cap_per_table > 0 && scanned++ >= aware_bucket_cap_per_table) break;
				const int pid = it->second;
				if (pid < 0 || pid >= N) continue;
				if (isDeleted(pid)) continue;
				++aware_cnt[pid];
			}
		}
	}

	// Original LSH scan (fallback / fill-up). If collision-aware seeding already produced candidates,
	// this still runs but will mostly skip already-seeded IDs via flag_ checks.
	while (!(lEntries.empty() && rEntries.empty())) {
		posInfo t;
		bool f = true;//TRUE:left; FALSE:right
		if (lEntries.empty()) f = false;
		else if (rEntries.empty()) f = true;
		else if (rEntries.top().dist > lEntries.top().dist) f = false;

		if (f) {
			t = lEntries.top();
			lEntries.pop();
			for (int i = 0; i < step; ++i) {
				++numAccess[t.id];
				res_pair.id = lpos[t.id]->second;
				if (res_pair.id >= 0 && res_pair.id < N && q_cand_marks_[(size_t)res_pair.id] != cand_tag) {
					res_pair.dist = cal_dist(q->queryPoint, q->myData[res_pair.id], dim);
					candTable.push(res_pair);
					q_cand_marks_[(size_t)res_pair.id] = cand_tag;
				}
				if (lpos[t.id] != hashTables[t.id].begin()) {
					--lpos[t.id];
				}
				else {
					break;
				}
			}

			if (lpos[t.id] != hashTables[t.id].begin()) {
#ifdef USE_LCCP
				t.dist = getLLCP(lpos[t.id]->first, keys[t.id]);
#else
				t.dist = getLevel(lpos[t.id]->first, qpos[t.id]->first);
#endif // USE_LCCP
				lEntries.push(t);
			}
		}
		else {
			t = rEntries.top();
			rEntries.pop();
			for (int i = 0; i < step; ++i) {
				++numAccess[t.id];
				res_pair.id = rpos[t.id]->second;
				if (res_pair.id >= 0 && res_pair.id < N && q_cand_marks_[(size_t)res_pair.id] != cand_tag) {
					res_pair.dist = cal_dist(q->queryPoint, q->myData[res_pair.id], dim);
					candTable.push(res_pair);
					q_cand_marks_[(size_t)res_pair.id] = cand_tag;
				}
				if (++rpos[t.id] == hashTables[t.id].end()) {
					break;
				}
			}
			if (rpos[t.id] != hashTables[t.id].end()) {
#ifdef USE_LCCP
				t.dist = getLLCP(rpos[t.id]->first, keys[t.id]);
#else
				t.dist = getLevel(rpos[t.id]->first, qpos[t.id]->first);
#endif // USE_LCCP
				rEntries.push(t);
			}
		}
		if (candTable.size() >= lshUB) break;
	}



	q->cost = candTable.size();
	while (candTable.size() > ef) candTable.pop();

	q->timeHash = timer.elapsed();
	timer.restart();
	if (candTable.empty()) {
		candTable.emplace(0, cal_dist(q->queryPoint, myData[0], dim));
	}

	// Materialize the LSH candidates.
	std::vector<Res> seeds;
	seeds.reserve(candTable.size());
	while (!candTable.empty()) {
		auto u = candTable.top();
		seeds.push_back(u);
		q->resHeap.push(u);
		candTable.pop();
	}
	if (q->resHeap.empty()) {
		Res u;
		u.id = 0;
		u.dist = cal_dist(q->queryPoint, myData[0], dim);
		seeds.push_back(u);
		q->resHeap.push(u);
	}
	q->minKdist = q->resHeap.top().dist;

	// Anchor warm-start helper: tiny local best-first expansion from a start node to pick a better entry.
	// Keeps cost low by capping expansions and neighbor fanout.
	auto warm_refine_from = [&](int startId) -> Res {
		if (startId < 0 || startId >= N) return Res(0, cal_dist(q->queryPoint, myData[0], dim));
		if (isDeleted(startId)) return Res(0, cal_dist(q->queryPoint, myData[0], dim));
		int bestId = startId;
		float bestDist = cal_dist(q->queryPoint, myData[startId], dim);

		const int expansions = std::max(0, anchor_warmup_expansions);
		if (!anchor_warm_start || expansions == 0) {
			// Optional 1-hop refinement fallback (cheap).
			if (anchor_local_refine &&
				bestId >= 0 && bestId < (int)linkLists.size() &&
				linkLists[bestId] != nullptr) {
				const int outN = linkLists[bestId]->out;
				const int cap = std::max(0, anchor_refine_degree_cap);
				const int lim = (cap > 0) ? std::min(cap, outN) : outN;
				for (int p = 0; p < lim; ++p) {
					const int nb = linkLists[bestId]->neighbors[p].id;
					if (nb < 0 || nb >= N) continue;
					if (isDeleted(nb)) continue;
					const float dnb = cal_dist(q->queryPoint, myData[nb], dim);
					if (dnb < bestDist) { bestDist = dnb; bestId = nb; }
				}
			}
			return Res(bestId, bestDist);
		}

		// Warm-start local best-first (priority by distance to query).
		using Pair = std::pair<float, int>;
		std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> pq;
		std::unordered_set<int> visited;
		visited.reserve((size_t)expansions * 4);
		pq.emplace(bestDist, startId);

		const int degCap = std::max(0, anchor_warmup_degree_cap);
		int pops = 0;
		while (!pq.empty() && pops < expansions) {
			auto [dcur, u] = pq.top();
			pq.pop();
			if (!visited.insert(u).second) continue;
			++pops;
			if (dcur < bestDist) { bestDist = dcur; bestId = u; }

			if (u < 0 || u >= (int)linkLists.size()) continue;
			Node2* node = linkLists[u];
			if (!node) continue;
			const int outN = node->out;
			const int lim = (degCap > 0) ? std::min(degCap, outN) : outN;
			for (int p = 0; p < lim; ++p) {
				const int nb = node->neighbors[p].id;
				if (nb < 0 || nb >= N) continue;
				if (isDeleted(nb)) continue;
				if (visited.find(nb) != visited.end()) continue;
				const float dnb = cal_dist(q->queryPoint, myData[nb], dim);
				pq.emplace(dnb, nb);
			}
		}
		return Res(bestId, bestDist);
	};

	// Choose entry points for expansion.
	if (!use_collision_aware_start && !use_anchor_start) {
		for (const auto& u : seeds) {
			pqEntries.push(std::make_pair(u, 1));
		}
		// Keep only the best entry point by default (legacy behavior),
		// but allow multi-start for pseudo traversal.
		const int m = (use_pseudo_search && pseudo_entry_count > 1)
			? std::min<int>(pseudo_entry_count, (int)pqEntries.size())
			: 1;
		entryHeap tempPQ;
		for (int i = 0; i < m && !pqEntries.empty(); ++i) {
			tempPQ.emplace(pqEntries.top());
			pqEntries.pop();
		}
		pqEntries.swap(tempPQ);
	}
	else if (use_collision_aware_start) {
		// Sort seeds by distance ascending.
		std::sort(seeds.begin(), seeds.end(), [](const Res& a, const Res& b) { return a.dist < b.dist; });
		const int topN = std::max(1, std::min((int)seeds.size(), aware_rerank_topN_by_dist));
		const int m = std::max(1, std::min(aware_entry_count, topN));

		// Smooth hybrid-aware scoring path:
		// - Prefer seeds that collide across many tables, but still reward closeness to the query
		// - This reduces the "zig-zag" (discrete switching) behavior
		//
		// Optional hard filter: require a minimum collision count if aware_min_collisions > 0.
		int minColl = aware_min_collisions;
		if (minColl > L) minColl = L;

		const float denomL = (float)std::max(1, L);
		const float denomRank = (float)std::max(1, topN - 1);

		std::vector<int> idx;
		idx.reserve((size_t)topN);
		for (int i = 0; i < topN; ++i) {
			int c = 0;
			auto itc = aware_cnt.find(seeds[i].id);
			if (itc != aware_cnt.end()) c = itc->second;
			if (minColl > 0 && c < minColl) continue;
			idx.push_back(i);
		}
		if ((int)idx.size() < m) {
			// Not enough seeds pass the hard filter -> remove the filter (fallback to full topN).
			idx.clear();
			for (int i = 0; i < topN; ++i) idx.push_back(i);
		}

		auto score_of = [&](int distRankIdx) -> float {
			const int pid = seeds[distRankIdx].id;
			int c = 0;
			auto itc = aware_cnt.find(pid);
			if (itc != aware_cnt.end()) c = itc->second;
			const float collNorm = (float)c / denomL;            // [0,1]
			const float rankNorm = (float)distRankIdx / denomRank; // [0,1]
			return aware_alpha * collNorm - aware_beta * rankNorm;
		};

		std::partial_sort(idx.begin(), idx.begin() + std::min<int>(m, (int)idx.size()), idx.end(),
			[&](int ia, int ib) {
				const float sa = score_of(ia);
				const float sb = score_of(ib);
				if (sa != sb) return sa > sb;
				return seeds[ia].dist < seeds[ib].dist;
			});

		for (int i = 0; i < m && i < (int)idx.size(); ++i) {
			pqEntries.push(std::make_pair(seeds[idx[i]], 1));
		}
	}
	else { // use_anchor_start
		// If anchors are missing, fall back to the best seed.
		if (anchor_ids.empty()) {
			pqEntries.push(std::make_pair(seeds.front(), 1));
		}
		else {
			const int m = std::max(1, std::min(anchor_entry_count, (int)anchor_ids.size()));
			std::vector<Res> ares;
			ares.reserve(anchor_ids.size());
			for (int id : anchor_ids) {
				if (id < 0 || id >= N) continue;
				if (isDeleted(id)) continue;
				float dist = cal_dist(q->queryPoint, myData[id], dim);
				ares.emplace_back(id, dist);
			}
			if (ares.empty()) {
				pqEntries.push(std::make_pair(seeds.front(), 1));
			}
			else {
				std::partial_sort(ares.begin(), ares.begin() + std::min<int>(m, (int)ares.size()), ares.end(),
					[](const Res& a, const Res& b) { return a.dist < b.dist; });
				// Warm-start refine each chosen anchor into a better entry.m
				std::vector<Res> refined;
				refined.reserve((size_t)m);
				std::unordered_set<int> seen;
				seen.reserve((size_t)m * 2);
				for (int i = 0; i < m && i < (int)ares.size(); ++i) {
					Res r = warm_refine_from(ares[i].id);
					if (seen.insert(r.id).second) refined.emplace_back(r.id, r.dist);
				}
				std::sort(refined.begin(), refined.end(), [](const Res& a, const Res& b) { return a.dist < b.dist; });
				for (const auto& r : refined) pqEntries.push(std::make_pair(r, 1));
			}
		}
	}

	// Solmaz (Idea1B-B1): learned anchor start takes priority over anchor_start if enabled.
	// (We put it last to keep seed/resHeap initialization identical.)
	if (use_learned_anchor_start) {
		// Preconditions: anchors loaded AND model loaded.
		if (anchor_ids.empty() || learned_anchor_num_classes <= 0 || learned_anchor_dim != dim ||
			(int)anchor_ids.size() != learned_anchor_num_classes) {
			// fall back: do nothing (keep whatever pqEntries already has)
		}
		else {
			// Clear existing pqEntries; use learned anchors as the entry points.
			entryHeap empty;
			pqEntries.swap(empty);

			// Compute logits = W x + b (no softmax needed for ranking).
			const int C = learned_anchor_num_classes;
			std::vector<std::pair<float, int>> scores;
			scores.reserve((size_t)C);

			// Optional: L2-normalize query for stability (matches common linear models).
			float norm2 = 0.0f;
			for (int d = 0; d < dim; ++d) norm2 += q->queryPoint[d] * q->queryPoint[d];
			const float inv_norm = (norm2 > 1e-12f) ? (1.0f / std::sqrt(norm2)) : 1.0f;

			for (int c = 0; c < C; ++c) {
				const float* wrow = &learned_anchor_W[(size_t)c * (size_t)dim];
				float s = learned_anchor_b[(size_t)c];
				for (int d = 0; d < dim; ++d) {
					s += wrow[d] * (q->queryPoint[d] * inv_norm);
				}
				scores.emplace_back(s, c);
			}

			const int m = std::max(1, std::min(learned_anchor_entry_count, C));
			std::partial_sort(scores.begin(), scores.begin() + m, scores.end(),
				[](const auto& a, const auto& b) { return a.first > b.first; });

			// Warm-start refine each predicted anchor into a better entry.
			std::vector<Res> refined;
			refined.reserve((size_t)m);
			std::unordered_set<int> seen;
			seen.reserve((size_t)m * 2);
			for (int i = 0; i < m; ++i) {
				const int cls = scores[i].second;
				const int pid0 = anchor_ids[(size_t)cls];
				if (pid0 < 0 || pid0 >= N) continue;
				if (isDeleted(pid0)) continue;
				Res r = warm_refine_from(pid0);
				if (seen.insert(r.id).second) refined.emplace_back(r.id, r.dist);
			}
			std::sort(refined.begin(), refined.end(), [](const Res& a, const Res& b) { return a.dist < b.dist; });
			for (const auto& r : refined) pqEntries.push(std::make_pair(r, 1));
		}
	}

	if (use_ligs_collision_search) bestFirstSearchLIGSCollision(q, q_visit_marks_, visit_tag, pqEntries);
	else if (use_pseudo_search) bestFirstSearchPseudo(q, q_visit_marks_, visit_tag, pqEntries);
	else bestFirstSearchInGraph(q, q_visit_marks_, visit_tag, pqEntries);
	q->timeSift = timer.elapsed();
	q->timeTotal = q->timeHash + q->timeSift;
	q->costs = numAccess;

	//delete[] q->hashval;
}

void divGraph::knnHNSW(queryN* q)
{
	lsh::timer timer;
	q->hashval = calHash(q->queryPoint);
#ifdef USE_SSE
	_mm_prefetch((char*)(q->queryPoint), _MM_HINT_T0);
#endif
	timer.restart();
	// Solmaz (Idea3): fast visited marking (no O(N) string reset).
	ensure_query_marks();
	const uint32_t visit_tag = next_tag(q_visit_tag_, q_visit_marks_);
	std::priority_queue<std::pair<Res, int>, std::vector<std::pair<Res, int>>, std::greater<std::pair<Res, int>>> pqEntries;
	//std::priority_queue<Res> candTable;
	Res res_pair;

	//ef = q->k;
	//ef += 200;
	std::priority_queue<Res, std::vector<Res>, std::greater<Res>> eps;
	int ep0 = 0;
	q_visit_marks_[(size_t)ep0] = visit_tag;
	float dist = cal_dist(q->queryPoint, myData[ep0], dim);
	//float lowerBound = dist;
	eps.emplace(dist, ep0);
	q->resHeap.emplace(dist, ep0);
	pqEntries.push(std::make_pair(Res(dist, ep0), 1));

	//bestFirstSearchInGraph2(q, flag_, visitedDists, pqEntries);
	//bestFirstSearchInGraphHNSW(q, flag_, visitedDists, pqEntries);
	if (use_ligs_collision_search) bestFirstSearchLIGSCollision(q, q_visit_marks_, visit_tag, pqEntries);
	else if (use_pseudo_search) bestFirstSearchPseudo(q, q_visit_marks_, visit_tag, pqEntries);
	else bestFirstSearchInGraph(q, q_visit_marks_, visit_tag, pqEntries);
	q->timeSift = timer.elapsed();
	q->timeTotal = q->timeHash + q->timeSift;
}

void divGraph::bestFirstSearchInGraph(queryN * q, std::vector<uint32_t>& visited, uint32_t vtag, entryHeap & pqEntries)
{
	while (!pqEntries.empty()) {
		auto u = pqEntries.top().first;
		if (u.dist > q->minKdist) {
			break;
		}
		int hop = pqEntries.top().second;
		pqEntries.pop();
		q->maxHop = q->maxHop > hop ? q->maxHop : hop;
		Res* nns = linkLists[u.id]->neighbors;
#ifdef USE_SSE
		//_mm_prefetch((char*)&(stateFlags[u.id]), _MM_HINT_T0);
		//_mm_prefetch((char*)(&(stateFlags[u.id]) + 64), _MM_HINT_T0);
		//_mm_prefetch(links + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
		_mm_prefetch((char*)(myData[nns[0].id]), _MM_HINT_T0);
		_mm_prefetch((char*)(myData[nns[1].id]), _MM_HINT_T0);
#endif

		// Speed: only scan actual out-degree rather than maxT slots.
		const int outN = linkLists[u.id]->out;
		for (int pos = 0; pos < outN; ++pos) {
			int v = (*(linkLists[u.id]))[pos];
			if (v < 0) continue;
			if (isDeleted(v)) continue;
			if (visited[(size_t)v] != vtag) {
				visited[(size_t)v] = vtag;
				if (0 || cal_dist(q->hashval, hashval[v], lowDim) * coeffq < q->minKdist) {
					float dist = cal_dist(q->queryPoint, myData[v], dim);
					++q->cost;
					if (false || dist < q->minKdist
						//|| visitedDists[v] < u.dist
						) {
						pqEntries.push(std::make_pair(Res(v, dist), hop + 1));
						q->resHeap.push(Res(v, dist));
						if (q->resHeap.size() > ef) {
							q->resHeap.pop();
							q->minKdist = q->resHeap.top().dist;
						}
					}
				}
				else {
					q->prunings++;
				}
			}
		}
	}

	while (q->resHeap.size() > q->k) q->resHeap.pop();
	// Solmaz: Ensure uniqueness of returned IDs and be robust if collision sets provide few candidates.
	std::unordered_map<int, float> bestDist;
	bestDist.reserve((size_t)ef * 2);
	while (!q->resHeap.empty()) {
		Res r = q->resHeap.top();
		q->resHeap.pop();
		auto it = bestDist.find(r.id);
		if (it == bestDist.end() || r.dist < it->second) bestDist[r.id] = r.dist;
	}

	std::vector<Res> uniq;
	uniq.reserve(bestDist.size());
	for (const auto& kv : bestDist) {
		Res r;
		r.id = kv.first;
		r.dist = kv.second;
		uniq.emplace_back(r);
	}
	std::sort(uniq.begin(), uniq.end(), [](const Res& a, const Res& b) { return a.dist < b.dist; });

	const int out_k = std::min<int>((int)uniq.size(), (int)q->k);
	q->res.assign(uniq.begin(), uniq.begin() + out_k);
}

// Solmaz: Pseudo-graph best-first search (Algorithm-1 style).
// Differences vs bestFirstSearchInGraph:
// - Does NOT apply the (hashval * coeffq < minKdist) pruning step.
// - Keeps the same ef-based result heap handling, so it can be compared fairly.
void divGraph::bestFirstSearchPseudo(queryN* q, std::vector<uint32_t>& visited, uint32_t vtag, entryHeap& pqEntries)
{
	while (!pqEntries.empty()) {
		auto u = pqEntries.top().first;
		if (u.dist > q->minKdist) {
			break;
		}
		int hop = pqEntries.top().second;
		pqEntries.pop();
		q->maxHop = q->maxHop > hop ? q->maxHop : hop;

		// Speed: only scan actual out-degree rather than maxT slots.
		const int outN = linkLists[u.id]->out;
		for (int pos = 0; pos < outN; ++pos) {
			int v = (*(linkLists[u.id]))[pos];
			if (v < 0) continue;
			if (isDeleted(v)) continue;

			if (visited[(size_t)v] == vtag) continue;
			visited[(size_t)v] = vtag;

			float dist = cal_dist(q->queryPoint, myData[v], dim);
			++q->cost;

			// Same ef handling as the original: keep a candidate set of size ef.
			if (dist < q->minKdist || q->resHeap.size() < (size_t)ef) {
				pqEntries.push(std::make_pair(Res(v, dist), hop + 1));
				q->resHeap.push(Res(v, dist));
				if (q->resHeap.size() > (size_t)ef) {
					q->resHeap.pop();
					q->minKdist = q->resHeap.top().dist;
				}
			}
		}
	}

	while (q->resHeap.size() > q->k) q->resHeap.pop();
	q->res.resize(q->k);
	for (int i = q->k - 1; i > -1; --i) {
		q->res[i] = q->resHeap.top();
		q->resHeap.pop();
	}
}

// Solmaz Seyed Monir: DAPG Algorithm-1 collision-set expansion.
// Differences vs bestFirstSearchPseudo:
// - Expands LSH collision sets C_i(u) via hashTables[i].equal_range(key_i(u)) rather than graph neighbors.
// - Keeps the same ef-based result heap handling for comparability.
void divGraph::bestFirstSearchLIGSCollision(queryN* q, std::vector<uint32_t>& visited, uint32_t vtag, entryHeap& pqEntries)
{
	if (L <= 0 || K <= 0) {
		// Fallback to the pseudo neighbor traversal if LSH tables are not available.
		bestFirstSearchPseudo(q, visited, vtag, pqEntries);
		return;
	}

	std::vector<zint> ukeys(L);

	while (!pqEntries.empty()) {
		auto u = pqEntries.top().first;
		if (u.dist > q->minKdist) {
			break;
		}
		int hop = pqEntries.top().second;
		pqEntries.pop();
		q->maxHop = q->maxHop > hop ? q->maxHop : hop;

		// Compute per-table keys for the popped node u.
		for (int j = 0; j < L; ++j) {
			ukeys[j] = getZ(hashval[u.id] + j * K);
		}

		// Expand collision sets across all tables (Algorithm 1, lines 10-17).
		for (int j = 0; j < L; ++j) {
			auto range = hashTables[j].equal_range(ukeys[j]);
			for (auto it = range.first; it != range.second; ++it) {
				int v = it->second;
				if (v < 0) continue;
				if (v == u.id) continue;
				if (isDeleted(v)) continue;
				if (visited[(size_t)v] == vtag) continue;
				visited[(size_t)v] = vtag;

				float dist = cal_dist(q->queryPoint, myData[v], dim);
				++q->cost;

				// Same ef handling as the original: keep a candidate set of size ef.
				if (dist < q->minKdist || q->resHeap.size() < (size_t)ef) {
					pqEntries.push(std::make_pair(Res(v, dist), hop + 1));
					q->resHeap.push(Res(v, dist));
					if (q->resHeap.size() > (size_t)ef) {
						q->resHeap.pop();
						q->minKdist = q->resHeap.top().dist;
					}
				}
			}
		}
	}

	while (q->resHeap.size() > q->k) q->resHeap.pop();
	q->res.resize(q->k);
	for (int i = q->k - 1; i > -1; --i) {
		q->res[i] = q->resHeap.top();
		q->resHeap.pop();
	}
}

void divGraph::showInfo(Preprocess* prep)
{
	float dist_total = 0.0f;
	size_t sqrMat = 0;
	size_t cnt = 0, rec = 0, N1 = 0, cnt1 = 0;
	int f = 1;
	for (int u = 0; u < N; ++u) {
		cnt += linkLists[u]->size();
		sqrMat += linkLists[u]->size() * linkLists[u]->size();
		N1++;
		if (u >= prep->benchmark.N - 200) continue;

		cnt1 += linkLists[u]->size();;
		auto& pt = linkLists[u]->neighbors;
		f = f & isUnique(pt, pt + linkLists[u]->size());
		if (f == 0) {
			printf("***%d\n", u);
			const auto& nns = pt;
			for (int j = 0; j < maxT; ++j) {
				if (j == linkLists[u]->size()) printf("size=%d\n", j);
				printf("%2d: dist=%f, id=%d\n", j, nns[j].dist, nns[j].id);
			}
#ifdef _MSC_VER
			system("pause");
#endif
		}

		for (int pos = 0; pos != linkLists[u]->size(); ++pos) {
			float dist = linkLists[u]->getNeighbor(pos).dist;
#ifdef USE_SQRDIST
			dist_total += sqrt(dist);
#else
			res += dist;
#endif
		}
	}

	float ratio = 0.0f;
	for (int u = 0; u < prep->benchmark.N - 200; ++u) {
		std::set<unsigned> set1, set2;
		std::vector<unsigned> set_intersection;
		set_intersection.clear();
		set1.clear();
		set2.clear();

		int j = 1;
		for (int pos = 0; pos != linkLists[u]->size(); ++pos) {
			set1.insert((*(linkLists[u]))[pos]);
			set2.insert((unsigned)prep->benchmark.indice[u + 200][j]);
			++j;
		}
		std::set_intersection(set1.begin(), set1.end(), set2.begin(), set2.end(),
			std::back_inserter(set_intersection));

		rec += set_intersection.size();
	}

	//return res / cnt;
	float derivation = (float)sqrMat / N1 - ((float)cnt / N1) * ((float)cnt / N1);
	float sigma = sqrt(derivation);
	auto idx = file.rfind('/');
	std::string fname(file.begin(), file.begin() + idx + 1);
	fname += "indexInfo.txt";
	FILE* fp = nullptr;
	fopen_s(&fp, fname.c_str(), "a");


	printf("dist=%f, cnt=%f, unique=%d, std=%f, Recall=%f\ncc=%f, pruning=%f\n\n", dist_total / cnt1, (float)cnt / N1, (int)f, sigma, (float)rec / cnt1,
		(float)compCostConstruction / N, (float)pruningConstruction / compCostConstruction);
	if (fp) fprintf(fp, "%s\nT=%d,L=%d,K=%d\ndist=%f, cnt=%f, unique=%d, std=%f, Recall=%f\ncc=%f, pruning=%f, IndexingTime=%f s.\n\n", file.c_str(), T, L, K, dist_total / cnt1, (float)cnt / N1, (int)f, sigma, (float)rec / cnt1,
		(float)compCostConstruction / N, (float)pruningConstruction / compCostConstruction, indexingTime);

	//if (compCostConstruction == 0) {
	//	printf("dist=%f, cnt=%f, unique=%d, Dcnt=%f, Recall=%f\n\n", dist_total / cnt1, (float)cnt / N1, (int)f, sigma, (float)rec / cnt1);
	//	if (fp) fprintf(fp, "dist=%f, cnt=%f, unique=%d, Dcnt=%f, Recall=%f\ncc=%f, pruning=%f, IndexingTime=%f s.\n\n", dist_total / cnt1, (float)cnt / N1, (int)f, sigma, (float)rec / cnt1,
	//		(float)compCostConstruction / N, (float)pruningConstruction / compCostConstruction, indexingTime);
	//}
	//else {
	//	printf("dist=%f, cnt=%f, unique=%d, Dcnt=%f, Recall=%f\ncc=%f, pruning=%f\n\n", dist_total / cnt1, (float)cnt / N1, (int)f, sigma, (float)rec / cnt1,
	//		(float)compCostConstruction / N, (float)pruningConstruction / compCostConstruction);
	//	if (fp) fprintf(fp, "dist=%f, cnt=%f, unique=%d, Dcnt=%f, Recall=%f\ncc=%f, pruning=%f, IndexingTime=%f s.\n\n", dist_total / cnt1, (float)cnt / N1, (int)f, sigma, (float)rec / cnt1,
	//		(float)compCostConstruction / N, (float)pruningConstruction / compCostConstruction, indexingTime);
	//}
}

static int connectivity(std::vector<std::vector<int>>& us)
{
	int N = us.size();
	std::vector<int> unions(N);
	int flag = 0;
	int ef = 0;
	int cnt = 0;
	int last_zero = 0;

	std::unordered_set<size_t> uniques;

	while (cnt < N) {
		flag++;
		for (int i = last_zero; i < N; ++i) {
			if (unions[i] == 0) {
				last_zero = i;
				break;
			}
		}
		ef = last_zero;
		last_zero++;
		unions[ef] = flag;
		cnt++;
		std::priority_queue<int> qs;
		qs.push(ef);
		while (qs.size()) {
			ef = qs.top();
			qs.pop();
			for (auto& v : us[ef]) {
				if (unions[v] == 0) {
					unions[v] = flag;
					qs.push(v);
					cnt++;
				}
				else if (unions[v] < flag) printf("alg Error!\n");

			}
		}
	}

	return flag;
}

void divGraph::traverse()
{
	std::vector<int> unions(N);
	int flag = 0;
	int ef = 0;
	int cnt = 0;
	int last_zero = 0;
	
	std::unordered_set<size_t> uniques;

	while (cnt < N) {
		flag++;
		for (int i = last_zero; i < N; ++i) {
			if (unions[i] == 0) {
				last_zero = i;
				break;
			}
		}
		ef = last_zero;
		last_zero++;
		unions[ef] = flag;
		cnt++;
		std::priority_queue<int> qs;
		qs.push(ef);
		while (qs.size()) {
			ef = qs.top();
			qs.pop();
			auto& nns = linkLists[ef]->neighbors;
			int size = linkLists[ef]->out;
			for (int i = 0; i < size; ++i) {
				if (unions[nns[i].id] == 0) {
					unions[nns[i].id] = flag;
					qs.push(nns[i].id);
					cnt++;
				}

				else if (unions[nns[i].id] < flag) {
					//flag--;
					uniques.insert(((size_t)N) * unions[nns[i].id] + flag);
				}
			}
		}
	}

	std::vector<std::vector<int>> us(flag + 1);
	us[0].push_back(1);
	us[1].push_back(0);
	for (auto& x : uniques) {
		size_t u = x % N;
		size_t v = x / N;
		us[u].push_back(v);
		us[v].push_back(u);
	}

	flag = connectivity(us);

	printf("The union number is:%d\n", flag);
}


void divGraph::save(const std::string & file)
{

	std::ofstream out(file, std::ios::binary);
	/*****************************LSH************************/
	out.write((char*)&N, sizeof(int));
	out.write((char*)&dim, sizeof(int));
	out.write((char*)&L, sizeof(int));
	out.write((char*)&K, sizeof(int));
	out.write((char*)&W, sizeof(float));
	out.write((char*)&u, sizeof(int));

	//hashval
	for (int i = 0; i < N; ++i) {
		out.write((char*)(hashval[i]), sizeof(float) * S);
	}

	//hashpar,hashmin,hashmax
	out.write((char*)hashPar.rndBs, sizeof(float) * S);
	out.write((char*)&hashMins[0], sizeof(float) * S);
	out.write((char*)&hashMaxs[0], sizeof(float) * S);
	for (int i = 0; i != S; ++i) {
		out.write((char*)hashPar.rndAs[i], sizeof(float) * dim);
	}

	// Solmaz Seyed Monir: Hash table serialization (backward-compatible with older indexes).
	// New format writes a magic + per-table entry counts so partial builds are supported.
	{
		const uint32_t MAGIC_HTBL = 0x4C425448; // "HTBL" in little-endian
		out.write((char*)&MAGIC_HTBL, sizeof(uint32_t));
		for (int i = 0; i < L; ++i) {
			uint64_t sz = (uint64_t)hashTables[i].size();
			out.write((char*)&sz, sizeof(uint64_t));
		}
		for (int i = 0; i < L; ++i) {
			for (auto iter = hashTables[i].begin(); iter != hashTables[i].end(); ++iter) {
				out.write((char*)&(iter->first), sizeof(zint));
				out.write((char*)&((iter->second)), sizeof(int));
			}
		}
	}

	/*****************************graph************************/

	out.write((char*)&T, sizeof(int));
	out.write((char*)&step, sizeof(int));
	out.write((char*)&nnD, sizeof(int));
	out.write((char*)&edgeTotal, sizeof(size_t));
	out.write((char*)&lowDim, sizeof(int));
	//out.write((char*)&dataSize, sizeof(int));
	//out.write((char*)&dim, sizeof(int));
	int len = flagStates.size();
	out.write((char*)&len, sizeof(int));
	out.write((char*)(flagStates.c_str()), sizeof(char) * len);

	len = linkLists.size();
	out.write((char*)&len, sizeof(int));
	for (int i = 0; i < len; ++i) {
		linkLists[i]->writeToFile(out);
	}
	// Solmaz: Persist indexingTime so that isbuilt=1 runs can report IndexingTime_s.
	// Backward compatible: older index files simply won't have these trailing bytes.
	out.write((char*)&indexingTime, sizeof(float));
	out.close();
}
















