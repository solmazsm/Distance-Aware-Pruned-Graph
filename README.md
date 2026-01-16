# DAPG: Distance-Aware Pruned Graph

## Anonymous review artifact. This repository contains the reference implementation of DAPG, a single-layer, geometry-aware ANN index introduced in the paper.


## **ABSTRACT**

DAPG introduces percentile-based local filtering and adaptive global sparsification to build degree-adaptive proximity graphs that preserve reachability while reducing redundant edges.  
DAPG improves latency–recall trade-offs over **state-of-the-art (SOTA)** baselines without multi-layer indexing.

<p>
  <kbd>+3.3% recall</kbd>
  <kbd>2.9× faster</kbd>
  <kbd>Single layer</kbd>
  <kbd>LSH seeding</kbd>
</p>

---
## Why This Work

### What existing ANN methods miss — and how **DAPG** fixes it

|  Common gaps in existing ANN methods |  How **DAPG** addresses these |
| :----------------------------------- | :------------------------------ |
| • **Fixed-degree graphs:** Static degree limits prevent adaptive sparsification. |  **Adaptive sparsity:** percentile-based local filtering and global capping balance recall vs. cost. |
| • **Costly rebuilds:** Traditional structures require full index reconstruction for new data. |  **Incremental updates:** supports amortized insertion with bounded rewire cost. |
| • **Uncontrolled expansion:** Greedy traversal often expands redundant nodes. |  **Pruned search:** expansion is capped by β(ℓ), maintaining efficiency. |
| • **Weak theoretical link:** Prior heuristics lack formal sublinear complexity bounds. |  **Theory-backed:** Lemma 3 + Lemma 2 prove sublinear query complexity O(d̄<sub>DAPG</sub> β(ℓ)) and bounded connectivity. |

> **Result:** Higher recall (+3.3%), 2.9× lower query latency, and up to 10× smaller memory footprint—while preserving the same O(d̄<sub>DAPG</sub> β(ℓ)) complexity.

---
## Contributions

## What We Bring
<table> <tr> <td width="48%" valign="top">

1) Theory
Formalizes distance-aware pruning and adaptive degree control in proximity graphs, providing probabilistic bounds on reachability and connectivity under percentile-based sparsification.

</td> <td width="48%" valign="top">

2) Method
DAPG introduces local percentile filtering (P<sub>local</sub>) and global capping (P<sub>global</sub>) to construct degree-adaptive graphs that minimize redundant edges while maintaining recall.

</td> </tr> <tr> <td width="48%" valign="top">

3) Empirics
Outperforms LSH-APG across DEEP1M, MNIST, and SIFT1M, achieving up to +3.3% recall and 2.9× lower query latency, while reducing graph density and index memory.

</td> <td width="48%" valign="top">

4) Guidance
Provides design rules for local/global pruning rates, ensuring consistent recall stability under dynamic insertions and scalable O(d C<sub>Q</sub>) maintenance.

</td> </tr> </table>


## INTRODUCTION
This repository provides the source code for **DAPG**, a novel graph-based Approximate Nearest Neighbor (ANN) indexing system introduced in the paper.

---


## What DAPG Adds - DISTANCE-AWARE PRUNED GRAPH FRAMEWORK

**Distance-Aware Local Pruning (percentile threshold per node).**

**Adaptive Global Sparsification (cap high-degree tails.**

**Parallel construction with thread-safe updates.**

**Serialization of graph + LSH structures for fast reloads.**

---
## **Cost Model**

**Local percentile threshold  + adaptive cap <code>T'</code>.**
**LSH seeding; no hierarchy.**

**Supports dynamic insertion and deletion with degree-stable connectivity.**

Expected query cost factorizes into average degree and expansion depth.

<pre>
C_Q = O(&macr;d_DAPG · β(ℓ))
T_Q = O(d · &macr;d_DAPG · β(ℓ))
β(ℓ) = O(log n) under small-world routing
</pre>

**Cost model: <code>C<sub>Q</sub> = O(&macr;d<sub>DAPG</sub>&nbsp;β(ℓ))</code>.**

---



## Compilation

The code is implemented in **C++11** and supports parallelism using **OpenMP**. It can be compiled on both Linux and Windows.

### Linux

Linux (g++ / clang++)

Requires C++11 and OpenMP.

The provided Makefile auto-detects -fopenmp. If your toolchain differs, edit cppCode/DAPG/Makefile.

```bash
cd ./cppCode/DAPG
make
```

###  Windows

Use **Visual Studio 2019+** to import the project located in:

```
./cppCode/DAPG/src/
```

Make sure to enable OpenMP and C++11 support in the build settings.

## Running DAPG

### Command Format

```bash
./dapg datasetName
```

- `datasetName`: The name of the dataset (e.g., `sift`, `mnist`, `audio`)

### Example

```bash
cd ./cppCode/DAPG
./dapg sift
```

This runs DAPG index construction and search on the `sift` dataset.

## Key Features

-  Distance-Aware Local Pruning: Adapts edge filtering based on local percentile distances.
-  Sparsity Control: Limits node degree while preserving connectivity in sparse regions.
-  Improved Recall–Latency Tradeoff: Reduces query time without degrading recall.
-  Compatible with ANN frameworks.
  
## Datasets

We support and have tested DAPG on:

- [Audio](https://github.com/RSIA-LIESMARS-WHU/LSHBOX-sample-data)
- [SIFT1M](http://corpus-texmex.irisa.fr/)
- [Deep1M](https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/deep1M.tar.gz)
- [MNIST](http://yann.lecun.com/exdb/mnist/)
- [SIFT100M](http://corpus-texmex.irisa.fr/)


Convert these into the `.data_new` format for compatibility.
## Dataset Format

The expected input format is a binary file containing float vectors, structured as:

```
{int: float size in bytes}
{int: number of vectors}
{int: dimension}
{float[]: all vector values, stored sequentially}
```

### Example: `sift.data_new`

To use your dataset:

1. Convert it into the binary format shown above.
2. Rename it as `[datasetName].data_new`
3. Place it in: `./dataset/`

A sample dataset (e.g., `audio.data_new`) is already provided.

## EVALUATIONS
### Dataset Geometry and Its Effect on DAPG's Pruning Threshold

DAPG adapts its pruning threshold to each node’s local neighborhood distance distribution, enabling enhanced pruning selectivity on low-LID datasets and more conservative edge retention on high-LID datasets.
This enables the algorithm to preserve essential connectivity on high-LID or heterogeneous datasets, while performing significantly stronger sparsification on low-LID datasets. As a result, DAPG simultaneously achieves higher accuracy and more compact graph construction

| Dataset      | Neighborhood Geometry            | LID | Effect on Local Threshold τᵢ                               | Resulting Graph Size |
|--------------|----------------------------------|-----|-------------------------------------------------------------|-----------------------|
| **Audio**    | Uniform MFCC neighborhoods       | 21.5 | Stable distances → stable τᵢ → consistent pruning           | **Small** |
| **MNIST**    | Pixel manifold; clustered        | 12.7 | Very small τᵢ → strong pruning                              | **Small** |
| **Deep1M**   | Heterogeneous, cosine-based      | 26.0 | Wide distance spread → large τᵢ → keep more edges           | **Larger** |
| **SIFT1M**   | Smooth ℓ₂ structure               | 12.9 | Small–medium τᵢ → remove redundant neighbors                | **Small–Medium** |
| **SIFT100M** | Smooth ℓ₂ structure, large scale | 23.7 | Moderate τᵢ → effective pruning even at scale               | **Smaller** |

## System Setup

Our experiments were conducted on both local and cloud-based environments to evaluate the efficiency and scalability of the DAPG system.

###  Local Workstation
- **Processor**: 13th Gen Intel® Core™ i9-13900HX (24 cores, 32 threads)
- **Base Frequency**: 2.2 GHz  
- **OS**: Ubuntu 20.04 LTS  
- **Precision**: `float32` for all vectors  
- **Implementation**: C++ with multi-threading via OpenMP  
- **Query Setup**: 104 queries per experiment, averaged over 5 independent runs

###  Microsoft Azure Virtual Machines

1. **Standard F32s v2**
   - **vCPUs**: 32  
   - **RAM**: 64 GiB  
   - **Dataset**: SIFT1M  
   - **Purpose**: Scalability evaluation

2. **Standard E64-32s v3 (High-Memory)**
   - **vCPUs**: 32 (Intel® Xeon® Platinum 8272CL)  
   - **RAM**: 432 GiB  
   - **Disk**: 1 TB Premium SSD  
   - **OS**: Ubuntu 22.04 LTS  
   - **Dataset**: SIFT100M  
   - **Purpose**: Large-scale indexing and ANN benchmarking

This high-memory configuration allowed for efficient scaling to large datasets, and multi-threaded execution ensured fast parallel processing during both index construction and query search.
 

#### Distance-Aware Pruning (DAP)
- Introduced a percentile-based thresholding mechanism.
- For each node, computed the 80th percentile distance (τ_q) over LSH candidates.
- Inserted only neighbors with `dist < τ_q` to ensure sparsity and relevance.
- Exposed `last_threshold` for optional diagnostics or debugging.


####  Parallel Construction
- Used `ParallelFor` to insert all nodes in parallel (except the first).
- Enabled thread safety via `std::shared_mutex` for concurrent graph updates.
- Fallbacks to `std::mutex` when C++17 is not available.

#### Serialization
- Graph (`linkLists`) and LSH hash tables are saved to and loaded from a binary format.
- Implemented in `save()` and the constructor `divGraph(Preprocess* prep, ...)`.

## Benchmark Logs

Each row logs detailed metrics:

- **Recall**: Top-k retrieval accuracy
- **Pruning**: Ratio of retained neighbors after DAP-based filtering
- **Time**, **Cost**, and additional performance indicators
- **Algorithm Name**: Includes pruning threshold information (e.g., `DAP_k10_th...`)


- The header includes configuration details such as:  
  `k=20, probQ=0.9, L=2, K=18, T=24`

Each row records:
- `algName`: algorithm configuration with pruning threshold
- `k`: number of neighbors
- `ef`: search parameter
- `Time`: average query time (ms)
- `Recall`: search recall
- `Cost`, `CPQ1`, `CPQ2`: computation cost metrics
- `Pruning`: pruning ratio applied

### Parameter Settings

We evaluate DAP (Distance-Aware Pruning) with:

- k=20, L=2, K=18, T=24, T′=48, W=1.0, pC=0.95, pQ=0.90, efC=80

  
**DAPG computes τ_q per node (default percentile = 80).**


DAP applies **local dynamic pruning**, computing a threshold `τ_q` per node.

---

We evaluate DAPG across a range of:

- `k ∈ {1, 10, 20, ..., 100}`
- `ef` values for query expansion

This allows robust analysis of recall and efficiency across diverse search settings

## Metrics
Each run logs:

Recall@k, Time(ms), Cost, CPQ*, Pruning(%)

algName encodes the pruning threshold (e.g., DAP_k10_th80)

Seed and environment are printed at the top for determinism.


## Complexity and Efficiency Comparison

> **Lemma 3** (supported by **Lemma 2**) proves that DAPG achieves sublinear query complexity  
> **O(d̄<sub>DAPG</sub> β(ℓ))** with bounded degree and probabilistic connectivity,  
> ensuring both **efficiency** and **graph connectivity**.


**DAPG** achieves lower query complexity and higher efficiency than LSH-APG and HNSW-style baselines.  
Its percentile-based local filtering and adaptive global sparsification yield a *degree-adaptive graph*
with reduced average degree while preserving reachability.

- **Build:** 𝒪̃(n d̄_seed) → d̄_DAPG 𝒪(d̄_DAPG β(ℓ))  
- **Query:** 𝒪(d C_Q) amortized per query  
- **No multi-layer structure** → smaller memory footprint and lower update cost  
- **Empirical:** up to **2.9× faster** and **+3.3 % higher recall** than LSH-APG


| Method | Build Complexity | Query Complexity | Notes |
|:-------|:----------------:|:----------------:|:------|
| HNSW | Õ(n d) | O(L d̄ β(ℓ)) | Multi-layer; log n levels |
| LSH-APG | Õ(n d) | O(d̄ β(ℓ)) | Fixed-degree pruning |
| **DAPG (Ours)** | Õ(n d<sub>seed</sub>) → d̄<sub>DAPG</sub> | **O(d̄<sub>DAPG</sub> β(ℓ))** | Local percentile + cap T′ |


- **Fewer edges per node** → less traversal per query  
- **Adaptive pruning** → lower redundancy  
- **Single-layer design** → reduced memory vs. multi-layer HNSW  
- **Dynamic updates** → amortized `O(d C_Q)` maintenance  


| **Dataset** | **Recall@10** | **Query Time (ms)** | **Improvement vs LSH-APG** |
|--------------|---------------|---------------------|-----------------------------|
| **DEEP1M** | 0.9632 vs 0.9590 | 2.30 vs 3.43 | +0.44 % recall  /  32.9 % faster |
| **MNIST** | 0.9984 vs 0.9972 | 0.56 vs 0.68 | +0.12 % recall  /  17.9 % faster |
| **SIFT1M** | 0.9870 vs 0.9580 | 0.83 vs 2.42 | +3.34 % recall  /  2.9× faster |

> DAPG achieves higher recall **and** lower latency on every dataset.


Performance comparison of DAPG vs. LSH-APG on DEEP1M, MNIST, and SIFT1M.
DAPG continues to achieve higher recall and lower query latency than LSH-APG across all datasets, demonstrating consistent scalability and efficiency gains at larger neighborhood sizes.
<table>
<tr>
<td>

<b>Results at k = 10</b>

| Dataset | Method | Recall@10 | Query Time (ms) | Index Size (MB) |
|----------|---------|------------|----------------|----------------|
| **DEEP1M** | LSH-APG | 0.9590 | 3.43 | 250 |
|  | **DAPG (Ours)** | **0.9632 ± 0.0057** | **2.30 ± 0.20** | **449** |
| **MNIST** | LSH-APG | 0.9972 | 0.682 | 10 |
|  | **DAPG (Ours)** | **0.9984 ± 0.0010** | **0.560 ± 0.290** | **27.77** |
| **SIFT1M** | LSH-APG | 0.9580 | 2.42 | 468 |
|  | **DAPG (Ours)** | **0.9870 ± 0.0025** | **0.83 ± 0.00** | **455** |

</td>
<td>

<b>Results at k = 50</b>

| Dataset | Method | Recall@50 | Query Time (ms) | Index Size (MB) |
|----------|---------|------------|----------------|----------------|
| **DEEP1M** | LSH-APG | 0.9590 | 3.43 | 250 |
|  | **DAPG (Ours)** | **0.9615 ± 0.0042** | **2.30 ± 0.22** | **449** |
| **MNIST** | LSH-APG | 0.9972 | 0.682 | 10 |
|  | **DAPG (Ours)** | **0.9984 ± 0.0010** | **0.560 ± 0.290** | **27.77** |
| **SIFT1M** | LSH-APG | 0.9580 | 2.42 | 468 |
|  | **DAPG (Ours)** | **0.9738 ± 0.0028** | **0.83 ± 0.00** | **455** |

</td>
</tr>
</table>

## Comparison

| **Aspect** | **LSH-APG** | **DAPG (Ours)** | **Outcome** |
|:--|:--|:--|:--|
| **Query Complexity** | `O(d β(ℓ))` | **O(d̄<sub>DAPG</sub> β(ℓ)) ≈ O(d C<sub>Q</sub>)** | Same asymptotic form but smaller constant |
| **Degree Control** | Fixed (no adaptivity) | **Adaptive via percentile P<sub>local</sub> + global cap P<sub>global</sub>** | Controllable sparsity |
| **Update Cost** | Rebuild required | **Incremental O(d C<sub>Q</sub>)** (amortized) | dynamic updates |
| **Proof Coverage** | Heuristic | **Formal bounds on reachability & connectivity** | Theoretical Analysis |

Compared to existing ANN frameworks such as HNSW, NSG, DB-LSH, and LSH-APG, DAPG provides the only single-layer hybrid structure with both a formally proven expected query complexity and a probabilistic connectivity guarantee, ensuring sublinear query cost and robust reachability under adaptive sparsification.
> **As shown in Figure 8 (paper)**, DAPG reduces index memory by up to **10×** compared to HNSW and NSG on SIFT100M, while maintaining comparable build time.  
>  
> **DAPG demonstrates higher efficiency** than prior ANN frameworks, reducing query latency by up to **2.9×**, lowering memory footprint by up to **10×**, and maintaining comparable build cost, while preserving the same theoretical complexity  
> **O(d̄<sub>DAPG</sub> β(ℓ))** and achieving consistently higher recall.


## HNSW vs. LSH-APG vs. DAPG


| Component / Feature       | HNSW                                          | LSH-APG                                         | DAPG (Ours)                                                        |
|---------------------------|-----------------------------------------------|-------------------------------------------------|--------------------------------------------------------------------|
| Graph structure           | Multi-layer hierarchical graph                | Single-layer pruned proximity graph             | Single-layer adaptive proximity graph                               |
| Entry point               | Highest layer entry point                     | LSH-selected candidate neighbors                | LSH-selected neighbors + adaptive refinement                        |
| Neighbor selection        | Heuristic: keep closest M                     | Global pruning using T, T′ and p                | Node-wise percentile threshold τᵢ + optional T′                     |
| Pruning rule              | Heuristic prune to limit degree               | Global: p = 95% threshold, fixed cap T′         | Local: percentile τᵢ based on distance distribution                 |
| Adaptivity to density     | Partial (through heuristic)                   | None                                            | Strong (τᵢ depends on local density)                                |
| Search algorithm          | Greedy + efSearch priority queue              | LSH-seeded greedy/best-first on APG graph       | LSH-seeded greedy/best-first on DAPG-pruned graph (Alg. 5)         |
| Construction complexity   | Ō(n d̂) (incremental; ≈ Ō(M log N) per insertion)  | Ō(n d_seed) (batch) | Ō(n d_seed) → d̂_DAPG (batch, adaptive) |
| Degree control            | Controlled by M, M_max                        | Strict global caps T, T′                         | Local adaptive τᵢ + optional global cap                             |
| Index size                | Higher (multiple layers)                      | Moderate                                         | More balanced degrees                           |
| Memory usage              | Higher due to layers                          | Lower                                            | Similar to LSH-APG                                                  |
| Parameters exposed        | M, M_max, efConstruction, efSearch            | K, L, T, T′, p, W                                | + local_percentile; more sparsity-control flexibility                |
| Navigability              | hierarchical small-world                      | single-layer                                      | More uniform due to adaptive sparsification                 |
| Sensitivity to tuning     | Moderate                                      | High (global p, T, T′ must be tuned)            | Lower (τᵢ adapts automatically to dataset)                          |
| Dense regions            | Many neighbors pruned by heuristic            | Over-pruned due to global rule                   | Preserves more neighbors (τᵢ chosen from local distribution)        |
| Sparse areas             | Few connections, but hierarchy helps          | over-prune                                       | Keeps edges (τᵢ expands for sparse nodes)                           |

##  Mapping DAPG Features
| Feature                      | LSH-APG | DAPG (Ours)                            | Algorithm(s) in Paper |
|-----------------------------|---------|--------------------------------------------|------------------------|
| Batch initial construction  | Yes     | Yes                                        | Alg. 1 or Alg. 2       |
| Adaptive local pruning (τᵢ) | No      | Yes (percentile pruning)                   | Alg. 3 (P_local)       |
| Global + local hybrid prune | No      | Yes (τᵢ + T′ refinement)                    | Alg. 3 + Alg. 4        |
| Better balancing of degrees | No      | Yes (local τᵢ + global T′)                | Alg. 3 + Alg. 4        |
| More stable search cost     | No      | Yes (bounded degree from pruning)          | Alg. 3 + Alg. 4        |
| Search procedure            | LSH-seeded greedy/best-first on APG graph    | Greedy Best-First Search (improved graph)  | Alg. 5 (DAPG-Query)    |
| Dynamic update              | No      | **Yes (Dynamic update algorithm)**     | **Alg. 6 (Update)**    |
| Pruned graph construction   | Yes     | Yes, but adaptive and two-stage            | Alg. 1 or Alg. 2       |

| Algorithm        | Category            | Notes                                                                                   |
|------------------|---------------------|-----------------------------------------------------------------------------------------|
| **HNSW**         | Incremental         | Requires hierarchical layers and higher memory, limiting scalability for large dynamic datasets  |
| **LSH-APG**      | Batch               | Single-phase construction with no support for dynamic graph maintenance                |
| **DAPG (Ours)** | **Batch + Dynamic** | Batch construction with efficient dynamic updates through locality-aware pruning |


## Key Result:

## DAPG is lightweight, efficient, and dynamically maintainable, outperforming hierarchical and refinement-based methods in both memory and build time.
### 1. Lightweight (memory-efficient)
> Figure 8 (Index Size):
> DAPG is much smaller on MNIST and SIFT100M, meaning the pruning is effective.
> Figure 8 (Indexing Time):
> DAPG builds faster than HNSW, NSG, and HCNNG, and is faster than LSH-APG on large datasets (DEEP1M, SIFT100M).
> This confirms that two-stage pruning does not increase construction overhead.
 ### 2. Efficient (fast query + fast build)
> Fast query: 
> **Figure 11 (SIFT100M)** demonstrates that DAPG achieves the lowest query latency (≈1.0–1.4 ms) and the highest recall across all k, outperforming all state-of-the-art ANN baselines.


> Fast build: Figure 8 shows that DAPG achieves faster index construction than hierarchical and refinement-based ANN methods, and consistently outperforms LSH-APG on large datasets (DEEP1M and SIFT100M).


> Query Time (DEEP1M):
> **Figure 10** and **Table 5 (Comparison of DAPG and fastG on DEEP1M)**,  
> **DAPG reduces query latency from ~3.4 ms (LSH-APG / fastG) to as low as ~1.3–1.5 ms, providing more than a 2× speedup while also achieving higher recall.**

### 3. Dynamically maintainable

> **Key Distinction: DAPG vs. LSH-APG**
>
> LSH-APG applies a fixed global degree cap **T′** in its construction procedure (Alg. 2), imposing a uniform degree bound that does not adapt to local geometric or statistical variation in the dataset.  
>  
> In contrast, **DAPG** derives a *node-wise*, data-adaptive threshold  
> τᵢ = Percentile_p { δ(vᵢ, vⱼ) ∣ vⱼ ∈ N(vᵢ) } 
> via the **LocalPrune** and **GlobalPrune** operators (Algs. 3–4), and uses this mechanism consistently throughout **DAPG-Index-Construction** (Algs. 1–2), **DAPG-Query** (Alg. 5), and **DAPG-Update** (Alg. 6).  
>  
> Because τᵢ is data-adaptive, DAPG preserves structurally essential neighbors in **high-LID** neighborhoods while performing strong sparsification on **low-LID** neighborhoods, providing higher recall, faster query times, and smaller index sizes than existing APG-based methods, particularly at large scale.



## Query Efficiency and Recall

As shown in **Figure 1**, DAPG consistently achieves lower query latency than LSH-APG across all evaluated datasets, while maintaining comparable or higher recall.

On **Deep1M** with `k = 100`, DAPG achieves a recall of **0.961** with a query time of **2.7 ms**, outperforming LSH-APG, which requires **3.43 ms** to reach similar recall.  
On the **Audio** dataset with `k = 50`, DAPG reaches a recall of **0.9951** in **0.48 ms**, compared to **0.528 ms** for the baseline.

Marker shapes correspond to different values of `k`, and **red dashed lines indicate baseline (LSH-APG) query times**. Overall, these results demonstrate that DAPG serves as an effective and lightweight refinement strategy, improving graph sparsity and query efficiency.

**Figure 1:** [View PDF](https://github.com/solmazsm/Distance-Aware-Pruned-Graph/blob/master/docs/result/figures/query_time_and_recall_vs_ef.pdf)

##  DAPG vs. LSH-APG (Recall–Latency)

As shown in **Figure 2 `dap_vs_lshapg`**, DAPG consistently outperforms the LSH-APG baseline across all neighborhood budgets (NB).  
For example, **DAPG_k=10** and **DAPG_k=20** achieve high recall (≈0.971) with substantially lower query times (**1.378 ms** and **1.508 ms**) than **LSH-APG**, which attains lower recall (<0.96) and higher latency (>1.6 ms).  
Even at larger budgets (**k = 50, 100**), DAPG maintains recall above **0.966** with competitive or improved latency. Overall, the best trade-off occurs at **k = 10**, where DAPG reaches **0.971 recall** at **1.378 ms** query time.

**Figure 2:** [View PDF](https://github.com/solmazsm/Distance-Aware-Pruned-Graph/blob/master/docs/result/figures/dap_vs_lshapg_comparison.pdf)


## Research Project Directory Structure

```
.
├── .vscode/                         # VS Code settings
│
├── Report/                          # Project report and documentation
│
├── cppCode/
├── DAPG/
│   └── Makefile                         # Build configuration
│   ├── src/                         # Distance-Aware Pruned Graph implementation
│   │   ├── indexes/                 # Precomputed/saved index statistics
│   │   │   ├── mnist_all_index_stats.txt
│   │   │   ├── audio_all_index_stats.txt
│   │   │
│   │   ├── divGraph.h               #Modified: Graph structure (updated Source files)
│   │   ├── dapgalg.h                
│   │   ├── Preprocess.h            
│   │   ├── Query.cpp               
│   │   ├── main.cpp                
│   │                
│   │
├── docs/
    └── result/
│       └── recall_vs_qt_sift100m.pdf
├── dataset/                         
├── .gitattributes
├── .gitignore
├── LICENSE
└── README.md
             
```
## Contact

For questions or contributions, please open an issue or contact the authors listed in the paper.


