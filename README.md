# Distance-Aware-Pruned-Graph

## DAPG: A Distance-Aware Pruned Graph Framework for Fast and Accurate Approximate Nearest Neighbor Search

> 
---

## Introduction

This repository provides the source code for **DAPG**, a novel graph-based Approximate Nearest Neighbor (ANN) indexing system introduced in the paper:
DAPG integrates a **Distance-Aware Pruning (DAP)** mechanism into proximity graph construction, enabling efficient ANN search with reduced edge redundancy and improved recall. It adapts pruning thresholds based on **local geometric statistics**, producing sparse yet high-quality graphs.


---

## Compilation

The code is implemented in **C++11** and supports parallelism using **OpenMP**. It can be compiled on both Linux and Windows.

### Linux

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

## Usage

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
-  Compatible with LSH-based Pipelines: Easily integrates with existing LSH-APG or other ANN frameworks.

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

## Datasets

We support and have tested DAPG on:

- [Audio](https://github.com/RSIA-LIESMARS-WHU/LSHBOX-sample-data)
- [SIFT1M](http://corpus-texmex.irisa.fr/)
- [Deep1M](https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/deep1M.tar.gz)
- [MNIST](http://yann.lecun.com/exdb/mnist/)
- [SIFT100M](http://corpus-texmex.irisa.fr/)


Convert these into the `.data_new` format for compatibility.

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
 

## [2025-08-01] – Local Percentile-Based Pruning Integration
### Key Features and Observations

#### Hybrid LSH + Graph Construction
- Integrated LSH-based candidate retrieval (`searchLSH`) with incremental graph building.
- Used `insertLSHRefine` to refine candidate neighbors before inserting them into the graph.

#### Distance-Aware Pruning (DAP)
- Introduced a percentile-based thresholding mechanism.
- For each node, computed the 80th percentile distance (τ_q) over LSH candidates.
- Inserted only neighbors with `dist < τ_q` to ensure sparsity and relevance.
- Exposed `last_threshold` for optional diagnostics or debugging.

####  Neighbor Management
- Supports two pruning strategies:
  - `chooseNN_div`: Ensures selected neighbors are both close and diverse (distance-based repulsion).
  - `chooseNN_simple`: Simplified version using max-heap filtering.

####  Parallel Construction
- Used `ParallelFor` to insert all nodes in parallel (except the first).
- Enabled thread safety via `std::shared_mutex` for concurrent graph updates.
- Fallbacks to `std::mutex` when C++17 is not available.

#### Serialization
- Graph (`linkLists`) and LSH hash tables are saved to and loaded from a binary format.
- Implemented in `save()` and the constructor `divGraph(Preprocess* prep, ...)`.

####  Search Procedure
- Two-stage hybrid search:
  1. LSH-based candidate generation (`searchLSH`).
  2. Graph-based search refinement (`bestFirstSearchInGraph`).
- Reflects an LSH-HNSW-style retrieval pipeline for improved accuracy and speed.

## Benchmark Logs



Each row logs detailed metrics:

- **Recall**: Top-k retrieval accuracy
- **Pruning**: Ratio of retained neighbors after DAP-based filtering
- **Time**, **Cost**, and additional performance indicators
- **Algorithm Name**: Includes pruning threshold information (e.g., `DAP_k10_th...`)

All data in this file was automatically logged during batch experiments executed using the modified `main()` function in `cppCode/LSH-APG/src`.

These results validate the effectiveness of our DAPG method, as submitted in the VLDB 2026 paper.


### Experimental Results Log

The experimental results presented here were produced by our DAPG system on the MNIST dataset to support reproducibility and performance evaluation.

- The file contains timestamped benchmark logs produced from running LSH-G with varying `ef`, `k`, and pruning thresholds.
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

These logs support reproducibility and provide a transparent view of DAPG’s performance characteristics.

### Parameter Settings

We evaluate DAP (Distance-Aware Pruning) with:

- `K = 18`, `L = 2`, `T = 24`, `T′ = 48`
- `W = 1.0`, `pC = 0.95`, `pQ = 0.9`, `efC = 80`

DAP applies **local dynamic pruning**, computing a threshold `τ_q` per node, unlike LSH-APG's global percentile pruning (`p = 95`). This results in better graph sparsity and recall-efficiency.

We tune `K = 18` (vs. LSH-APG's default `K = 16`) for fairness.

#### Baseline Settings

- **HNSW**: `M = 48`, `ef = 80`  
- **NSG**: `L = 40`, `R = 50`, `C = 500`  
- **HCNNG**: 10 iterations, cluster size ≤ 500  
- **DB-LSH**: `c = 1.5`, `K = 12`, `L = 5`

---

### Evaluation Setup

We evaluate DAP across a range of:

- `k ∈ {1, 10, 20, ..., 100}`
- `ef` values for query expansion

This allows robust analysis of recall and efficiency across diverse search settings


## Research Project Directory Structure

```
.
├── .vscode/                         # VS Code settings
├── Report/                          # Project report and documentation
├── cppCode/                         # Main C++ codebase
│   ├── DAPG/                        # Modified: Distance-Aware Pruned Graph implementation
│                 
    
│       ├── indexes/                # Precomputed or saved index files
│       ├── src/                    # Modified: Updated source files 
│       ├── Makefile                # Build configuration
│       ├── all_index_stats.txt     # index performance stats (DAP)
│       ├── audio_all_index_stats.txt
│       ├── lgo                     # Executable 
│       └── mnist_all_index_stats.txt
├── dataset/                         # Benchmark datasets (e.g., MNIST, Audio)(DAPG)
├── .gitattributes                   
├── .gitignore                       
├── LICENSE                          
└── README.md                       
```
## Contact

For questions or contributions, please open an issue or contact the authors listed in the paper.


