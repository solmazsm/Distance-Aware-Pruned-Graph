#pragma once
//
// Preprocess.h
// -----------------------------------------------------------------------------
// Dataset and Benchmark Utility for ANN Indexing Methods
// -----------------------------------------------------------------------------
// This header provides loading, benchmarking, and parameter management.
// DAPG (Distance-Aware Pruned Graph) and LSH-APG experiments.
// Serves as a compatibility/util class for ANN experiments 
// 
// -----------------------------------------------------------------------------
// 
#include "def.h"
#include <cmath>
#include <assert.h>
#include <unordered_map>
#include <string>

// Dataset loading, parameter hub, and benchmarking for ANN experiments.

class Preprocess
{
public:
    Data data;
    float* SquareLen = nullptr;
    float** Dists = nullptr;
    Ben benchmark;
    std::string data_file;
    std::string ben_file;
    bool hasT = false;
    float beta = 0.1f;

    // Main constructors
    Preprocess(const std::string& path, const std::string& ben_file_);
    Preprocess(const std::string& path, const std::string& ben_file_, float beta_);
    // Key methods
    void load_data(const std::string& path);
    void ben_make();
    void ben_save();
    void ben_correct();
    void ben_correct_inverse();
    void ben_load();
    void ben_create();
    void showDataset();
    ~Preprocess();
};

// Helper struct for sorting by distance, for internal routines
struct Dist_id
{
    unsigned id = 0;
    float dist = 0;
    bool operator < (const Dist_id& rhs) {
        return dist < rhs.dist;
    }
};

// Parameter storage for LSH & ANN algorithms; supports both baseline & DAPG
class Parameter // N, dim, S, L, K, M, W;
{
public:
    unsigned N = 0;
    unsigned dim = 0;
    unsigned S = 0;   // Total hash functions
    unsigned L = 0;   // Tables
    unsigned K = 0;   // Functions per table
    float W = 1.0f;
    int MaxSize = 0;
    float R_min = 0.3f;

    Parameter(Preprocess& prep, unsigned L_, unsigned K_, float rmin_);
    ~Parameter();
};

