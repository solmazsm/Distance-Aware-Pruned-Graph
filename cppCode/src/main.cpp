
#include "alg.h"
#include <iomanip> // for std::setprecision
#include <map>
#include <tuple>
#include <filesystem>
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

int main(int argc, char const* argv[])
{

#if (__cplusplus >= 201703L) || (defined(_MSVC_LANG) && (_MSVC_LANG >= 201703L) && (_MSC_VER >= 1913))
	std::cout<<"C++17!\n";
#else
#endif // _HAS_CXX17


  //  W and k Value Loops]

	float c = 1.5;
	std::vector<float> W_values = {0.1f, 0.3f, 0.5f, 1.0f};
	for (float W : W_values) {
		std::vector<unsigned> k_values = {10, 20, 50, 100}; // Try different k values
		unsigned k = 0;
		for (unsigned k_val : k_values) {
			k = k_val;

			unsigned L = 8, K = 10;//NUS
			//L = 10, K = 5;
			float beta = 0.1;
			unsigned Qnum = 100;
			float W = 1.0f;
			int T = 24;
			int efC = 80;
			L = 2;
			K = 18;
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
      
     
      // LSH-APG Configuration Adjustment for DAPG]

			if (argc == 1) {
				const std::string datas[] = { "audio","mnist","SIFT100M","SIFT1M","deep1m" };
				datasetName = datas[0];
				//datasetName = "sift1B"; 
				setW(datasetName, W);
				std::cout << "Using the default configuration!\n\n";
			}

			#if defined(unix) || defined(__unix__)
				

			std::cout << "Using LSH-Graph for " << datasetName << " ..." << std::endl;
			std::cout << "c=        " << c << std::endl;
			std::cout << "k=        " << k << std::endl;
			std::cout << "L=        " << L << std::endl;
			std::cout << "K=        " << K << std::endl;
			std::cout << "T=        " << T << std::endl;
			std::cout << "lsh_UB=   " << _lsh_UB << std::endl;
			Preprocess prep(data_fold + datasetName + ".data", data_fold + "ANN/" + datasetName + ".bench_graph");

			//return 0;

			showMemoryInfo();

			std::string path = index_fold + datasetName + ".index";
			Parameter param1(prep, L, K, 1.0f);
			param1.W = W;
			zlsh* gLsh= nullptr;
			divGraph* divG = nullptr;
			if (isbuilt&&find_file(path + "_divGraph")) {
				divG = new divGraph(&prep, path + "_divGraph", pQ);
				
				divG->L = L;
				if (L == 0) divG->coeffq = 0;
			}
			else {
				if (!GenericTool::CheckPathExistence(index_fold.c_str())) {
					GenericTool::EnsurePathExistence(index_fold.c_str());
				}
       
        // DAPG Graph Construction
      
        
				divG = new divGraph(prep, param1, path + "_divGraph", T, efC, pC, pQ);
				
			}

			//divG->traverse();
			//return 0;

			std::cout << "Loading FastGraph...\n";
			fastGraph* fsG = new fastGraph(divG);
  

      // Naming and Tracking DAP Config]
algNameStream << "DAP_k" << k << "_th" << std::fixed << std::setprecision(3) << divG->last_threshold;

			// Construct algName for use in all graphSearch calls
			std::ostringstream algNameStream;
			algNameStream << "DAP_k" << k << "_th" << std::fixed << std::setprecision(3) << divG->last_threshold;
			std::string algName = algNameStream.str();

			std::stringstream ss;
			ss  << "*******************************************************************************************************\n"
				<< "The result of LSH-G for " << datasetName << " is as follow: "<< "k=" << k<< ", probQ = " << pQ << ", L = " << L << ", K = " << K << ", T = " << T
				<< "\n"
				<< "******************************************************************************************************\n";

			ss << std::setw(_lspace) << "algName"
				<< std::setw(_sspace) << "k"
				<< std::setw(_sspace) << "ef"
				<< std::setw(_lspace) << "Time"
				<< std::setw(_lspace) << "Recall"
				//<< std::setw(_lspace) << "Ratio"
				<< std::setw(_lspace) << "Cost"
				<< std::setw(_lspace) << "CPQ1"
				<< std::setw(_lspace) << "CPQ2"
				<< std::setw(_lspace) << "Pruning"
				//<< std::setw(_lspace) << "MaxHop"
				<< std::endl
				<< std::endl;

			std::cout << ss.str();

			std::string query_result(divG->getFilename());
			auto idx = query_result.rfind('/');
			query_result.assign(query_result.begin(), query_result.begin() + idx + 1);
			std::filesystem::create_directories("indexes");
			std::string result_file = "indexes/" + datasetName + "_result.txt";
			std::ofstream os(result_file, std::ios_base::app);
			os.seekp(0, std::ios_base::end);
			os << ss.str();
			os.close();

			std::vector<size_t> efs;
			for (int i = k; i < 100; i += 10) {
				efs.push_back(i);
			}

	#ifdef _DEBUG
	#else
			for (int i = 100; i < 250; i += 10) {
				efs.push_back(i);
			}
			for (int i = 250; i < 300; i += 50) {
				efs.push_back(i);
			}
			//for (int i = 500; i < 3000; i += 300) {
			//	efs.push_back(i);
			//}
			//if (datasetName == "mnist") {
			//	for (int i = 500; i < 6000; i += 300) {
			//		efs.push_back(i);
			//	}
			//}
	#endif // _DEBUG
			if (k == 50) {
				// for (auto& ef : efs) {
				// 	if (divG) divG->ef = ef;
				// 	graphSearch(c, k, divG, prep, beta, datasetName, data_fold, 2);
				// }
				// std::cout << std::endl;
			}
			else {
				std::vector<int> ks = { 1,10,20,30,40,50,60,70,80,90,100 };

				for (auto& kk : ks) {
					k = kk;

					if (divG) divG->ef = k + 150;

					// Print header dynamically for each configuration
					std::stringstream ss;
					ss  << "*******************************************************************************************************\n"
						<< "The result of LSH-G for " << datasetName << " is as follow: "<< "k=" << k<< ", probQ = " << pQ << ", L = " << L << ", K = " << K << ", T = " << T
						<< "\n"
						<< "******************************************************************************************************\n";
					ss << std::setw(_lspace) << "algName"
						<< std::setw(_sspace) << "k"
						<< std::setw(_sspace) << "ef"
						<< std::setw(_lspace) << "Time"
						<< std::setw(_lspace) << "Recall"
						//<< std::setw(_lspace) << "Ratio"
						<< std::setw(_lspace) << "Cost"
						<< std::setw(_lspace) << "CPQ1"
						<< std::setw(_lspace) << "CPQ2"
						<< std::setw(_lspace) << "Pruning"
						//<< std::setw(_lspace) << "MaxHop"
						<< std::endl
						<< std::endl;
					std::cout << ss.str();

					graphSearch(c, k, divG, prep, beta, datasetName, data_fold, 2, algName);

					//for (auto& ef : efs) {
					//	if (ef < k) continue;
					//	if (divG) divG->ef = ef;
					//	graphSearch(c, kk, divG, prep, beta, datasetName, data_fold, 2);
					//}
					std::cout << std::endl;
				}
			}
			

			//for (auto& ef : efs) {
			//	if (divG) divG->ef = ef;
			//	graphSearch(c, k, divG, prep, beta, datasetName, data_fold, 3);
			//}
			//std::cout << std::endl;
			efs={200};
			SearchResult bestRecall, bestCost, bestTime, bestPruning;
			bestRecall.recall = -1.0f;
			bestCost.cost = std::numeric_limits<float>::max();
			bestTime.time = std::numeric_limits<float>::max();
			bestPruning.pruning = std::numeric_limits<float>::max();
			// --- Begin: Unique results collection (multi-metric) ---
      
  
     //  Dynamic Evaluation and Logging]

			std::map<std::tuple<std::string, int, int>, MultiBest> uniqueResults;
			// --- End: Unique results collection ---
			for (auto& ef : efs) {
				if (divG) divG->ef = ef;
				SearchResult result = graphSearch(c, k, divG, prep, beta, datasetName, data_fold, 2, algName);
				// Use (algName, k, ef) as the key
				auto key = std::make_tuple(algName, k, ef);
				auto& mb = uniqueResults[key];
				if (!mb.hasRecall || result.recall > mb.bestRecall.recall) { mb.bestRecall = result; mb.hasRecall = true; }
				if (!mb.hasCost   || result.cost   < mb.bestCost.cost)     { mb.bestCost   = result; mb.hasCost   = true; }
				if (!mb.hasTime   || result.time   < mb.bestTime.time)     { mb.bestTime   = result; mb.hasTime   = true; }
				if (!mb.hasPruning|| result.pruning> mb.bestPruning.pruning){ mb.bestPruning= result; mb.hasPruning= true; }

				// --- Begin: Write summary line for this run ---
				std::string index_file = path + "_divGraph";
				double is_mb = 0.0;
				if (std::filesystem::exists(index_file)) {
					is_mb = std::filesystem::file_size(index_file) / (1024.0 * 1024.0);
				}
				// Use datasetName in the summary file name
				std::string summary_file = datasetName + "_all_index_stats.txt";
				std::ofstream summary(summary_file, std::ios::app);
				if (summary.tellp() == 0) {
					summary << "Dataset,Algorithm,k,ef,IS_MB,NMCS,IT_s,Recall,Pruning\n";
				}
				summary << datasetName << ","
						<< algName << ","
						<< k << ","
						<< ef << ","
						<< is_mb << ","
						<< result.cost << ","
						<< divG->indexingTime << ","
						<< result.recall << ","
						<< result.pruning << "\n";
				summary.close();
				// --- End: Write summary line for this run ---
			}
			// Print all bests for each config
			std::cout << "\n************ UNIQUE/BEST RESULTS BY METRIC ************\n";
			for (const auto& kv : uniqueResults) {
				std::cout << "Best Recall:  "  << kv.second.bestRecall.rowString;
				std::cout << "Best Cost:    " << kv.second.bestCost.rowString;
				std::cout << "Best Time:    " << kv.second.bestTime.rowString;
				std::cout << "Best Pruning: " << kv.second.bestPruning.rowString;
				std::cout << "---------------------------------------------\n";
			}
			std::cout << "******************************************************\n";

			std::vector<float> pts={0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95};

			std::cout << std::endl;

			//for (auto& ef : efs) {
			//	if (fsG) fsG->ef = ef;
			//	graphSearch(c, k, fsG, prep, beta, datasetName, data_fold, 1);
			//}
			//std::cout << std::endl;

			time_t now = time(0);
			
			
			time_t zero_point = 1635153971 - 17 * 3600 - 27 * 60;//Let me set the time at 2021.10.25. 17:27 as the zero point
			size_t diff = (size_t)(now - zero_point);

			//ss.flush();
			//ss.erase_event();
			ss.str("");
	#if defined(unix) || defined(__unix__)
			llt lt(diff);
	#endif

	#if defined(unix) || defined(__unix__)
			ss << "\n******************************************************************************************************\n"
				<< "                                                                                    "
				<< lt.date << '-' << lt.h << ':' << lt.m << ':' << lt.s
				<< "\n*****************************************************************************************************\n\n\n";
	#else
			tm* ltm = new tm[1];
			localtime_s(ltm, &now);
			ss << "\n******************************************************************************************************\n"
				<< "                                                                                    "
				<< ltm->tm_mon + 1 << '-' << ltm->tm_mday << ' ' << ltm->tm_hour << ':' << ltm->tm_min
				<< "\n*****************************************************************************************************\n\n\n";
	#endif
			std::ofstream os1(result_file, std::ios_base::app);
			os1.seekp(0, std::ios_base::end);
			os1 << ss.str();
			os1.close();
			std::cout << ss.str();
		}
	}
	return 0;
}
