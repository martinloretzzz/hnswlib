#include "../../hnswlib/hnswlib.h"
#include <chrono>

int main() {
    int dim = 1024;               // Dimension of the elements
    int max_elements = 10000;   // Maximum number of elements, should be known beforehand
    int M = 32;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff
    int time_elements = 10000;

    // Initing index
    hnswlib::InnerProductSpace space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    float* data = new float[dim * max_elements];
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);
    }

    float correct = 0;
    float recall = 0;
    std::string hnsw_path = "hnsw.bin";

    using picoseconds = std::chrono::duration<long long, std::pico>;
    auto ti0 = std::chrono::steady_clock::now();

    // Add data to index
    for (int i = 0; i < max_elements; i++) {
        alg_hnsw->addPoint(data + i * dim, i);
    }

    auto ti1 = std::chrono::steady_clock::now();
    auto di = picoseconds{ti1 - ti0};

    std::cout << "Index build time " << di.count() / 1000000000 << "ms\n";


    // Query the elements for themselves and measure recall
    correct = 0;
    for (int i = 0; i < max_elements; i++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + i * dim, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i) correct++;
    }
    recall = correct / max_elements;
    std::cout << "Recall: " << recall << "\n";

    // Serialize index
    alg_hnsw->saveIndex(hnsw_path);
    delete alg_hnsw;

    // Deserialize index and check recall
    alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path);

    /*
    correct = 0;
    for (int i = 0; i < max_elements; i++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + i * dim, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i) correct++;
    }
    recall = (float)correct / max_elements;
    std::cout << "Recall of deserialized index: " << recall << "\n";
    */

    using picoseconds = std::chrono::duration<long long, std::pico>;
    auto t0 = std::chrono::steady_clock::now();

    correct = 0;
    for (int i = 0; i < time_elements; i++) {
        int element = i % max_elements;
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + element * dim, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i) correct++;
    }
    recall = (float)correct / max_elements;

    auto t1 = std::chrono::steady_clock::now();
    auto d = picoseconds{t1 - t0};

    std::cout << "Recall: " << recall << "\n";
    std::cout << d.count() / 1000000000 << "ms\n";

    delete[] data;
    delete alg_hnsw;
    return 0;
}
