// Test if FAISS is installed with GPU support
#include <iostream>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexFlat.h>

int main() {
    try {
        faiss::gpu::StandardGpuResources res;
        std::cout << "FAISS GPU support is available." << std::endl;
        // Optionally, create a simple GPU index to further verify
        int d = 32; // dimension
        faiss::gpu::GpuIndexFlat index(&res, d, faiss::MetricType::METRIC_L2);
        std::cout << "FAISS GPU index created successfully." << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAISS GPU support is NOT available: " << e.what() << std::endl;
        return 1;
    }
}
