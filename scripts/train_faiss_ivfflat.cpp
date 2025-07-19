#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFAdditiveQuantizer.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/gpu/GpuIndexIVFScalarQuantizer.h>
#include <faiss/index_io.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <output_index_file>" << std::endl;
    return 1;
  }
  std::string datafile = argv[1];
  std::string output_index = argv[2];
  std::string mode = argv[3];
  output_index = output_index + "_" + mode + ".index.bin";

  // Load descriptors from file
  std::ifstream ifs(datafile, std::ios::binary);
  if (!ifs) {
    std::cerr << "Descriptor file not found: " << datafile << std::endl;
    return 1;
  }
  int nb = 0, dim = 0;
  ifs.read(reinterpret_cast<char*>(&nb), sizeof(int));
  ifs.read(reinterpret_cast<char*>(&dim), sizeof(int));
  std::vector<float> faiss_db(nb * dim);
  ifs.read(reinterpret_cast<char*>(faiss_db.data()), nb * dim * sizeof(float));
  ifs.close();

  int nlist = 256*2;  // number of clusters
  if (mode == "ivfpq") {
    nlist = 256;    // for IVFPQ, you might want more clusters
    int m = 16;     // number of subquantizers
    int nbits = 8;  // bits per subquantizer (typical: 8)

    faiss::IndexFlatL2 quantizer(dim);
    faiss::IndexIVFPQ index(&quantizer, dim, nlist, m, nbits);
    // Train the index
    index.train(nb, faiss_db.data());
    // Save index to disk
    faiss::write_index(&index, output_index.c_str());
    std::cout << "Trained and saved Faiss IndexIVFPQ to " << output_index << std::endl;
  } else if (mode == "ivfflat") {
    faiss::IndexFlatL2 quantizer(dim);

    faiss::IndexIVFFlat index(&quantizer, dim, nlist);
    // Train the index
    index.train(nb, faiss_db.data());

    // Save index to disk
    faiss::write_index(&index, output_index.c_str());

    std::cout << "Trained and saved Faiss IndexIVFFlat to " << output_index << std::endl;
  } else {
    std::cerr << "Unsupported mode: " << mode << std::endl;
    return 1;
  }

  return 0;
}
