#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <output_index_file>" << std::endl;
    return 1;
  }
  std::string output_index_file = argv[1];

  // Load descriptors from file
  std::ifstream ifs(output_index_file + ".desc.bin", std::ios::binary);
  if (!ifs) {
    std::cerr << "Descriptor file not found: " << output_index_file << ".desc.bin" << std::endl;
    return 1;
  }
  int nb = 0, dim = 0;
  ifs.read(reinterpret_cast<char*>(&nb), sizeof(int));
  ifs.read(reinterpret_cast<char*>(&dim), sizeof(int));
  std::vector<float> faiss_db(nb * dim);
  ifs.read(reinterpret_cast<char*>(faiss_db.data()), nb * dim * sizeof(float));
  ifs.close();

  // Faiss CPU setup
  int nlist = 128;  // number of clusters
  faiss::IndexFlatL2 quantizer(dim);
  faiss::IndexIVFFlat index(&quantizer, dim, nlist);

  // Train the index
  index.train(nb, faiss_db.data());
  // Add vectors
  index.add(nb, faiss_db.data());

  // Save index to disk
  faiss::write_index(&index, output_index_file.c_str());
  std::cout << "Trained and saved Faiss IndexIVFFlat (CPU) to " << output_index_file << std::endl;
  return 0;
}
