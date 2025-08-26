#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFAdditiveQuantizer.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexIVFPQFastScan.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/VectorTransform.h>
#include <faiss/gpu/GpuIndexIVFScalarQuantizer.h>
#include <faiss/index_io.h>


#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <datafile> <output_index_file> <mode> [pca_dim]" << std::endl;
    return 1;
  }
  std::string datafile = argv[1];
  std::string output_index = argv[2];
  std::string mode = argv[3];
  int pca_dim = -1;  // -1 means no PCA
  if (argc >= 5) {
    pca_dim = std::stoi(argv[4]);
  }

  // Add PCA suffix to output filename if PCA is used
  if (pca_dim > 0) {
    output_index = output_index + "_" + mode + "_pca" + std::to_string(pca_dim) + ".index.bin";
  } else {
    output_index = output_index + "_" + mode + ".index.bin";
  }

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

  int nlist = 256 * 2;  // number of clusters
  if (mode == "ivfpq") {
    nlist = 256;    // for IVFPQ, you might want more clusters
    int m = 16;     // number of subquantizers
    int nbits = 4;  // bits per subquantizer (typical: 8)

    int final_dim = (pca_dim > 0) ? pca_dim : dim;

    if (pca_dim > 0 && pca_dim < dim) {
      // Create PCA transform + IVFPQ index
      faiss::PCAMatrix pca_transform(dim, pca_dim);
      faiss::IndexFlatL2 quantizer(pca_dim);
      faiss::IndexIVFPQFastScan base_index(&quantizer, pca_dim, nlist, m, nbits);

      faiss::IndexPreTransform index(&pca_transform, &base_index);

      // Train the index (this will train both PCA and the base index)
      index.train(nb, faiss_db.data());

      // Save index to disk
      faiss::write_index(&index, output_index.c_str());
      std::cout << "Trained and saved Faiss IndexIVFPQ with PCA(" << pca_dim << ") to " << output_index << std::endl;
    } else {
      // Original IVFPQ without PCA
      faiss::IndexFlatL2 quantizer(dim);
      faiss::IndexIVFPQFastScan index(&quantizer, dim, nlist, m, nbits);
      // Train the index
      index.train(nb, faiss_db.data());
      // Save index to disk
      faiss::write_index(&index, output_index.c_str());
      std::cout << "Trained and saved Faiss IndexIVFPQ to " << output_index << std::endl;
    }
  } else if (mode == "ivfflat") {
    if (pca_dim > 0 && pca_dim < dim) {
      // Create PCA transform + IVFFlat index
      faiss::PCAMatrix pca_transform(dim, pca_dim);
      faiss::IndexFlatL2 quantizer(pca_dim);
      faiss::IndexIVFFlat base_index(&quantizer, pca_dim, nlist);

      faiss::IndexPreTransform index(&pca_transform, &base_index);

      // Train the index (this will train both PCA and the base index)
      index.train(nb, faiss_db.data());

      // Save index to disk
      faiss::write_index(&index, output_index.c_str());

      std::cout << "Trained and saved Faiss IndexIVFFlat with PCA(" << pca_dim << ") to " << output_index << std::endl;
    } else {
      // Original IVFFlat without PCA
      faiss::IndexFlatL2 quantizer(dim);

      faiss::IndexIVFFlat index(&quantizer, dim, nlist);
      // Train the index
      index.train(nb, faiss_db.data());

      // Save index to disk
      faiss::write_index(&index, output_index.c_str());

      std::cout << "Trained and saved Faiss IndexIVFFlat to " << output_index << std::endl;
    }
  } else {
    std::cerr << "Unsupported mode: " << mode << std::endl;
    return 1;
  }

  return 0;
}
