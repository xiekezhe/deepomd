#include <mpi.h>

#include "mpi_adam.h"
#include "open_spiel/utils/thread_pool.h"

namespace open_spiel {
namespace algorithms {

void cal() {
  int k = 0;
  for (int i = 0; i != 10000; ++i) {
    for (int j = 0; j != 10000000; ++j) {
      k = (i * j) % 10;
    }
  }
}

void TestMpiAdam() {
  int mpi_size, mpi_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  std::cout << MPI_COMM_WORLD << " "
            << "mpi_size: " << mpi_size << " mpi_rank: " << mpi_rank
            << std::endl;
  ThreadPool pool(10);
  for (int i = 0; i != 10; ++i) {
    pool.enqueue(cal);
  }
  int size = 10;
  Eigen::ArrayXf vars = Eigen::ArrayXf::Zero(size);
  std::function<Eigen::ArrayXf()> GetFlat = [&vars]() { return vars; };
  std::function<void(const Eigen::ArrayXf&)> SetFlat =
      [&vars](Eigen::ArrayXf flat) { vars = flat; };
  MpiAdam adam(size, GetFlat, SetFlat);
  adam.Sync();
  for (int i = 0; i != 10; ++i) {
    Eigen::ArrayXf grad =
        -(Eigen::ArrayXf::LinSpaced(size, 0.1, 1.0) + 0.1 * mpi_rank);
    adam.Update(10, grad, 0.1, 1);
    if (mpi_rank == 0) {
      std::cout << "vars [" << vars << "]" << std::endl;
    }
  }
}
}
}

int main(int argc, char** argv) {
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
  std::cout << MPI_THREAD_SERIALIZED << " " << provided << std::endl;
  if (provided != MPI_THREAD_SERIALIZED) {
    std::cout << "Warning MPI did not provide MPI_THREAD_SERIALIZED"
              << std::endl;
  }
  open_spiel::algorithms::TestMpiAdam();
  MPI_Finalize();
}