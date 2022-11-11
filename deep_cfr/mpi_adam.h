#ifndef DEEP_CFR_MPI_ADAM_H_
#define DEEP_CFR_MPI_ADAM_H_

#include <mpi.h>

#include <cmath>
#include <functional>

#include "Eigen/Core"
#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
namespace open_spiel {
namespace algorithms {
class MpiAdam {
 public:
  MpiAdam(int flat_size, const std::function<Eigen::ArrayXf()> &GetFlat,
          const std::function<void(const Eigen::ArrayXf &)> &SetFlat,
          double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8,
          bool scale_grad_by_procs = true)
      : flat_size_(flat_size),
        GetFlat(GetFlat),
        SetFlat(SetFlat),
        m_(flat_size),
        v_(flat_size),
        beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon),
        scale_grad_by_procs_(scale_grad_by_procs),
        step_(0) {
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
    m_.setZero();
    v_.setZero();
  }
  ~MpiAdam() {}

  void Update(int batch_size, const Eigen::ArrayXf &localg, double step_size,
              double grad_clip) {
    if (step_ % 1000 == 0) {
      CheckSynced();
    }
    Eigen::ArrayXf globalg(localg.size());
    Eigen::ArrayXf localg_clone = localg;
    int sum_batch_size;
    MPI_Allreduce(&batch_size, &sum_batch_size, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    SPIEL_CHECK_GE(batch_size, 0);
    SPIEL_CHECK_GE(sum_batch_size, 0);
    if (!sum_batch_size) {
      return;
    }
    if (scale_grad_by_procs_) {
      localg_clone *= batch_size;
    }
    MPI_Allreduce(localg_clone.data(), globalg.data(), localg.size(), MPI_FLOAT,
                  MPI_SUM, MPI_COMM_WORLD);
    if (scale_grad_by_procs_) {
      globalg /= sum_batch_size;
    }
    if (grad_clip > 0) {
      auto gradnorm = globalg.matrix().norm();
      if (gradnorm > grad_clip) {
        globalg /= gradnorm * grad_clip;
      }
    }
    step_ += 1;
    double a = step_size * std::sqrt(1 - std::pow(beta2_, step_)) /
               (1 - std::pow(beta1_, step_));
    m_ = beta1_ * m_ + (1 - beta1_) * globalg;
    v_ = beta2_ * v_ + (1 - beta2_) * Eigen::pow(globalg, 2);
    auto diff = -a * m_ / (Eigen::sqrt(v_) + epsilon_);
    // if (mpi_rank_ == 0) {
    //   std::cout << "global [" << globalg << "]" << std::endl;
    //   std::cout << "m [" << m_ << "]" << std::endl;
    //   std::cout << "v [" << v_ << "]" << std::endl;
    //   std::cout << "diff [" << diff << "]" << std::endl;
    // }
    SetFlat(GetFlat() + diff);
  }

  void Update_SGD(int batch_size, const Eigen::ArrayXf &localg,
                  double step_size, double grad_clip) {
    if (step_ % 1000 == 0) {
      CheckSynced();
    }
    Eigen::ArrayXf globalg(localg.size());
    Eigen::ArrayXf localg_clone = localg;
    int sum_batch_size;
    MPI_Allreduce(&batch_size, &sum_batch_size, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    SPIEL_CHECK_GE(batch_size, 0);
    SPIEL_CHECK_GE(sum_batch_size, 0);
    if (!sum_batch_size) {
      return;
    }
    if (scale_grad_by_procs_) {
      localg_clone *= batch_size;
    }
    MPI_Allreduce(localg_clone.data(), globalg.data(), localg.size(), MPI_FLOAT,
                  MPI_SUM, MPI_COMM_WORLD);
    if (scale_grad_by_procs_) {
      globalg /= sum_batch_size;
    }
    if (grad_clip > 0) {
      auto gradnorm = globalg.matrix().norm();
      if (gradnorm > grad_clip) {
        globalg /= gradnorm * grad_clip;
      }
    }
    step_ += 1;

    auto diff = -step_size * globalg;
    SetFlat(GetFlat() + diff);
  }

  void Sync() {
    Eigen::ArrayXf theta = GetFlat();
    MPI_Bcast(theta.data(), theta.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    SetFlat(theta);
  }

  void CheckSynced() {
    if (mpi_rank_ == 0) {
      Eigen::ArrayXf theta = GetFlat();
      MPI_Bcast(theta.data(), theta.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    } else {
      Eigen::ArrayXf theta_local = GetFlat();
      Eigen::ArrayXf theta_root(theta_local.size());
      MPI_Bcast(theta_root.data(), theta_root.size(), MPI_FLOAT, 0,
                MPI_COMM_WORLD);
      // if (!((theta_local - theta_root).abs().maxCoeff() < 1e-8)) {
      //   std::cout << "step " << step_ << std::endl;
      //   std::cout << (theta_local - theta_root).abs().maxCoeff() <<
      //   std::endl; std::cout << "local" << theta_local.head(10) << std::endl;
      //   std::cout << "root" << theta_root.head(10) << std::endl;
      // }
      SPIEL_CHECK_TRUE((theta_local - theta_root).abs().maxCoeff() < 1e-8);
      SetFlat(theta_root);
    }
  }

 private:
  int mpi_rank_;
  int mpi_size_;
  int flat_size_;
  Eigen::ArrayXf m_;
  Eigen::ArrayXf v_;
  double beta1_;
  double beta2_;
  double epsilon_;
  bool scale_grad_by_procs_;
  int step_;
  std::function<Eigen::ArrayXf()> GetFlat;
  std::function<void(const Eigen::ArrayXf &)> SetFlat;
};
}  // namespace algorithms
}  // namespace open_spiel
#endif  // DEEP_CFR_MPI_ADAM_H_