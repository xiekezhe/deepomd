// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DEEPCFR_OSSBCFR_SOLVER_H_
#define DEEPCFR_OSSBCFR_SOLVER_H_

#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/algorithms/public_tree.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "vpevaluator.h"
#include "vpnet.h"

namespace open_spiel {
namespace algorithms {

enum class AverageType {
  kSimple,
  kFull,
  kCurrent,
};

struct ReplayNode {
  std::string info_str;
  std::vector<double> information;
  open_spiel::Player current_player;
  std::vector<open_spiel::Action> legal_actions;
  std::vector<double> policy;
  std::vector<double> value;
  int aidx;
  double weight;
  double player_reach;
  double opponent_reach;
  double sampling_reach;

  ReplayNode operator+=(const ReplayNode& other) {
    SPIEL_CHECK_EQ(current_player, other.current_player);
    SPIEL_CHECK_EQ(info_str, other.info_str);
    SPIEL_CHECK_EQ(value.size(), other.value.size());
    for (int i = 0; i != information.size(); ++i) {
      SPIEL_CHECK_EQ(information[i], other.information[i]);
    }
    for (int i = 0; i != legal_actions.size(); ++i) {
      SPIEL_CHECK_EQ(legal_actions[i], other.legal_actions[i]);
    }
    for (int i = 0; i != policy.size(); ++i) {
      policy[i] = other.policy[i];
    }
    SPIEL_CHECK_GE(weight, 0);
    SPIEL_CHECK_GE(other.weight, 0);
    for (int i = 0; i != value.size(); ++i) {
      value[i] = (value[i] + other.value[i]);
    }
    aidx = other.aidx;
    weight = other.weight;
    player_reach = other.player_reach;
    opponent_reach = other.opponent_reach;
    sampling_reach = other.sampling_reach;

    return *this;
  }

  ReplayNode operator+(const ReplayNode& other) const {
    ReplayNode ret = *this;
    ret += other;
    return ret;
  }
};

struct PolicyReplayNode {
  std::string info_str;
  std::vector<double> information;
  open_spiel::Player current_player;
  std::vector<open_spiel::Action> legal_actions;
  std::vector<double> policy;
  double weight;
};

std::ostream& operator<<(std::ostream& out, const ReplayNode& node);

struct Trajectory {
  std::vector<ReplayNode> states;
};

class DeepOSSBCFRSolver {
 public:
  static constexpr double kInitialTableValues = 0.000001;

  // allow to get cfr value or policy by value_eval_.
  DeepOSSBCFRSolver(
      const Game& game,
      std::vector<std::shared_ptr<VPNetEvaluator>> value_evals,
      std::vector<std::shared_ptr<VPNetEvaluator>> global_value_evals,
      std::vector<std::shared_ptr<VPNetEvaluator>> policy_evals,
      std::vector<std::shared_ptr<VPNetEvaluator>> current_policy_evals,
      bool use_regret_net, bool use_policy_net, bool use_tabular,
      bool anticipatory, double alpha, double eta, double epsilon,
      bool symmetry, std::mt19937* rng,
      AverageType avg_type = AverageType::kSimple);

  std::vector<Trajectory> RunIteration(Player player, double alpha, int step);

  // Same as above, but uses the specified random number generator instead.
  std::vector<Trajectory> RunIteration(std::mt19937* rng, Player player,
                                       double alpha, int step);

  int NodeTouched() { return node_touch_; }

 protected:
  virtual double UpdateRegrets(PublicNode* node, Player player,
                               double player_reach, double oppoment_reach,
                               double ave_opponent_reach, double sampling_reach,
                               Trajectory& value_trajectory,
                               Trajectory& history_trajectory,
                               Trajectory& policy_trajectory, int step,
                               std::mt19937* rng,
                               const ChanceData& chance_data);

  std::shared_ptr<const Game> game_;
  std::mt19937* rng_;
  uint32_t iterations_;
  AverageType avg_type_;
  std::uniform_real_distribution<double> dist_;
  std::vector<std::shared_ptr<VPNetEvaluator>> value_eval_;
  std::vector<std::shared_ptr<VPNetEvaluator>> global_value_eval_;
  std::vector<std::shared_ptr<VPNetEvaluator>> policy_eval_;
  std::vector<std::shared_ptr<VPNetEvaluator>> current_policy_eval_;
  PublicTree tree_;
  PublicNode* root_node_;
  State* root_state_;
  bool use_regret_net;
  bool use_policy_net;
  bool use_tabular;
  bool anticipatory_;
  double alpha_;
  double eta_;
  double epsilon_;
  bool symmetry_;
  int node_touch_;
};

}  // namespace algorithms
}  // namespace open_spiel

#endif  // DEEPCFR_OSSBCFR_SOLVER_H_
