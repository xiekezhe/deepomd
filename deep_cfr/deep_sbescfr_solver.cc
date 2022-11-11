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

#include "deep_sbescfr_solver.h"

#include <memory>
#include <numeric>
#include <random>

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {

double backward_update_regret(double alpha, double opponent_reach,
                              const std::vector<double>& values, int step) {
  int num_actions = values.size();
  double threshold = alpha * opponent_reach;
  std::vector<double> sorted_value = values;
  std::sort(sorted_value.begin(), sorted_value.end());
  std::vector<double> available_values(num_actions);
  auto global_min_value = sorted_value[0] - sqrt(threshold);
  std::vector<double> ubound(num_actions, 0.0);
  for (int i = 0; i != num_actions; ++i) {
    for (int j = i; j != num_actions; ++j) {
      ubound[i] += std::pow(sorted_value[j] - sorted_value[i], 2);
    }
  }
  for (int i = 0; i != num_actions; ++i) {
    if (threshold <= ubound[i]) {
      continue;
    }
    double a = num_actions - i;
    double b = 0;
    double c = -threshold;
    for (int j = i; j != num_actions; ++j) {
      b += -2 * sorted_value[j];
      c += std::pow(sorted_value[j], 2);
    }
    auto x1 = (-b - sqrt(std::max(0.0, b * b - 4 * a * c))) / (2 * a);
    return x1;
  }
  return 0.0;
}

double cfrplus_backward_update_regret(double alpha, double opponent_reach,
                                      const std::vector<double>& values,
                                      std::vector<double>& sub_values,
                                      const std::vector<double>& policy,
                                      int step) {
  double threshold_t_1 = alpha * opponent_reach;
  double policy_norm_2 = 0;
  for (int i = 0; i != policy.size(); ++i) {
    policy_norm_2 += policy[i] * policy[i];
  }
  SPIEL_CHECK_EQ(values.size(), sub_values.size());
  SPIEL_CHECK_EQ(values.size(), policy.size());
  if (step == 1) {
    sub_values = values;
  } else {
    for (int i = 0; i != values.size(); ++i) {
      double sub_v =
          values[i] + policy[i] * (std::sqrt(threshold_t_1 / policy_norm_2));
      sub_values[i] = sub_v;
    }
  }
  return backward_update_regret(alpha, opponent_reach, sub_values, step);
}

double adapt(const DeepCFRConfig& config, double alpha,
             const std::vector<double>& values) {
  double sum_values = 0;
  double max_values = 0;
  static const double scale_up_coeff_ = config.cfr_rm_amp;
  static const double scale_down_coeff_ = config.cfr_rm_damp;
  static const double scale_ub_ = config.cfr_scale_ub;
  static const double scale_lb_ = config.cfr_scale_lb;
  static const double alpha_ub_ = config.cfr_rm_ub;
  static const double alpha_lb_ = config.cfr_rm_lb;
  for (auto& v : values) {
    if (abs(v) > max_values) {
      max_values = abs(v);
    }
    sum_values += v;
  }
  if (sum_values > scale_ub_ * max_values) {
    alpha *= scale_up_coeff_;
  } else if (sum_values < scale_lb_ * max_values) {
    alpha *= scale_down_coeff_;
  }
  alpha = std::min(alpha_ub_, std::max(alpha, alpha_lb_));
  return alpha;
}

DeepSbESCFRSolver::DeepSbESCFRSolver(
    const Game& game, const DeepCFRConfig& config,
    std::vector<std::shared_ptr<VPNetEvaluator>> value_evals,
    std::vector<std::shared_ptr<VPNetEvaluator>> global_value_evals,
    std::vector<std::shared_ptr<VPNetEvaluator>> policy_evals,
    std::vector<std::shared_ptr<VPNetEvaluator>> current_policy_evals,
    std::mt19937* rng)
    : game_(game.Clone()),
      config_(config),
      rng_(rng),
      iterations_(0),
      step_(0),
      player_(-1),
      avg_type_(GetAverageType(config.average_type)),
      weight_type_(GetWeightType(config.weight_type)),
      dist_(0.0, 1.0),
      value_eval_(value_evals),
      global_value_eval_(global_value_evals),
      policy_eval_(policy_evals),
      current_policy_eval_(current_policy_evals),
      tree_(game_->NewInitialState()),
      root_node_(tree_.Root()),
      root_state_(root_node_->GetState()),
      use_regret_net(config.use_regret_net),
      use_policy_net(config.use_policy_net),
      use_tabular(config.use_tabular) {
  if (game_->GetType().dynamics != GameType::Dynamics::kSequential) {
    SpielFatalError(
        "MCCFR requires sequential games. If you're trying to run it "
        "on a simultaneous (or normal-form) game, please first transform it "
        "using turn_based_simultaneous_game.");
  }
}

std::vector<Trajectory> DeepSbESCFRSolver::RunIteration(Player player,
                                                        double alpha, int step,
                                                        int max_iterations) {
  return RunIteration(rng_, player, alpha, step, max_iterations);
}

std::vector<Trajectory> DeepSbESCFRSolver::RunIteration(std::mt19937* rng,
                                                        Player player,
                                                        double alpha, int step,
                                                        int max_iterations) {
  alpha_ = alpha;
  node_touch_ = 0;
  max_iterations_ = max_iterations;
  if (step_ != step || player_ != player) {
    iterations_ = 0;
  }
  step_ = step;
  player_ = player;
  ++iterations_;
  Trajectory value_trajectory;
  Trajectory policy_trajectory;
  // Sample a chace seed at the start of an iteration.
  ChanceData chance_data = root_state_->SampleChance(rng);
  double value = UpdateRegrets(root_node_, player, 1, 1, 1, 1, value_trajectory,
                               policy_trajectory, step, rng, chance_data);
  value_trajectory.node_touched = node_touch_;
  value_trajectory.value = value;
  return {value_trajectory, policy_trajectory};
}

double DeepSbESCFRSolver::UpdateRegrets(
    PublicNode* node, Player player, double player_reach, double opponent_reach,
    double ave_opponent_reach, double sampling_reach,
    Trajectory& value_trajectory, Trajectory& policy_trajectory, int step,
    std::mt19937* rng, const ChanceData& chance_data) {
  State& state = *(node->GetState());
  universal_poker::UniversalPokerState* poker_state =
      static_cast<universal_poker::UniversalPokerState*>(node->GetState());
  state.SetChance(chance_data);
  // std::cout << state.ToString() << std::endl;
  if (state.IsTerminal()) {
    return state.PlayerReturn(player);
  } else if (state.IsChanceNode()) {
    Action action = SampleAction(state.ChanceOutcomes(), dist_(*rng)).first;
    return UpdateRegrets(node->GetChild(action), player, player_reach,
                         opponent_reach, ave_opponent_reach, sampling_reach,
                         value_trajectory, policy_trajectory, step, rng,
                         chance_data);
  } else if (state.IsSimultaneousNode()) {
    SpielFatalError(
        "Simultaneous moves not supported. Use "
        "TurnBasedSimultaneousGame to convert the game first.");
  }

  node_touch_ += 1;

  Player cur_player = state.CurrentPlayer();
  std::string is_key = state.InformationStateString(cur_player);
  std::vector<Action> legal_actions = state.LegalActions();
  std::vector<double> information_tensor = state.InformationStateTensor();

  CFRInfoStateValues info_state_copy(legal_actions, kInitialTableValues);
  CFRInfoStateValues current_info_state(legal_actions, kInitialTableValues);
  if (step != 1) {
    // get current policy
    CFRNetModel::InferenceInputs inference_input{is_key, legal_actions,
                                                 information_tensor};
    auto cfr_policy = current_policy_eval_[cur_player]
                          ->Inference(cur_player, inference_input)
                          .value;
    current_info_state.SetPolicy(cfr_policy);
    info_state_copy.SetPolicy(cfr_policy);
  }

  double value = 0.0;
  std::vector<double> child_values(legal_actions.size(), 0.0);
  int action_index = 0;

  if (cur_player == player) {
    // Walk over all actions at my nodes
    for (int aidx = 0; aidx < legal_actions.size(); ++aidx) {
      double child_reach =
          current_info_state.current_policy[aidx] * player_reach;
      child_values[aidx] = UpdateRegrets(
          node->GetChild(legal_actions[aidx]), player, child_reach,
          opponent_reach, ave_opponent_reach, sampling_reach, value_trajectory,
          policy_trajectory, step, rng, chance_data);
      value += current_info_state.current_policy[aidx] * child_values[aidx];
    }
  } else {
    // Sample at opponent nodes.
    int aidx = current_info_state.SampleActionIndex(0.0, dist_(*rng));
    action_index = aidx;
    double new_reach = current_info_state.current_policy[aidx] * opponent_reach;
    double new_ave_reach = 1.0 / legal_actions.size() * ave_opponent_reach;
    double new_sampling_reach =
        current_info_state.current_policy[aidx] * sampling_reach;
    value = UpdateRegrets(node->GetChild(legal_actions[aidx]), player,
                          player_reach, new_reach, new_ave_reach,
                          new_sampling_reach, value_trajectory,
                          policy_trajectory, step, rng, chance_data);
  }

  if (cur_player == player) {
    CFRNetModel::InferenceInputs inference_input{is_key, legal_actions,
                                                 information_tensor};
    double delta = poker_state->MaxUtility() - poker_state->MinUtility();
    double Z_weight = 1.0;
    if (weight_type_ == WeightType::kLinear) {
      Z_weight = step;
    }
    std::vector<double> next_policy(legal_actions.size());
    std::vector<double> next_regret(legal_actions.size());
    std::vector<double> sub_values(legal_actions.size());
    if (step == 1) {
      sub_values = child_values;
    } else {
      value = cfrplus_backward_update_regret(
          alpha_,
          delta * delta * legal_actions.size() * Z_weight * ave_opponent_reach,
          child_values, sub_values, current_info_state.current_policy, step);
    }
    for (int aidx = 0; aidx < legal_actions.size(); ++aidx) {
      next_regret[aidx] = next_policy[aidx] =
          std::max(sub_values[aidx] - value, 0.0);
    }
    CFRInfoStateValues next_info_state(legal_actions, kInitialTableValues);
    next_info_state.SetRegret(next_policy);
    next_info_state.ApplyRegretMatching();
    next_policy = next_info_state.current_policy;
    for (int aidx = 0; aidx != legal_actions.size(); ++aidx) {
      SPIEL_CHECK_TRUE(next_policy[aidx] >= 0 && next_policy[aidx] <= 1);
    }

    if (!use_regret_net || use_tabular) {
      CFRNetModel::TrainInputs value_train_input{
          is_key, legal_actions, legal_actions, information_tensor, next_regret,
          1.0};
      value_eval_[cur_player]->SetCFRTabular(value_train_input);
      CFRNetModel::TrainInputs train_input{is_key,        legal_actions,
                                           legal_actions, information_tensor,
                                           next_policy,   1.0};
      current_policy_eval_[cur_player]->SetCFRTabular(train_input);
    }
    if (use_regret_net) {
      value_trajectory.states.push_back(ReplayNode{
          is_key, information_tensor, cur_player, legal_actions, next_policy,
          next_regret, -1, 1.0, player_reach,
          delta * delta * legal_actions.size() * Z_weight * ave_opponent_reach,
          sampling_reach});
    }
  }

  if ((avg_type_ == AverageType::kOpponent ||
       avg_type_ == AverageType::kLinearOpponent) &&
      cur_player == ((player + 1) % game_->NumPlayers())) {
    double policy_weight = 1.0;
    if (avg_type_ == AverageType::kLinearOpponent) {
      policy_weight = step;
    }
    if (!use_policy_net || use_tabular) {
      std::vector<double> policy(legal_actions.size());
      for (int aidx = 0; aidx < legal_actions.size(); ++aidx) {
        policy[aidx] = current_info_state.current_policy[aidx] * policy_weight;
      }
      CFRNetModel::TrainInputs train_input{is_key,        legal_actions,
                                           legal_actions, information_tensor,
                                           policy,        1.0};
      policy_eval_[cur_player]->AccumulateCFRTabular(train_input);
    }
    if (use_policy_net) {
      policy_trajectory.states.push_back(
          ReplayNode{is_key,
                     information_tensor,
                     cur_player,
                     legal_actions,
                     current_info_state.current_policy,
                     {},
                     action_index,
                     policy_weight,
                     player_reach,
                     opponent_reach,
                     sampling_reach});
    }
  }
  return value;
}

}  // namespace algorithms
}  // namespace open_spiel
