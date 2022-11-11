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

#include "ossbcfr_solver.h"

#include <memory>
#include <numeric>
#include <random>

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/games/universal_poker.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "vpnet.h"

namespace open_spiel {
namespace algorithms {

std::ostream& operator<<(std::ostream& out, const ReplayNode& node) {
  out << node.information << " " << node.current_player << " "
      << node.legal_actions << " " << node.value << " " << node.player_reach
      << " " << node.opponent_reach << " " << node.sampling_reach;
  return out;
}

DeepOSSBCFRSolver::DeepOSSBCFRSolver(
    const Game& game, std::vector<std::shared_ptr<VPNetEvaluator>> value_evals,
    std::vector<std::shared_ptr<VPNetEvaluator>> global_value_evals,
    std::vector<std::shared_ptr<VPNetEvaluator>> policy_evals,
    std::vector<std::shared_ptr<VPNetEvaluator>> current_policy_evals,
    bool use_regret_net, bool use_policy_net, bool use_tabular,
    bool anticipatory, double alpha, double eta, double epsilon, bool symmetry,
    std::mt19937* rng, AverageType avg_type)
    : game_(game.Clone()),
      rng_(rng),
      iterations_(0),
      avg_type_(avg_type),
      dist_(0.0, 1.0),
      value_eval_(value_evals),
      global_value_eval_(global_value_evals),
      policy_eval_(policy_evals),
      current_policy_eval_(current_policy_evals),
      tree_(game_->NewInitialState()),
      root_node_(tree_.Root()),
      root_state_(root_node_->GetState()),
      use_regret_net(use_regret_net),
      use_policy_net(use_policy_net),
      use_tabular(use_tabular),
      anticipatory_(anticipatory),
      alpha_(alpha),
      eta_(eta),
      epsilon_(epsilon),
      symmetry_(symmetry) {
  if (game_->GetType().dynamics != GameType::Dynamics::kSequential) {
    SpielFatalError(
        "MCCFR requires sequential games. If you're trying to run it "
        "on a simultaneous (or normal-form) game, please first transform it "
        "using turn_based_simultaneous_game.");
  }
}

std::vector<Trajectory> DeepOSSBCFRSolver::RunIteration(Player player,
                                                        double alpha,
                                                        int step) {
  return RunIteration(rng_, player, alpha, step);
}

std::vector<Trajectory> DeepOSSBCFRSolver::RunIteration(std::mt19937* rng,
                                                        Player player,
                                                        double alpha,
                                                        int step) {
  alpha_ = alpha;
  node_touch_ = 0;
  ++iterations_;
  Trajectory value_trajectory;
  Trajectory policy_trajectory;
  Trajectory history_trajectory;
  // Sample a chace seed at the start of an iteration.
  ChanceData chance_data = root_state_->SampleChance(rng);
  // NOTE: We do not need to clearCache if the networks are never updated. So
  // the Cache should be clear by the learner. Don't do this:
  // value_eval_->ClearCache();
  UpdateRegrets(root_node_, player, 1, 1, 1, 1, value_trajectory,
                policy_trajectory, history_trajectory, step, rng, chance_data);
  return {value_trajectory, policy_trajectory, history_trajectory};
}

double DeepOSSBCFRSolver::UpdateRegrets(
    PublicNode* node, Player player, double player_reach, double opponent_reach,
    double ave_opponent_reach, double sampling_reach,
    Trajectory& value_trajectory, Trajectory& policy_trajectory,
    Trajectory& history_trajectory, int step, std::mt19937* rng,
    const ChanceData& chance_data) {
  State& state = *(node->GetState());
  universal_poker::UniversalPokerState* poker_state =
      static_cast<universal_poker::UniversalPokerState*>(node->GetState());
  state.SetChance(chance_data);
  // std::cout << state.ToString() << std::endl;
  if (state.IsTerminal()) {
    double value = state.PlayerReturn(player);
    return value;
  } else if (state.IsChanceNode()) {
    Action action = SampleAction(state.ChanceOutcomes(), dist_(*rng)).first;
    return UpdateRegrets(node->GetChild(action), player, player_reach,
                         opponent_reach, ave_opponent_reach, sampling_reach,
                         value_trajectory, policy_trajectory,
                         history_trajectory, step, rng, chance_data);
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
  std::vector<double> state_tensor = state.ObservationTensor(player);
  std::string obs_key = state.ObservationString(player);

  CFRInfoStateValues info_state_copy(legal_actions, kInitialTableValues);
  CFRInfoStateValues current_info_state(legal_actions, kInitialTableValues);
  std::vector<double> global_values(legal_actions.size(), 0.0);
  if (step != 1) {
    // get current policy
    CFRNetModel::InferenceInputs inference_input{is_key, legal_actions,
                                                 information_tensor};
    auto cfr_policy = current_policy_eval_[cur_player]
                          ->Inference(cur_player, inference_input)
                          .value;
    current_info_state.SetPolicy(cfr_policy);
    std::vector<double> uniform_strategy(legal_actions.size(),
                                         1.0 / legal_actions.size());
    info_state_copy.SetPolicy(uniform_strategy);
    // get global values.
    CFRNetModel::InferenceInputs global_inference_input{obs_key, legal_actions,
                                                        state_tensor};
    global_values = global_value_eval_[player]
                        ->Inference(player, global_inference_input)
                        .value;
  }
  double value = 0;
  double sub_value = 0;
  std::vector<double> m = global_values;
  int action_index = 0;
  if (cur_player == player) {
    action_index = info_state_copy.SampleActionIndex(0.0, dist_(*rng));
    double new_reach =
        current_info_state.current_policy[action_index] * player_reach;
    double new_sampling_reach =
        info_state_copy.current_policy[action_index] * sampling_reach;
    sub_value =
        UpdateRegrets(node->GetChild(legal_actions[action_index]), player,
                      new_reach, opponent_reach, ave_opponent_reach,
                      new_sampling_reach, value_trajectory, policy_trajectory,
                      history_trajectory, step, rng, chance_data);
    m[action_index] =
        m[action_index] + (sub_value - m[action_index]) /
                              info_state_copy.current_policy[action_index];
    for (int aidx = 0; aidx != legal_actions.size(); ++aidx) {
      value += m[aidx] * current_info_state.current_policy[aidx];
    }
  } else {
    action_index = current_info_state.SampleActionIndex(0.0, dist_(*rng));
    double new_reach =
        current_info_state.current_policy[action_index] * opponent_reach;
    double new_ave_reach = 1.0 / legal_actions.size() * ave_opponent_reach;
    double new_sampling_reach =
        current_info_state.current_policy[action_index] * sampling_reach;
    sub_value = UpdateRegrets(
        node->GetChild(legal_actions[action_index]), player, player_reach,
        new_reach, new_ave_reach, new_sampling_reach, value_trajectory,
        policy_trajectory, history_trajectory, step, rng, chance_data);
    m[action_index] =
        m[action_index] + (sub_value - m[action_index]) /
                              current_info_state.current_policy[action_index];
    for (int aidx = 0; aidx != legal_actions.size(); ++aidx) {
      value += m[aidx] * current_info_state.current_policy[aidx];
    }
  }

  history_trajectory.states.push_back(
      ReplayNode{obs_key, state_tensor, player, legal_actions,
                 current_info_state.current_policy, m, action_index, 1.0,
                 player_reach, opponent_reach, sampling_reach});

  if (cur_player == player) {
    std::vector<double> regret(m.size());
    for (int aidx = 0; aidx != legal_actions.size(); ++aidx) {
      regret[aidx] = (m[aidx] - value) * opponent_reach / sampling_reach;
    }
    double delta = poker_state->MaxUtility() - poker_state->MinUtility();
    value_trajectory.states.push_back(ReplayNode{
        is_key, information_tensor, cur_player, legal_actions,
        current_info_state.current_policy, regret, action_index, 1.0,
        player_reach, delta * delta * legal_actions.size() * ave_opponent_reach,
        sampling_reach});
  }

  if (cur_player == player) {
    if (!use_policy_net || use_tabular) {
      std::vector<double> policy(legal_actions.size());
      for (int aidx = 0; aidx < legal_actions.size(); ++aidx) {
        policy[aidx] = current_info_state.current_policy[aidx] * player_reach /
                       sampling_reach;
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
                     1.0,
                     player_reach,
                     opponent_reach,
                     sampling_reach});
    }
  }
  return value;
}

}  // namespace algorithms
}  // namespace open_spiel
