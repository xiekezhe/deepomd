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

#include "deep_rscfr_solver.h"

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
double DeepRSCFRSolver::UpdateRegrets(
    PublicNode* node, Player player, double player_reach, double opponent_reach,
    double sampling_reach, Trajectory& value_trajectory,
    Trajectory& policy_trajectory, int step, std::mt19937* rng,
    const ChanceData& chance_data) {
  State& state = *(node->GetState());
  state.SetChance(chance_data);
  if (state.IsTerminal()) {
    SPIEL_CHECK_DOUBLE_EQ(opponent_reach, sampling_reach);
    return state.PlayerReturn(player) * opponent_reach / sampling_reach;
  } else if (state.IsChanceNode()) {
    Action action = SampleAction(state.ChanceOutcomes(), dist_(*rng)).first;
    return UpdateRegrets(node->GetChild(action), player, player_reach,
                         opponent_reach, sampling_reach, value_trajectory,
                         policy_trajectory, step, rng, chance_data);
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

  // NOTE: why we need a copy here? don't copy, just create one.
  CFRInfoStateValues info_state_copy(legal_actions, kInitialTableValues);
  if (step != 1) {
    auto cfr_value = value_eval_[cur_player]->Inference(state);
    std::vector<double> cfr_regret = cfr_value.value;
    info_state_copy.SetRegret(cfr_regret);
  }
  info_state_copy.ApplyRegretMatching();
  double value = 0;
  std::vector<int> sampled_indexes(legal_actions.size());
  std::vector<double> child_values(legal_actions.size(), 0);

  if (cur_player != player) {
    // Sample at opponent nodes.
    int aidx = info_state_copy.SampleActionIndex(0.0, dist_(*rng));
    double new_reach = info_state_copy.current_policy[aidx] * opponent_reach;
    double new_sampling_reach =
        info_state_copy.current_policy[aidx] * sampling_reach;
    value =
        UpdateRegrets(node->GetChild(legal_actions[aidx]), player, player_reach,
                      new_reach, new_sampling_reach, value_trajectory,
                      policy_trajectory, step, rng, chance_data);
  } else {
    // Walk over sampled actions at my nodes
    absl::c_iota(sampled_indexes, 0);
    if (legal_actions.size() > kSampling) {
      std::cout << legal_actions << std::endl;
      absl::c_shuffle(sampled_indexes, *rng);
      sampled_indexes.resize(kSampling);
    }
    double new_sampling_reach =
        ((double)sampled_indexes.size() / legal_actions.size()) *
        sampling_reach;
    for (int aidx : sampled_indexes) {
      double child_reach = info_state_copy.current_policy[aidx] * player_reach;
      child_values[aidx] = UpdateRegrets(
          node->GetChild(legal_actions[aidx]), player, child_reach,
          opponent_reach, new_sampling_reach, value_trajectory,
          policy_trajectory, step, rng, chance_data);
      value += info_state_copy.current_policy[aidx] * child_values[aidx];
    }
  }

  if (cur_player == player) {
    // NOTE: only for debug.
    std::vector<double> regret(legal_actions.size());
    for (int aidx = 0; aidx < legal_actions.size(); ++aidx) {
      regret[aidx] = (child_values[aidx] - value);
    }
    if (!use_regret_net || use_tabular) {
      value_eval_[cur_player]->AccumulateCFRTabular(state, regret);
    }
    if (use_regret_net) {
      value_trajectory.states.push_back(ReplayNode{is_key, information_tensor,
                                                   cur_player, legal_actions,
                                                   regret, player_reach, 1.0});
    }
  }

  // If avg_type_ is KCurrent, update current player's strategy only.
  if (avg_type_ == AverageType::kCurrent && cur_player == player) {
    std::vector<double> policy(legal_actions.size());
    for (int aidx = 0; aidx < legal_actions.size(); ++aidx) {
      policy[aidx] = info_state_copy.current_policy[aidx];
    }
    if (!use_policy_net || use_tabular) {
      if (updated_infos.find(is_key) == updated_infos.end()) {
        policy_eval_->AccumulateCFRTabular(state, policy);
        updated_infos.insert(is_key);
      }
    }
    if (use_policy_net) {
      policy_trajectory.states.push_back(ReplayNode{is_key, information_tensor,
                                                    cur_player, legal_actions,
                                                    policy, player_reach, 1.0});
    }
  }

  if (avg_type_ == AverageType::kSimple && cur_player != player) {
    std::vector<double> policy(legal_actions.size());
    for (int aidx = 0; aidx < legal_actions.size(); ++aidx) {
      policy[aidx] = info_state_copy.current_policy[aidx];
    }
    if (!use_policy_net || use_tabular) {
      policy_eval_->AccumulateCFRTabular(state, policy);
    }
    if (use_policy_net) {
      policy_trajectory.states.push_back(ReplayNode{is_key, information_tensor,
                                                    cur_player, legal_actions,
                                                    policy, player_reach, 1.0});
    }
  }
  return value;
}

}  // namespace algorithms
}  // namespace open_spiel
