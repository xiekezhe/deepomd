#include "sbcfr1_solver.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <queue>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/games/universal_poker/logic/card_set.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {

namespace {
long long C(int n, int r) {
  if (r > n - r) r = n - r;
  long long ans = 1;
  int i;

  for (i = 1; i <= r; i++) {
    ans *= n - r + i;
    ans /= i;
  }
  return ans;
}
}  // namespace

SbCFR::SbCFR(const Game &game, std::shared_ptr<VPNetEvaluator> value_0_eval,
             std::shared_ptr<VPNetEvaluator> value_1_eval,
             std::shared_ptr<VPNetEvaluator> policy_0_eval,
             std::shared_ptr<VPNetEvaluator> policy_1_eval, bool use_regret_net,
             bool use_policy_net, bool use_tabular, std::mt19937 *rng,
             int cfr_batch_size, double alpha, double scale_up,
             double scale_down, Mode mode, int num_threads)
    : game_(static_cast<const universal_poker::UniversalPokerGame *>(&game)),
      rng_(rng),
      cfr_batch_size_(cfr_batch_size),
      step_(0),
      iterations_(0),
      dist_(0.0, 1.0),
      value_eval_{value_0_eval, value_1_eval},
      policy_eval_{policy_0_eval, policy_1_eval},
      use_regret_net(use_regret_net),
      use_policy_net(use_policy_net),
      use_tabular(use_tabular),
      mode_(mode),
      alpha_(alpha),
      scale_up_coeff_(scale_up),
      scale_down_coeff_(scale_down),
      alpha_ub_(1),
      alpha_lb_(1e-9),
      num_threads_(num_threads),
      acpc_game_(game_->GetACPCGame()),
      deck_(/*num_suits=*/acpc_game_->NumSuitsDeck(),
            /*num_ranks=*/acpc_game_->NumRanksDeck()) {
  num_hole_cards_ = acpc_game_->GetNbHoleCardsRequired();
  num_board_cards_ = acpc_game_->GetTotalNbBoardCards();
  player_outcomes_ = deck_.SampleCards(num_hole_cards_);
  player_outcome_arrays_.resize(player_outcomes_.size());
  absl::c_transform(player_outcomes_, player_outcome_arrays_.begin(),
                    [](const logic::CardSet &cs) { return cs.ToCardArray(); });
  if (num_board_cards_) {
    board_outcomes_ = deck_.SampleCards(num_board_cards_);
  }
  board_outcome_arrays_.resize(board_outcomes_.size());
  absl::c_transform(board_outcomes_, board_outcome_arrays_.begin(),
                    [](const logic::CardSet &cs) { return cs.ToCardArray(); });

  num_outcomes_ = player_outcomes_.size() * player_outcomes_.size();
  int num_cards = acpc_game_->NumSuitsDeck() * acpc_game_->NumRanksDeck();
  root_proba_ = 1.0 / (C(num_cards, num_hole_cards_) *
                       C(num_cards - num_hole_cards_, num_hole_cards_));
  board_proba_ = 1.0 / C(num_cards - 2 * num_hole_cards_, num_board_cards_);
  valid_matrix_ =
      Eigen::ArrayXXd(player_outcomes_.size(), player_outcomes_.size());
  for (int i = 0; i < player_outcomes_.size(); ++i) {
    for (int j = 0; j < player_outcomes_.size(); ++j) {
      logic::CardSet cs = player_outcomes_[i];
      cs.Combine(player_outcomes_[j]);
      valid_matrix_(i, j) = (cs.NumCards() == 2 * num_hole_cards_);
    }
  }
  int num_trees = num_threads_;
  for (int i = 0; i < num_trees; ++i) {
    trees_.emplace_back(game_->NewInitialState());
  }
  compared_cache_.resize(num_trees);
  alpha_cache_.resize(num_trees);
}

std::pair<Trajectory, Trajectory> SbCFR::RunIteration(Player player,
                                                      int iterations,
                                                      int step) {
  return RunIteration(rng_, player, iterations, step);
}

std::pair<Trajectory, Trajectory> SbCFR::RunIteration(std::mt19937 *rng,
                                                      Player player,
                                                      int iterations,
                                                      int step) {
  step_ = step;
  node_touch_ = 0;
  info_policies_.clear();
  info_regrets_.clear();
  Trajectory value_trajectory;
  Trajectory policy_trajectory;
  Eigen::ArrayXd valid_index = Eigen::ArrayXd::Ones(player_outcomes_.size());
  Eigen::ArrayXd player_reach = Eigen::ArrayXd::Ones(player_outcomes_.size());
  Eigen::ArrayXd average_player_reach =
      Eigen::ArrayXd::Ones(player_outcomes_.size());
  Eigen::ArrayXd opponent_reach = Eigen::ArrayXd::Ones(player_outcomes_.size());
  Eigen::ArrayXd average_opponent_reach =
      Eigen::ArrayXd::Ones(player_outcomes_.size());
  double sampler_reach = 1.0;
  std::vector<std::thread> threads;
  for (int tidx = 0; tidx != num_threads_; ++tidx) {
    threads.emplace_back([&, tidx]() {
      for (int iter = 0; iter != iterations; ++iter) {
        if (iter % num_threads_ != 0) {
          continue;
        }
        if (mode_ == Mode::kCFR || mode_ == Mode::kSbCFR ||
            mode_ == Mode::kPostSbCFR) {
          _enter_cfr(player, trees_[tidx].Root(), false, valid_index,
                     player_reach, average_player_reach, opponent_reach,
                     average_opponent_reach, sampler_reach, default_cards_,
                     value_trajectory, policy_trajectory, step, rng, tidx);
        } else {
          _oscfr_recursive(player, trees_[tidx].Root(), false, valid_index,
                           player_reach, average_player_reach, opponent_reach,
                           average_opponent_reach, sampler_reach,
                           default_cards_, value_trajectory, policy_trajectory,
                           step, rng, tidx);
        }
      }
    });
  }
  for (auto &t : threads) {
    t.join();
  }
  if ((mode_ == Mode::kSbCFR || mode_ == Mode::kSubSbCFR)) {
    if (use_regret_net) {
      for (auto &kv : info_regrets_) {
        value_trajectory.states.push_back(kv.second);
      }
    }
    if (use_policy_net) {
      for (auto &kv : info_policies_) {
        policy_trajectory.states.push_back(kv.second);
      }
    }
  }
  return {value_trajectory, policy_trajectory};
}

Eigen::ArrayXd SbCFR::_enter_cfr(
    Player player, algorithms::PublicNode *node, bool recover,
    const Eigen::ArrayXd &valid_index, const Eigen::ArrayXd &player_reach,
    const Eigen::ArrayXd &average_player_reach,
    const Eigen::ArrayXd &opponent_reach,
    const Eigen::ArrayXd &average_opponent_reach, double sampler_reach,
    const logic::CardSet &outcome, Trajectory &value_trajectory,
    Trajectory &policy_trajectory, int step, std::mt19937 *rng, int index) {
  Eigen::ArrayXd recover_values;
  Eigen::ArrayXd values;
  std::string public_state = outcome.ToString() + node->GetHistory();
  std::string player_state = std::to_string(player) + public_state;
  {
    std::lock_guard<std::mutex> guard(alpha_mutex_);
    if (alpha_dict_.find(public_state) == alpha_dict_.end()) {
      alpha_dict_[public_state] = alpha_;
    }
  }
  if ((mode_ == Mode::kSbCFR || mode_ == Mode::kSubSbCFR) && step > 1) {
    {
      std::lock_guard<std::mutex> guard(alpha_mutex_);
      alpha_cache_[index] = alpha_dict_[public_state];
    }
    // SbCFR requires to recover the regret first.
    recover_values =
        _cfr_recursive(player, node, true, valid_index, player_reach,
                       average_opponent_reach, sampler_reach, outcome,
                       value_trajectory, policy_trajectory, step, rng, index);
    recover_values *= average_player_reach;
    _adapt(player, public_state, recover_values, step, index);
  }
  values =
      _cfr_recursive(player, node, false, valid_index, player_reach,
                     opponent_reach, sampler_reach, outcome, value_trajectory,
                     policy_trajectory, step, rng, index);
  // value_eval_[player]->ClearCache();
  // policy_eval_[player]->ClearCache();
  // policy_eval_[player == 0 ? 1 : 0]->ClearCache();
  if (mode_ == Mode::kPostSbCFR || mode_ == Mode::kSubPostSbCFR) {
    {
      std::lock_guard<std::mutex> guard(alpha_mutex_);
      alpha_cache_[index] = alpha_dict_[public_state];
    }
    recover_values =
        _cfr_recursive(player, node, true, valid_index, player_reach,
                       average_opponent_reach, sampler_reach, outcome,
                       value_trajectory, policy_trajectory, step, rng, index);
    recover_values *= average_player_reach;
    _adapt(player, public_state, recover_values, step, index);
  }
  return values;
}

double SbCFR::old_adapt(double alpha, const std::vector<double> &values) {
  double sum_values = 0;
  for (auto &v : values) {
    sum_values += v;
  }
  if (sum_values > 0) {
    alpha *= scale_up_coeff_;
  } else if (sum_values < 0) {
    alpha *= scale_down_coeff_;
  }
  alpha = std::min(alpha_ub_, std::max(alpha, alpha_lb_));
  return alpha;
}

void SbCFR::_adapt(Player player, const std::string &public_state,
                   const Eigen::ArrayXd &recover_values, int step, int index) {
  std::lock_guard<std::mutex> guard(alpha_mutex_);
  int recover_step = step - 1;
  if (mode_ == Mode::kPostSbCFR || mode_ == Mode::kSubPostSbCFR) {
    recover_step = step;
  }
  std::string player_state = std::to_string(player) + public_state;
  std::string opponent_state =
      std::to_string(player == 0 ? 1 : 0) + public_state;
  recover_dict_[player_state] = recover_values;
  if (recover_step > 0 &&
      recover_dict_.find(player_state) != recover_dict_.end() &&
      recover_dict_.find(opponent_state) != recover_dict_.end()) {
    std::vector<double> cfr_values{recover_dict_[player_state].sum(),
                                   recover_dict_[opponent_state].sum()};
    alpha_dict_[public_state] =
        old_adapt(alpha_dict_[public_state], cfr_values);
  }
}

Eigen::ArrayXd SbCFR::_backward_update_regret(
    double delta, double alpha, const Eigen::ArrayXd &opponent_reach,
    const Eigen::ArrayXXd &values, int step) {
  int num_infs = values.rows();
  int num_actions = values.cols();
  Eigen::ArrayXd threshold =
      alpha * opponent_reach * delta * delta * num_actions / step;
  Eigen::ArrayXXd sorted_value = values;
  for (auto row_value : sorted_value.rowwise()) {
    std::sort(row_value.begin(), row_value.end());
  }
  Eigen::ArrayXXd available_values(values.rows(), values.cols());
  auto global_min_value = sorted_value.col(0) - Eigen::sqrt(threshold);
  for (int i = 0; i != num_actions; ++i) {
    Eigen::ArrayXd min_value;
    if (i == 0) {
      min_value = global_min_value;
    } else {
      min_value = sorted_value.col(i - 1);
    }
    auto max_value = sorted_value.col(i);
    double a = num_actions - i;
    auto b = -2 * sorted_value.rightCols(num_actions - i).rowwise().sum();
    auto c = sorted_value.rightCols(num_actions - i).pow(2).rowwise().sum() -
             threshold;
    auto x1 = (-b + Eigen::sqrt((b.pow(2) - 4 * a * c).max(0))) / (2 * a);
    auto x2 = -x1 - 2 * b / (2 * a);
    Eigen::ArrayXd min_x = x1.min(x2);
    for (int idx = 0; idx != num_infs; ++idx) {
      if (min_x(idx) < min_value(idx)) {
        min_x(idx) = global_min_value(idx);
      }
    }
    for (int idx = 0; idx != num_infs; ++idx) {
      if (min_x(idx) > max_value(idx)) {
        min_x(idx) = max_value(idx);
      }
    }
    available_values.col(i) = min_x;
  }
  // return values.rowwise().maxCoeff();
  return available_values.rowwise().maxCoeff();
}

Eigen::ArrayXd SbCFR::Evaluate(
    Player player, const universal_poker::UniversalPokerState *state,
    const Eigen::ArrayXd &p, const Eigen::ArrayXd &q,
    const logic::CardSet &outcome, int index) {
  // values = p * (value_matrx * valid_matrix_) \dot q.
  // For two players.
  double player_spent = state->acpc_state_.CurrentSpent(player);
  double other_spent =
      state->acpc_state_.TotalSpent() - state->acpc_state_.CurrentSpent(player);
  if (state->acpc_state_.NumFolded() >= state->acpc_game_->GetNbPlayers() - 1) {
    // Some one folded here.
    double scale_value = 0;
    if (state->acpc_state_.PlayerFolded(player)) {
      scale_value = -player_spent;
    } else {
      scale_value = other_spent;
    }
    return scale_value * root_proba_ * p *
           (valid_matrix_.matrix() * q.matrix()).array();
  } else {
    // Show down here. For two player limited game, all players should spent the
    // same money.
    SPIEL_CHECK_EQ(player_spent, other_spent);
    Eigen::ArrayXi ranks(player_outcomes_.size());
    for (int i = 0; i < player_outcomes_.size(); ++i) {
      logic::CardSet cs = player_outcomes_[i];
      cs.Combine(outcome);
      if (cs.NumCards() != (num_hole_cards_ + num_board_cards_)) {
        ranks(i) = -1;
      } else {
        ranks(i) = cs.RankCards();
      }
    }
    if (!compared_cache_[index].size()) {
      compared_cache_[index] =
          Eigen::ArrayXXd(player_outcomes_.size(), player_outcomes_.size());
      Eigen::ArrayXXd &compared = compared_cache_[index];
      for (int i = 0; i < player_outcomes_.size(); ++i) {
        for (int j = i; j < player_outcomes_.size(); ++j) {
          compared(i, j) =
              (ranks[i] > ranks[j]) ? 1 : ((ranks[i] < ranks[j]) ? -1 : 0);
          compared(j, i) = -compared(i, j);
        }
      }
      compared_cache_[index] = compared * valid_matrix_;
    }
    return other_spent * root_proba_ * p *
           (compared_cache_[index].matrix() * q.matrix()).array();
  }
}

Eigen::ArrayXd SbCFR::_dist_cfr_chance(
    Player player, algorithms::PublicNode *node, bool recover,
    const Eigen::ArrayXd &valid_index, const Eigen::ArrayXd &player_reach,
    const Eigen::ArrayXd &opponent_reach, double sampler_reach,
    const logic::CardSet &outcome, const logic::CardSet &cards,
    Trajectory &value_trajectory, Trajectory &policy_trajectory, int step,
    std::mt19937 *rng, int index) {
  logic::CardSet new_outcome = outcome;
  new_outcome.Combine(cards);
  algorithms::PublicNode *new_node = node->GetChild(node->GetChildActions()[0]);
  int check_num = num_hole_cards_ + new_outcome.NumCards();
  Eigen::ArrayXd new_valid_index = valid_index;
  Eigen::ArrayXd new_player_reach = player_reach;
  Eigen::ArrayXd new_opponent_reach = opponent_reach;
  for (int inf_id = 0; inf_id != player_outcomes_.size(); ++inf_id) {
    logic::CardSet check_cards = new_outcome;
    check_cards.Combine(player_outcomes_[inf_id]);
    if (check_cards.NumCards() != check_num) {
      new_valid_index(inf_id) = 0;
      new_player_reach(inf_id) = 0;
      new_opponent_reach(inf_id) = 0;
    }
  }
  return _cfr_recursive(player, new_node, recover, new_valid_index,
                        new_player_reach, new_opponent_reach, sampler_reach,
                        new_outcome, value_trajectory, policy_trajectory, step,
                        rng, index);
}

Eigen::ArrayXd SbCFR::_cfr_recursive(
    Player player, algorithms::PublicNode *node, bool recover,
    const Eigen::ArrayXd &valid_index, const Eigen::ArrayXd &player_reach,
    const Eigen::ArrayXd &opponent_reach, double sampler_reach,
    const logic::CardSet &outcome, Trajectory &value_trajectory,
    Trajectory &policy_trajectory, int step, std::mt19937 *rng, int index) {
  universal_poker::UniversalPokerState *state =
      static_cast<universal_poker::UniversalPokerState *>(node->GetState());
  if (state->IsTerminal()) {
    if (recover) {
      return Evaluate(player, state, valid_index, opponent_reach, outcome,
                      index);
    } else {
      // SPIEL_CHECK_TRUE((sampler_reach == 1).any());
      return Evaluate(player, state, valid_index,
                      opponent_reach / sampler_reach, outcome, index);
    }
  }
  if (state->IsChanceNode()) {
    Eigen::ArrayXd values = Eigen::ArrayXd::Zero(player_outcomes_.size());
    for (auto &cards : board_outcomes_) {
      compared_cache_[index] = Eigen::ArrayXXd(0, 0);
      values += _dist_cfr_chance(player, node, recover, valid_index,
                                 player_reach, opponent_reach, sampler_reach,
                                 outcome, cards, value_trajectory,
                                 policy_trajectory, step, rng, index);
    }
    values *= board_proba_;
    return values;
  }
  {
    std::lock_guard<std::mutex> guard(node_mutex_);
    node_touch_ += 1;
  }
  Player current_player = state->CurrentPlayer();
  std::vector<Action> legal_actions = state->LegalActions();
  // dim 0: information index, dim 1: action index.
  Eigen::ArrayXXd current_policies(player_outcomes_.size(),
                                   legal_actions.size());
  std::vector<std::vector<double>> information_tensors;
  std::vector<std::string> information_strings;
  for (int inf_id = 0; inf_id != player_outcomes_.size(); ++inf_id) {
    state->SetHoleCards(current_player, player_outcome_arrays_[inf_id]);
    state->SetBoardCards(outcome);
    information_tensors.emplace_back(state->State::InformationStateTensor());
    information_strings.emplace_back(state->State::InformationStateString());
    if (valid_index(inf_id)) {
      CFRInfoStateValues info_state_copy(legal_actions,
                                         DeepESCFRSolver::kInitialTableValues);
      if (step > 1) {
        if (recover) {
          std::vector<double> cfr_policy =
              policy_eval_[current_player]->Inference(*state).value;
          info_state_copy.SetPolicy(cfr_policy);
        } else {
          std::vector<double> cfr_regret =
              value_eval_[current_player]->Inference(*state).value;
          // cfr_regret =
          //     value_eval_[current_player]->GetCFRTabular(*state, true);
          info_state_copy.SetRegret(cfr_regret);
          info_state_copy.ApplyRegretMatchingUsingMax();
        }
      } else {
        info_state_copy.ApplyRegretMatchingUsingMax();
      }
      for (int axid = 0; axid != legal_actions.size(); ++axid) {
        current_policies(inf_id, axid) = info_state_copy.current_policy[axid];
      }
    } else {
      current_policies.row(inf_id) = 0;
    }
  }

  Eigen::ArrayXd values = Eigen::ArrayXd::Zero(player_outcomes_.size());
  Eigen::ArrayXXd m(player_outcomes_.size(), legal_actions.size());
  for (int axid = 0; axid != legal_actions.size(); ++axid) {
    if (current_player == player) {
      algorithms::PublicNode *next_node = node->GetChild(legal_actions[axid]);
      Eigen::ArrayXd new_player_reach =
          player_reach * current_policies.col(axid);
      m.col(axid) = _cfr_recursive(player, next_node, recover, valid_index,
                                   new_player_reach, opponent_reach,
                                   sampler_reach, outcome, value_trajectory,
                                   policy_trajectory, step, rng, index);
      values += current_policies.col(axid) * m.col(axid);
    } else {
      algorithms::PublicNode *next_node = node->GetChild(legal_actions[axid]);
      Eigen::ArrayXd new_opponent_reach =
          opponent_reach * current_policies.col(axid);
      values +=
          _cfr_recursive(player, next_node, recover, valid_index, player_reach,
                         new_opponent_reach, sampler_reach, outcome,
                         value_trajectory, policy_trajectory, step, rng, index);
    }
  }
  if (current_player == player) {
    if (recover) {
      double delta = state->MaxUtility() - state->MinUtility();
      Eigen::ArrayXd sum_opponent_reach =
          root_proba_ * opponent_reach.sum() *
          Eigen::ArrayXd::Ones(opponent_reach.size());
      // std::cout << information_strings[0] << " " << delta << " "
      //           << sum_opponent_reach[0] << std::endl;
      int recover_step = step - 1;
      values = _backward_update_regret(delta, alpha_cache_[index],
                                       sum_opponent_reach, m, recover_step);
      for (int inf_id = 0; inf_id != player_outcomes_.size(); ++inf_id) {
        if (!valid_index(inf_id)) {
          continue;
        }
        auto regret = (m.row(inf_id) - values(inf_id));
        std::vector<double> regret_vec(regret.begin(), regret.end());
        // save the infered regret value.
        {
          std::lock_guard<std::mutex> guard(info_regret_mutex_);
          if (info_regrets_.find(information_strings[inf_id]) ==
              info_regrets_.end()) {
            info_regrets_[information_strings[inf_id]] =
                ReplayNode{information_strings[inf_id],
                           information_tensors[inf_id],
                           current_player,
                           legal_actions,
                           regret_vec,
                           player_reach(inf_id),
                           (double)recover_step};
          }
        }
      }
    } else {
      for (int inf_id = 0; inf_id != player_outcomes_.size(); ++inf_id) {
        if (!valid_index(inf_id)) {
          continue;
        }
        auto regret = m.row(inf_id) - values(inf_id);
        std::vector<double> regret_vec(legal_actions.size());
        for (int aidx = 0; aidx != legal_actions.size(); ++aidx) {
          regret_vec[aidx] = regret(aidx) * sampler_reach;
        }
        CFRNetModel::TrainInputs train_input{
            information_strings[inf_id], legal_actions, legal_actions,
            information_tensors[inf_id], regret_vec,    1.0};
        if (mode_ == Mode::kCFR || mode_ == Mode::kSubCFR) {
          value_eval_[current_player]->AccumulateCFRTabular(train_input);
        } else {
          auto inst_replay_node =
              ReplayNode{information_strings[inf_id],
                         information_tensors[inf_id],
                         current_player,
                         legal_actions,
                         regret_vec,
                         player_reach(inf_id),
                         1.0 / sampler_reach / cfr_batch_size_};
          if (mode_ == Mode::kSbCFR || mode_ == Mode::kSubSbCFR) {
            {
              std::lock_guard<std::mutex> guard(info_regret_mutex_);
              if (info_regrets_.find(information_strings[inf_id]) !=
                  info_regrets_.end()) {
                info_regrets_[information_strings[inf_id]] += inst_replay_node;
              } else {
                SPIEL_CHECK_EQ(step, 1);
                info_regrets_[information_strings[inf_id]] = inst_replay_node;
              }
              if (!use_regret_net || use_tabular) {
                CFRNetModel::TrainInputs train_input{
                    information_strings[inf_id],
                    legal_actions,
                    legal_actions,
                    information_tensors[inf_id],
                    info_regrets_[information_strings[inf_id]].value,
                    info_regrets_[information_strings[inf_id]].weight};
                value_eval_[current_player]->SetCFRTabular(train_input);
              }
            }
          }
        }
      }
    }
  }

  if (current_player != player && !recover) {
    for (int inf_id = 0; inf_id != player_outcomes_.size(); ++inf_id) {
      if (!valid_index(inf_id)) {
        continue;
      }
      auto policy = current_policies.row(inf_id);
      if (!use_policy_net || use_tabular) {
        std::vector<double> policy_vec(policy.begin(), policy.end());
        for (auto &p : policy_vec) {
          p *= opponent_reach(inf_id) / sampler_reach / cfr_batch_size_;
        }
        CFRNetModel::TrainInputs train_input{
            information_strings[inf_id], legal_actions, legal_actions,
            information_tensors[inf_id], policy_vec,    1.0};
        policy_eval_[current_player]->AccumulateCFRTabular(train_input);
      }
      if (use_policy_net) {
        std::vector<double> policy_vec(policy.begin(), policy.end());
        auto inst_replay_node = ReplayNode{
            information_strings[inf_id],
            information_tensors[inf_id],
            current_player,
            legal_actions,
            policy_vec,
            opponent_reach(inf_id),
            opponent_reach(inf_id) / sampler_reach / cfr_batch_size_};
        {
          std::lock_guard<std::mutex> guard(info_policy_mutex_);
          if (info_policies_.find(information_strings[inf_id]) !=
              info_policies_.end()) {
            info_policies_[information_strings[inf_id]] += inst_replay_node;
          } else {
            info_policies_[information_strings[inf_id]] = inst_replay_node;
          }
        }
      }
    }
  }

  return values;
}

Eigen::ArrayXd SbCFR::_os_dist_cfr_chance(
    Player player, algorithms::PublicNode *node, bool recover,
    const Eigen::ArrayXd &valid_index, const Eigen::ArrayXd &player_reach,
    const Eigen::ArrayXd &average_player_reach,
    const Eigen::ArrayXd &opponent_reach,
    const Eigen::ArrayXd &average_opponent_reach, double sampler_reach,
    const logic::CardSet &outcome, const logic::CardSet &cards,
    Trajectory &value_trajectory, Trajectory &policy_trajectory, int step,
    std::mt19937 *rng, int index) {
  logic::CardSet new_outcome = outcome;
  new_outcome.Combine(cards);
  algorithms::PublicNode *new_node = node->GetChild(node->GetChildActions()[0]);
  int check_num = num_hole_cards_ + new_outcome.NumCards();
  Eigen::ArrayXd new_valid_index = valid_index;
  Eigen::ArrayXd new_player_reach = player_reach;
  Eigen::ArrayXd new_average_player_reach = average_player_reach;
  Eigen::ArrayXd new_opponent_reach = opponent_reach;
  Eigen::ArrayXd new_average_opponent_reach = average_opponent_reach;
  for (int inf_id = 0; inf_id != player_outcomes_.size(); ++inf_id) {
    logic::CardSet check_cards = new_outcome;
    check_cards.Combine(player_outcomes_[inf_id]);
    if (check_cards.NumCards() != check_num) {
      new_valid_index(inf_id) = 0;
      new_player_reach(inf_id) = 0;
      new_average_player_reach(inf_id) = 0;
      new_opponent_reach(inf_id) = 0;
      new_average_opponent_reach(inf_id) = 0;
    }
  }
  if (mode_ == Mode::kOSCFR) {
    return _oscfr_recursive(player, new_node, recover, new_valid_index,
                            new_player_reach, new_average_player_reach,
                            new_opponent_reach, new_average_opponent_reach,
                            sampler_reach, new_outcome, value_trajectory,
                            policy_trajectory, step, rng, index);
  } else {
    return _enter_cfr(player, new_node, recover, new_valid_index,
                      new_player_reach, new_average_player_reach,
                      new_opponent_reach, new_average_opponent_reach,
                      sampler_reach, new_outcome, value_trajectory,
                      policy_trajectory, step, rng, index);
  }
}

Eigen::ArrayXd SbCFR::_oscfr_recursive(
    Player player, algorithms::PublicNode *node, bool recover,
    const Eigen::ArrayXd &valid_index, const Eigen::ArrayXd &player_reach,
    const Eigen::ArrayXd &average_player_reach,
    const Eigen::ArrayXd &opponent_reach,
    const Eigen::ArrayXd &average_opponent_reach, double sampler_reach,
    const logic::CardSet &outcome, Trajectory &value_trajectory,
    Trajectory &policy_trajectory, int step, std::mt19937 *rng, int index) {
  universal_poker::UniversalPokerState *state =
      static_cast<universal_poker::UniversalPokerState *>(node->GetState());
  if (state->IsTerminal()) {
    return Evaluate(player, state, valid_index, opponent_reach / sampler_reach,
                    outcome, index);
  }
  if (state->IsChanceNode()) {
    double board_eta = 1.0 / board_outcomes_.size();
    double new_sampler_reach = sampler_reach * board_eta;
    std::uniform_int_distribution<int> dist(0, board_outcomes_.size() - 1);
    int board_index = dist(*rng);
    const logic::CardSet &card = board_outcomes_[board_index];
    compared_cache_[index] = Eigen::ArrayXXd(0, 0);
    Eigen::ArrayXd values = _os_dist_cfr_chance(
        player, node, recover, valid_index, player_reach, average_player_reach,
        opponent_reach, average_opponent_reach, new_sampler_reach, outcome,
        card, value_trajectory, policy_trajectory, step, rng, index);
    values *= board_proba_;
    return values;
  }

  {
    std::lock_guard<std::mutex> guard(node_mutex_);
    node_touch_ += 1;
  }
  Player current_player = state->CurrentPlayer();
  std::vector<Action> legal_actions = state->LegalActions();
  // dim 0: information index, dim 1: action index.
  Eigen::ArrayXXd current_policies(player_outcomes_.size(),
                                   legal_actions.size());
  Eigen::ArrayXXd average_policies(player_outcomes_.size(),
                                   legal_actions.size());
  std::vector<std::vector<double>> information_tensors;
  std::vector<std::string> information_strings;
  for (int inf_id = 0; inf_id != player_outcomes_.size(); ++inf_id) {
    state->SetHoleCards(current_player, player_outcome_arrays_[inf_id]);
    state->SetBoardCards(outcome);
    information_tensors.emplace_back(state->State::InformationStateTensor());
    information_strings.emplace_back(state->State::InformationStateString());
    if (valid_index(inf_id)) {
      CFRInfoStateValues info_state_copy(legal_actions,
                                         DeepESCFRSolver::kInitialTableValues);

      std::vector<double> cfr_policy(legal_actions.size(),
                                     1.0 / legal_actions.size());
      if (step > 1) {
        std::vector<double> cfr_regret =
            value_eval_[current_player]->GetCFRTabular(*state);
        info_state_copy.SetRegret(cfr_regret);
        info_state_copy.ApplyRegretMatchingUsingMax();
        cfr_policy = policy_eval_[current_player]->Inference(*state).value;
      } else {
        info_state_copy.ApplyRegretMatchingUsingMax();
      }
      for (int axid = 0; axid != legal_actions.size(); ++axid) {
        current_policies(inf_id, axid) = info_state_copy.current_policy[axid];
        average_policies(inf_id, axid) = cfr_policy[axid];
      }
    } else {
      current_policies.row(inf_id) = 0;
    }
  }

  std::uniform_int_distribution<int> dist(0, legal_actions.size() - 1);
  int axid = dist(*rng);
  double eta = 1.0 / legal_actions.size();

  Eigen::ArrayXd values = Eigen::ArrayXd::Zero(player_outcomes_.size());
  // NOTE: we must initialize m to zero as some action may not be updated.
  Eigen::ArrayXXd m =
      Eigen::ArrayXXd::Zero(player_outcomes_.size(), legal_actions.size());
  {
    double new_sampler_reach = sampler_reach * eta;
    if (current_player == player) {
      algorithms::PublicNode *next_node = node->GetChild(legal_actions[axid]);
      Eigen::ArrayXd new_player_reach =
          player_reach * current_policies.col(axid);
      Eigen::ArrayXd new_average_player_reach =
          average_player_reach * average_policies.col(axid);
      m.col(axid) = _oscfr_recursive(
          player, next_node, recover, valid_index, new_player_reach,
          new_average_player_reach, opponent_reach, average_opponent_reach,
          new_sampler_reach, outcome, value_trajectory, policy_trajectory, step,
          rng, index);
      values += current_policies.col(axid) * m.col(axid);
    } else {
      algorithms::PublicNode *next_node = node->GetChild(legal_actions[axid]);
      Eigen::ArrayXd new_opponent_reach =
          opponent_reach * current_policies.col(axid);
      Eigen::ArrayXd new_average_opponent_reach =
          average_opponent_reach * average_policies.col(axid);
      values += _oscfr_recursive(player, next_node, recover, valid_index,
                                 player_reach, average_player_reach,
                                 new_opponent_reach, new_average_opponent_reach,
                                 new_sampler_reach, outcome, value_trajectory,
                                 policy_trajectory, step, rng, index);
    }
  }
  if (current_player == player && !recover) {
    for (int inf_id = 0; inf_id != player_outcomes_.size(); ++inf_id) {
      if (!valid_index(inf_id)) {
        continue;
      }
      auto regret = m.row(inf_id) - values(inf_id);
      std::vector<double> regret_vec(legal_actions.size());

      for (int aidx = 0; aidx != legal_actions.size(); ++aidx) {
        regret_vec[aidx] = regret(aidx);
      }
      CFRNetModel::TrainInputs train_input{
          information_strings[inf_id], legal_actions, legal_actions,
          information_tensors[inf_id], regret_vec,    1.0};
      value_eval_[current_player]->AccumulateCFRTabular(train_input);
    }
  }

  if (current_player != player && !recover) {
    for (int inf_id = 0; inf_id != player_outcomes_.size(); ++inf_id) {
      if (!valid_index(inf_id)) {
        continue;
      }
      auto policy = current_policies.row(inf_id);
      std::vector<double> policy_vec(policy.begin(), policy.end());
      for (auto &p : policy_vec) {
        p *= opponent_reach(inf_id) / sampler_reach / cfr_batch_size_;
      }
      CFRNetModel::TrainInputs train_input{
          information_strings[inf_id], legal_actions, legal_actions,
          information_tensors[inf_id], policy_vec,    1.0};
      policy_eval_[current_player]->AccumulateCFRTabular(train_input);
      if (use_policy_net) {
        std::vector<double> policy_vec(policy.begin(), policy.end());
        auto inst_replay_node = ReplayNode{
            information_strings[inf_id],
            information_tensors[inf_id],
            current_player,
            legal_actions,
            policy_vec,
            opponent_reach(inf_id),
            opponent_reach(inf_id) / sampler_reach / cfr_batch_size_};
        {
          std::lock_guard<std::mutex> guard(info_policy_mutex_);
          if (info_policies_.find(information_strings[inf_id]) !=
              info_policies_.end()) {
            info_policies_[information_strings[inf_id]] += inst_replay_node;
          } else {
            info_policies_[information_strings[inf_id]] = inst_replay_node;
          }
        }
      }
    }
  }

  return values;
}
}  // namespace algorithms
}  // namespace open_spiel