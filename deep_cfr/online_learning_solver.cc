#include "online_learning_solver.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <numeric>
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

OnlineLearningOptimization::OnlineLearningOptimization(
    const Game &game, std::shared_ptr<VPNetEvaluator> value_0_eval,
    std::shared_ptr<VPNetEvaluator> value_1_eval,
    std::shared_ptr<VPNetEvaluator> policy_0_eval,
    std::shared_ptr<VPNetEvaluator> policy_1_eval, bool use_regret_net,
    bool use_policy_net, bool use_tabular, std::mt19937 *rng,
    int cfr_batch_size, double alpha, Mode mode, AverageType type)
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
      average_type_(type),
      alpha_(alpha),
      scale_up_coeff_(1.1),
      scale_down_coeff_(0.9),
      scale_ub_(-0.05),
      scale_lb_(-0.2),
      alpha_ub_(1),
      alpha_lb_(1e-9),
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
  int num_trees = 1;
  for (int i = 0; i < num_trees; ++i) {
    trees_.emplace_back(game_->NewInitialState());
  }
  compared_cache_.resize(num_trees);
}

std::pair<Trajectory, Trajectory> OnlineLearningOptimization::RunIteration(
    Player player, int step) {
  return RunIteration(rng_, player, step);
}

std::pair<Trajectory, Trajectory> OnlineLearningOptimization::RunIteration(
    std::mt19937 *rng, Player player, int step) {
  if (step > step_) {
    step_ = step;
    iterations_ = 0;
  }
  ++iterations_;
  node_touch_ = 0;
  Trajectory value_trajectory;
  Trajectory policy_trajectory;
  Eigen::ArrayXd valid_index = Eigen::ArrayXd::Ones(player_outcomes_.size());
  Eigen::ArrayXd player_reach = Eigen::ArrayXd::Ones(player_outcomes_.size());
  Eigen::ArrayXd average_player_reach =
      Eigen::ArrayXd::Ones(player_outcomes_.size());
  Eigen::ArrayXd opponent_reach = Eigen::ArrayXd::Ones(player_outcomes_.size());
  Eigen::ArrayXd average_opponent_reach =
      Eigen::ArrayXd::Ones(player_outcomes_.size());
  Eigen::ArrayXd sampler_reach = Eigen::ArrayXd::Ones(player_outcomes_.size());
  Eigen::ArrayXd weights = Eigen::ArrayXd::Zero(player_outcomes_.size());
  _enter_cfr(player, trees_[0].Root(), valid_index, player_reach,
             average_player_reach, opponent_reach, average_opponent_reach,
             sampler_reach, default_cards_, weights, value_trajectory,
             policy_trajectory, step, rng, 0);
  return {value_trajectory, policy_trajectory};
}

Eigen::ArrayXd OnlineLearningOptimization::_enter_cfr(
    Player player, algorithms::PublicNode *node,
    const Eigen::ArrayXd &valid_index, const Eigen::ArrayXd &player_reach,
    const Eigen::ArrayXd &average_player_reach,
    const Eigen::ArrayXd &opponent_reach,
    const Eigen::ArrayXd &average_opponent_reach,
    const Eigen::ArrayXd &sampler_reach, const logic::CardSet &outcome,
    Eigen::ArrayXd &weights, Trajectory &value_trajectory,
    Trajectory &policy_trajectory, int step, std::mt19937 *rng, int index) {
  Eigen::ArrayXd values;
  values = _cfr_recursive(
      player, node, valid_index, player_reach, average_player_reach,
      opponent_reach, average_opponent_reach, sampler_reach, outcome, weights,
      value_trajectory, policy_trajectory, step, rng, index);
  value_eval_[player]->ClearCache();
  policy_eval_[player]->ClearCache();
  policy_eval_[player == 0 ? 1 : 0]->ClearCache();
  return values;
}

Eigen::ArrayXd OnlineLearningOptimization::Evaluate(
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

Eigen::ArrayXd OnlineLearningOptimization::_FTRLProx(
    const Eigen::ArrayXd &weights, const Eigen::ArrayXXd &values,
    const Eigen::ArrayXXd &policy, int step) {
  int num_infs = values.rows();
  int num_actions = values.cols();
  Eigen::ArrayXd ret = Eigen::ArrayXd(num_infs);
  for (int i = 0; i != num_infs; ++i) {
    Eigen::ArrayXd segment = values.row(i);
    std::stable_sort(segment.begin(), segment.end());
    int left_bound = segment.size() - 1;
    double target = 0;
    for (; left_bound >= 0; --left_bound) {
      target = (values.row(i) - segment(left_bound)).max(0).sum();
      if (target > weights(i)) {
        break;
      }
    }
    SPIEL_CHECK_LT(left_bound, segment.size() - 1);
    // Z is in range (segment(left_bound), segement(left_bound + 1)]
    double s = 0;
    for (int seg = left_bound + 1; seg != segment.size(); ++seg) {
      s += segment(seg);
    }

    double v = (s - weights(i)) / (segment.size() - left_bound - 1);
    ret(i) = v;
    SPIEL_CHECK_FALSE(std::isnan(v));
    SPIEL_CHECK_LE(ret(i), segment(left_bound + 1) + 1e-5);
    double Z = (segment - v).max(0).sum();
    SPIEL_CHECK_FLOAT_NEAR(Z, weights(i), 1e-9);
    if (Z > 0) {
      Eigen::ArrayXd x = (segment - v).max(0) / Z;
      double true_value =
          (x * segment).sum() - 0.5 * weights(i) * (x * x).sum();
      double app_value = v + 0.5 * weights(i) * (x * x).sum();
      SPIEL_CHECK_DOUBLE_EQ(true_value, app_value);
    }
  }
  return ret;
}

Eigen::ArrayXd OnlineLearningOptimization::_OMDProx(
    const Eigen::ArrayXd &weights, const Eigen::ArrayXXd &values,
    const Eigen::ArrayXXd &policy, int step) {
  return _FTRLProx(weights, values + (policy.colwise() * weights), policy,
                   step);
}

Eigen::ArrayXd OnlineLearningOptimization::_OMDCFRProx(
    const Eigen::ArrayXd &weights, const Eigen::ArrayXd &old_weights,
    const Eigen::ArrayXXd &values, const Eigen::ArrayXXd &policy, int step) {
  return _FTRLProx(weights, values + (policy.colwise() * old_weights), policy,
                   step);
}

Eigen::ArrayXd OnlineLearningOptimization::_dist_cfr_chance(
    Player player, algorithms::PublicNode *node,
    const Eigen::ArrayXd &valid_index, const Eigen::ArrayXd &player_reach,
    const Eigen::ArrayXd &average_player_reach,
    const Eigen::ArrayXd &opponent_reach,
    const Eigen::ArrayXd &average_opponent_reach,
    const Eigen::ArrayXd &sampler_reach, const logic::CardSet &outcome,
    Eigen::ArrayXd &weights, const logic::CardSet &cards,
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
  return _cfr_recursive(player, new_node, new_valid_index, new_player_reach,
                        new_average_player_reach, new_opponent_reach,
                        new_average_opponent_reach, sampler_reach, new_outcome,
                        weights, value_trajectory, policy_trajectory, step, rng,
                        index);
}

Eigen::ArrayXd OnlineLearningOptimization::_cfr_recursive(
    Player player, algorithms::PublicNode *node,
    const Eigen::ArrayXd &valid_index, const Eigen::ArrayXd &player_reach,
    const Eigen::ArrayXd &average_player_reach,
    const Eigen::ArrayXd &opponent_reach,
    const Eigen::ArrayXd &average_opponent_reach,
    const Eigen::ArrayXd &sampler_reach, const logic::CardSet &outcome,
    Eigen::ArrayXd &weights, Trajectory &value_trajectory,
    Trajectory &policy_trajectory, int step, std::mt19937 *rng, int index) {
  universal_poker::UniversalPokerState *state =
      static_cast<universal_poker::UniversalPokerState *>(node->GetState());
  if (state->IsTerminal()) {
    Eigen::ArrayXd evaluate_rach = opponent_reach;
    // FTRL requires to compute average loss.
    if (mode_ == Mode::kFTRL || mode_ == Mode::kFTRLCFR) {
      evaluate_rach =
          (average_opponent_reach * (step - 1) + opponent_reach) / step;
    }
    auto values = Evaluate(player, state, valid_index,
                           evaluate_rach / sampler_reach, outcome, index);
    if (mode_ == Mode::kFTRLCFR || mode_ == Mode::kOMDCFR) {
      // FTRL CFR computes weights according to regrets of CFR.
      // We compute the regrets of current iteartion by weights.
      weights = Evaluate(player, state, valid_index,
                         opponent_reach / sampler_reach, outcome, index);
    } else {
      weights = Eigen::ArrayXd::Zero(weights.size()) * valid_index;
    }
    return values;
  }
  if (state->IsChanceNode()) {
    Eigen::ArrayXd values = Eigen::ArrayXd::Zero(player_outcomes_.size());
    Eigen::ArrayXXd children_weights =
        Eigen::ArrayXXd::Zero(player_outcomes_.size(), board_outcomes_.size());
    for (int cidx = 0; cidx != board_outcomes_.size(); ++cidx) {
      compared_cache_[index] = Eigen::ArrayXXd(0, 0);
      auto &cards = board_outcomes_[cidx];
      Eigen::ArrayXd cw(children_weights.rows());
      Eigen::ArrayXd new_opponent_reach = opponent_reach * board_proba_;
      Eigen::ArrayXd new_average_opponent_reach =
          average_opponent_reach * board_proba_;
      values += _dist_cfr_chance(player, node, valid_index, player_reach,
                                 average_player_reach, new_opponent_reach,
                                 new_average_opponent_reach, sampler_reach,
                                 outcome, cw, cards, value_trajectory,
                                 policy_trajectory, step, rng, index);
      children_weights.col(cidx) = cw;
    }
    weights = children_weights.rowwise().sum() * valid_index;
    return values;
  }

  node_touch_ += 1;
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
      if (step > 1) {
        // This gives the cumulative strategy.
        std::vector<double> cfr_policy =
            policy_eval_[current_player]->Inference(*state).value;
        info_state_copy.SetCumulatePolicy(cfr_policy);
        // This gives the current strategy.
        std::vector<double> cfr_regret =
            value_eval_[current_player]->Inference(*state).value;
        info_state_copy.SetRegret(cfr_regret);
        // Not necessary, just for normalization.
        info_state_copy.ApplyRegretMatching();
      } else {
        info_state_copy.ApplyRegretMatching();
      }
      for (int aidx = 0; aidx != legal_actions.size(); ++aidx) {
        current_policies(inf_id, aidx) = info_state_copy.current_policy[aidx];
        average_policies(inf_id, aidx) =
            info_state_copy.cumulative_policy[aidx];
      }
    } else {
      current_policies.row(inf_id) = 0;
      average_policies.row(inf_id) = 0;
    }
  }

  Eigen::ArrayXd values = Eigen::ArrayXd::Zero(player_outcomes_.size());
  Eigen::ArrayXXd m(player_outcomes_.size(), legal_actions.size());
  Eigen::ArrayXXd children_weights =
      Eigen::ArrayXXd::Zero(player_outcomes_.size(), legal_actions.size());
  for (int aidx = 0; aidx != legal_actions.size(); ++aidx) {
    Eigen::ArrayXd cw = Eigen::ArrayXd::Zero(children_weights.rows());
    if (current_player == player) {
      algorithms::PublicNode *next_node = node->GetChild(legal_actions[aidx]);
      Eigen::ArrayXd new_player_reach = player_reach;
      Eigen::ArrayXd new_average_player_reach = average_player_reach;
      new_player_reach *= current_policies.col(aidx);
      new_average_player_reach *= average_policies.col(aidx);
      m.col(aidx) =
          _cfr_recursive(player, next_node, valid_index, new_player_reach,
                         new_average_player_reach, opponent_reach,
                         average_opponent_reach, sampler_reach, outcome, cw,
                         value_trajectory, policy_trajectory, step, rng, index);
      values += current_policies.col(aidx) * m.col(aidx);
    } else {
      algorithms::PublicNode *next_node = node->GetChild(legal_actions[aidx]);
      Eigen::ArrayXd new_opponent_reach = opponent_reach;
      Eigen::ArrayXd new_average_opponent_reach = average_opponent_reach;
      new_opponent_reach *= current_policies.col(aidx);
      new_average_opponent_reach *= average_policies.col(aidx);
      values += _cfr_recursive(
          player, next_node, valid_index, player_reach, average_player_reach,
          new_opponent_reach, new_average_opponent_reach, sampler_reach,
          outcome, cw, value_trajectory, policy_trajectory, step, rng, index);
    }
    children_weights.col(aidx) = cw;
  }
  // update weights
  Eigen::ArrayXd current_weights = weights;
  Eigen::ArrayXd old_weights = weights;
  if (current_player == player &&
      (mode_ == Mode::kFTRLCFR || mode_ == Mode::kOMDCFR)) {
    // Do one CFR / CFR Plus and compute the weights for FTRL and OMD.
    for (int inf_id = 0; inf_id != player_outcomes_.size(); ++inf_id) {
      weights(inf_id) = 0.0;
      current_weights(inf_id) = 0.0;
      old_weights(inf_id) = 0.0;
      if (!valid_index(inf_id)) {
        continue;
      }
      if (info_regrets_.find(information_strings[inf_id]) ==
          info_regrets_.end()) {
        // use the initial value to prevent 0 regret.
        info_regrets_[information_strings[inf_id]] =
            std::vector<double>(legal_actions.size(), 1e-6);
      }
      // old weights is L1 norm of the cumulative regrets.
      auto &old_cum_regrets = info_regrets_[information_strings[inf_id]];
      for (int aidx = 0; aidx != legal_actions.size(); ++aidx) {
        // normalize the weights vector by sqrt(step).
        old_weights(inf_id) += std::max(old_cum_regrets[aidx], 0.0);
      }
      SPIEL_CHECK_GT(old_weights(inf_id), 0.0);
      // here ''weights'' is the expected value of current iteration.
      std::vector<double> old_regret =
          info_regrets_[information_strings[inf_id]];
      Eigen::ArrayXd cpolicy(old_regret.size());
      double sum_old_regret = 0;
      for (int aidx = 0; aidx != legal_actions.size(); ++aidx) {
        sum_old_regret += std::max(old_regret[aidx], 0.0);
      }
      for (int aidx = 0; aidx != legal_actions.size(); ++aidx) {
        cpolicy(aidx) = std::max(old_regret[aidx], 0.0) / sum_old_regret;
        // SPIEL_CHECK_FLOAT_NEAR(cpolicy(aidx), current_policies(inf_id, aidx),
        //                        0.01);
        weights(inf_id) += cpolicy(aidx) * children_weights(inf_id, aidx);
      }
      Eigen::ArrayXd regret = children_weights.row(inf_id) - weights(inf_id);
      // record the cumulative regrets computed by CFR.
      std::vector<double> regret_vec(legal_actions.size());
      for (int aidx = 0; aidx != legal_actions.size(); ++aidx) {
        if (mode_ == Mode::kOMDCFR) {
          // the update for CFRPlus
          regret_vec[aidx] = std::max(old_regret[aidx] + regret(aidx), 0.0);
        } else {
          // the update for CFR
          regret_vec[aidx] = old_regret[aidx] + regret(aidx);
        }
      }
      info_regrets_[information_strings[inf_id]] = regret_vec;
      // current weights is L1 norm of the cumulative regrets.
      auto &cum_regrets = info_regrets_[information_strings[inf_id]];
      for (int aidx = 0; aidx != legal_actions.size(); ++aidx) {
        // normalize the weights vector by sqrt(step).
        current_weights(inf_id) += std::max(cum_regrets[aidx], 0.0);
      }
      SPIEL_CHECK_GT(current_weights(inf_id), 0.0);
    }
    old_weights /= std::sqrt(double(step));
    old_weights *= valid_index;
    current_weights /= std::sqrt(double(step));
    current_weights *= valid_index;
    SPIEL_CHECK_TRUE((current_weights >= 0).all());
  } else if (current_player == player) {
    weights = (2 + 2 * children_weights.rowwise().maxCoeff()) * valid_index;
    current_weights = weights;
  } else {
    weights = children_weights.rowwise().sum() * valid_index;
    current_weights = weights;
  }
  if (current_player == player) {
    if (mode_ == Mode::kFTRL || mode_ == Mode::kFTRLCFR) {
      double delta = state->MaxUtility() - state->MinUtility();
      // we need to divide sqrt(step) because m is the ''avearge value.'',
      // instead of the cumulative values.
      Eigen::ArrayXd eta = alpha_ * current_weights / std::sqrt(double(step));
      Eigen::ArrayXd lambda =
          _FTRLProx(eta, m + 1e-6 / step, current_policies, step);
      Eigen::ArrayXXd next_policy = (m.colwise() - lambda + 1e-6 / step).max(0);
      // Set next strategy.
      for (int inf_id = 0; inf_id != player_outcomes_.size(); ++inf_id) {
        if (!valid_index(inf_id)) {
          continue;
        }
        // next policy should be well defined if eta(inf_id) > 0
        std::vector<double> np(legal_actions.size(), 0.0);
        double Z = next_policy.row(inf_id).sum();
        SPIEL_CHECK_DOUBLE_EQ(Z, eta(inf_id));
        for (int aidx = 0; aidx != legal_actions.size(); ++aidx) {
          np[aidx] = next_policy(inf_id, aidx) / Z;
        }
        CFRNetModel::TrainInputs train_input{information_strings[inf_id],
                                             legal_actions,
                                             legal_actions,
                                             information_tensors[inf_id],
                                             np,
                                             1.0};
        value_eval_[current_player]->SetCFRTabular(train_input);
        // recursively updates parent's value vector.
        values(inf_id) =
            lambda(inf_id) +
            0.5 * next_policy.row(inf_id).pow(2).sum() / eta(inf_id);
        if (mode_ == Mode::kFTRLCFR) {
          values(inf_id) = lambda(inf_id);
        }
        // Eigen::ArrayXd FTRL_regret = m.row(inf_id) - values(inf_id);
        // std::cout << FTRL_regret << std::endl;
        // std::cout << info_regrets_[information_strings[inf_id]] << std::endl;
        SPIEL_CHECK_FALSE(Eigen::isnan(values).any());
      }
      // std::cout << information_strings[0] << std::endl;
      // std::cout << next_policy << std::endl;
      // std::cout << lambda << std::endl;
      // std::cout << values << std::endl;
    } else if (mode_ == Mode::kOMD || mode_ == Mode::kOMDCFR) {
      double delta = state->MaxUtility() - state->MinUtility();
      // OMD requires the weights to increase in the rate of sqrt(step).
      Eigen::ArrayXd eta = alpha_ * current_weights * std::sqrt(double(step));
      Eigen::ArrayXd old_eta = eta;
      Eigen::ArrayXd lambda = Eigen::ArrayXd(player_outcomes_.size());
      if (mode_ == Mode::kOMD) {
        lambda = _OMDProx(eta, m, current_policies, step);
      } else if (mode_ == Mode::kOMDCFR) {
        old_eta = alpha_ * old_weights * std::sqrt(double(step));
        lambda = _OMDCFRProx(eta, old_eta, m, current_policies, step);
      }
      Eigen::ArrayXXd next_policy =
          ((current_policies.colwise() * old_eta) + (m.colwise() - lambda))
              .max(0.0);
      // Set next strategy.
      for (int inf_id = 0; inf_id != player_outcomes_.size(); ++inf_id) {
        if (!valid_index(inf_id)) {
          continue;
        }
        // next policy should be well defined if eta(inf_id) > 0
        std::vector<double> np(legal_actions.size(), 0.0);
        double Z = next_policy.row(inf_id).sum();
        SPIEL_CHECK_DOUBLE_EQ(Z, eta(inf_id));
        for (int aidx = 0; aidx != legal_actions.size(); ++aidx) {
          np[aidx] = next_policy(inf_id, aidx) / Z;
        }
        CFRNetModel::TrainInputs train_input{information_strings[inf_id],
                                             legal_actions,
                                             legal_actions,
                                             information_tensors[inf_id],
                                             np,
                                             1.0};
        value_eval_[current_player]->SetCFRTabular(train_input);
        // recursively updates parent's value vector.
        values(inf_id) =
            lambda(inf_id) -
            0.5 * eta(inf_id) * current_policies.row(inf_id).pow(2).sum() +
            0.5 * next_policy.row(inf_id).pow(2).sum() / eta(inf_id);
        if (mode_ == Mode::kOMDCFR) {
          values(inf_id) = lambda(inf_id);
        }
        SPIEL_CHECK_FALSE(Eigen::isnan(values).any());
      }
    } else if (mode_ == Mode::kCFRPLUS) {
      // CFR Plus
      for (int inf_id = 0; inf_id != player_outcomes_.size(); ++inf_id) {
        if (!valid_index(inf_id)) {
          continue;
        }
        auto regret = m.row(inf_id) - values(inf_id);
        std::vector<double> regret_vec(legal_actions.size());
        CFRNetModel::InferenceInputs inf_input{information_strings[inf_id],
                                               legal_actions};
        std::vector<double> old_regret =
            value_eval_[current_player]->GetCFRTabular(inf_input);
        for (int aidx = 0; aidx != legal_actions.size(); ++aidx) {
          regret_vec[aidx] = std::max(old_regret[aidx] + regret(aidx), 0.0);
        }
        CFRNetModel::TrainInputs train_input{
            information_strings[inf_id], legal_actions, legal_actions,
            information_tensors[inf_id], regret_vec,    1.0};
        value_eval_[current_player]->SetCFRTabular(train_input);
      }
    } else {
      // Vanilla CFR
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
  }

  if (average_type_ == AverageType::kCurrent && current_player == player) {
    for (int inf_id = 0; inf_id != player_outcomes_.size(); ++inf_id) {
      if (!valid_index(inf_id)) {
        continue;
      }
      double policy_weight = player_reach(inf_id) / sampler_reach(inf_id);
      auto policy = policy_weight * current_policies.row(inf_id);
      std::vector<double> policy_vec(policy.begin(), policy.end());
      CFRNetModel::TrainInputs train_input{
          information_strings[inf_id], legal_actions, legal_actions,
          information_tensors[inf_id], policy_vec,    1.0};
      policy_eval_[current_player]->AccumulateCFRTabular(train_input);
    }
  } else if ((average_type_ == AverageType::kOpponent ||
              average_type_ == AverageType::kLinearOpponent) &&
             current_player != player) {
    for (int inf_id = 0; inf_id != player_outcomes_.size(); ++inf_id) {
      if (!valid_index(inf_id)) {
        continue;
      }
      double policy_weight = opponent_reach(inf_id) / sampler_reach(inf_id);
      if (average_type_ == AverageType::kLinearOpponent) {
        policy_weight *= step;
      }
      auto policy = policy_weight * current_policies.row(inf_id);
      std::vector<double> policy_vec(policy.begin(), policy.end());
      CFRNetModel::TrainInputs train_input{
          information_strings[inf_id], legal_actions, legal_actions,
          information_tensors[inf_id], policy_vec,    1.0};
      policy_eval_[current_player]->AccumulateCFRTabular(train_input);
    }
  }

  return values;
}

}  // namespace algorithms
}  // namespace open_spiel