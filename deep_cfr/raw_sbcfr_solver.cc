#include "raw_sbcfr_solver.h"

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

RawSbCFR::RawSbCFR(const Game &game,
                   std::shared_ptr<VPNetEvaluator> value_0_eval,
                   std::shared_ptr<VPNetEvaluator> value_1_eval,
                   std::shared_ptr<VPNetEvaluator> policy_0_eval,
                   std::shared_ptr<VPNetEvaluator> policy_1_eval,
                   bool use_regret_net, bool use_policy_net, bool use_tabular,
                   std::mt19937 *rng, int cfr_batch_size, double alpha,
                   Mode mode, AverageType type, WeightType wtype)
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
      weight_type_(wtype),
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
  alpha_cache_.resize(num_trees);
}

std::pair<Trajectory, Trajectory> RawSbCFR::RunIteration(Player player,
                                                         int step) {
  return RunIteration(rng_, player, step);
}

std::pair<Trajectory, Trajectory> RawSbCFR::RunIteration(std::mt19937 *rng,
                                                         Player player,
                                                         int step) {
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
  Eigen::ArrayXd predicted_values =
      Eigen::ArrayXd::Zero(player_outcomes_.size());
  _enter_cfr(player, trees_[0].Root(), false, valid_index, player_reach,
             average_player_reach, opponent_reach, average_opponent_reach,
             sampler_reach, default_cards_, predicted_values, value_trajectory,
             policy_trajectory, step, rng, 0);
  return {value_trajectory, policy_trajectory};
}

Eigen::ArrayXd RawSbCFR::_enter_cfr(
    Player player, algorithms::PublicNode *node, bool recover,
    const Eigen::ArrayXd &valid_index, const Eigen::ArrayXd &player_reach,
    const Eigen::ArrayXd &average_player_reach,
    const Eigen::ArrayXd &opponent_reach,
    const Eigen::ArrayXd &average_opponent_reach,
    const Eigen::ArrayXd &sampler_reach, const logic::CardSet &outcome,
    Eigen::ArrayXd &predicted_values, Trajectory &value_trajectory,
    Trajectory &policy_trajectory, int step, std::mt19937 *rng, int index) {
  Eigen::ArrayXd values;
  std::string public_state = outcome.ToString() + node->GetHistory();
  std::string player_state = std::to_string(player) + public_state;
  alpha_cache_[index] = alpha_;
  values = _cfr_recursive(
      player, node, false, valid_index, player_reach, average_player_reach,
      opponent_reach, average_opponent_reach, sampler_reach, outcome,
      predicted_values, value_trajectory, policy_trajectory, step, rng, index);
  value_eval_[player]->ClearCache();
  policy_eval_[player]->ClearCache();
  policy_eval_[player == 0 ? 1 : 0]->ClearCache();
  _adapt(player, public_state, values, step, index);
  return values;
}

double RawSbCFR::old_adapt(double alpha, const std::vector<double> &values) {
  double sum_values = 0;
  double max_values = 0;
  for (auto &v : values) {
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

void RawSbCFR::_adapt(Player player, const std::string &public_state,
                      const Eigen::ArrayXd &values, int step, int index) {
  int recover_step = step - 1;
  if (mode_ == Mode::kPostSbCFR) {
    recover_step = step;
  }
  std::string player_state = std::to_string(player) + public_state;
  std::string opponent_state =
      std::to_string(player == 0 ? 1 : 0) + public_state;
  recover_dict_[player_state] = values;
  if (recover_step > 0 &&
      recover_dict_.find(player_state) != recover_dict_.end() &&
      recover_dict_.find(opponent_state) != recover_dict_.end()) {
    std::vector<double> cfr_values{recover_dict_[player_state].sum(),
                                   recover_dict_[opponent_state].sum()};
    // alpha_ = old_adapt(alpha_, cfr_values);
    // if (step % 100 == 0) {
    //   std::cout << alpha_ << ": " << cfr_values << " "
    //             << std::accumulate(cfr_values.begin(), cfr_values.end(), 0.0)
    //             << std::endl;
    // }
  }
}

Eigen::ArrayXd RawSbCFR::_CFR_backward_update_regret(
    double delta, double alpha, const Eigen::ArrayXd &opponent_reach,
    const Eigen::ArrayXXd &values, const Eigen::ArrayXXd &policy, int step) {
  int num_infs = values.rows();
  int num_actions = values.cols();
  Eigen::ArrayXd threshold =
      alpha * opponent_reach * delta * delta * num_actions / step;
  if (weight_type_ == WeightType::kConstant) {
    threshold /= step;
  }
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

Eigen::ArrayXd RawSbCFR::_CFR_Plus_backward_update_regret(
    double delta, double alpha, const Eigen::ArrayXd &opponent_reach,
    const Eigen::ArrayXXd &values, const Eigen::ArrayXXd &policy, int step) {
  // set step here.
  if (weight_type_ == WeightType::kLinear) {
    alpha *= step;
  }
  // we set step to 1 to avoide _CFR_backward_update_regret touching it.
  return _CFR_backward_update_regret(delta, alpha, opponent_reach, values,
                                     policy, 1);
}

Eigen::ArrayXd RawSbCFR::Evaluate(
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

Eigen::ArrayXd RawSbCFR::_dist_cfr_chance(
    Player player, algorithms::PublicNode *node, bool recover,
    const Eigen::ArrayXd &valid_index, const Eigen::ArrayXd &player_reach,
    const Eigen::ArrayXd &average_player_reach,
    const Eigen::ArrayXd &opponent_reach,
    const Eigen::ArrayXd &average_opponent_reach,
    const Eigen::ArrayXd &sampler_reach, const logic::CardSet &outcome,
    Eigen::ArrayXd &predicted_values, const logic::CardSet &cards,
    Trajectory &value_trajectory, Trajectory &policy_trajectory, int step,
    std::mt19937 *rng, int index) {
  logic::CardSet new_outcome = outcome;
  new_outcome.Combine(cards);
  algorithms::PublicNode *new_node = node->GetChild(node->GetChildActions()[0]);
  int check_num = num_hole_cards_ + new_outcome.NumCards();
  Eigen::ArrayXd new_valid_index = valid_index;
  Eigen::ArrayXd new_player_reach = player_reach;
  Eigen::ArrayXd new_opponent_reach = opponent_reach;
  Eigen::ArrayXd new_average_player_reach = average_player_reach;
  Eigen::ArrayXd new_average_opponent_reach = average_opponent_reach;
  for (int inf_id = 0; inf_id != player_outcomes_.size(); ++inf_id) {
    logic::CardSet check_cards = new_outcome;
    check_cards.Combine(player_outcomes_[inf_id]);
    if (check_cards.NumCards() != check_num) {
      new_valid_index(inf_id) = 0;
      new_player_reach(inf_id) = 0;
      new_opponent_reach(inf_id) = 0;
      new_average_player_reach(inf_id) = 0;
      new_average_opponent_reach(inf_id) = 0;
    }
  }
  return _cfr_recursive(player, new_node, recover, new_valid_index,
                        new_player_reach, new_average_player_reach,
                        new_opponent_reach, new_average_opponent_reach,
                        sampler_reach, new_outcome, predicted_values,
                        value_trajectory, policy_trajectory, step, rng, index);
}

Eigen::ArrayXd RawSbCFR::_cfr_recursive(
    Player player, algorithms::PublicNode *node, bool recover,
    const Eigen::ArrayXd &valid_index, const Eigen::ArrayXd &player_reach,
    const Eigen::ArrayXd &average_player_reach,
    const Eigen::ArrayXd &opponent_reach,
    const Eigen::ArrayXd &average_opponent_reach,
    const Eigen::ArrayXd &sampler_reach, const logic::CardSet &outcome,
    Eigen::ArrayXd &predicted_values, Trajectory &value_trajectory,
    Trajectory &policy_trajectory, int step, std::mt19937 *rng, int index) {
  universal_poker::UniversalPokerState *state =
      static_cast<universal_poker::UniversalPokerState *>(node->GetState());
  if (state->IsTerminal()) {
    if (mode_ == Mode::kPostSbCFR || mode_ == Mode::kPSbCFR) {
      Eigen::ArrayXd reach;
      if (average_type_ == AverageType::kOpponent) {
        if (mode_ == Mode::kPSbCFR) {
          reach = (average_opponent_reach * (step - 1) + 2 * opponent_reach) /
                  (step + 1);
        } else {
          reach = (average_opponent_reach * (step - 1) + opponent_reach) / step;
        }
      } else if (average_type_ == AverageType::kLinearOpponent) {
        if (mode_ == Mode::kPSbCFR) {
          reach = (average_opponent_reach * (step - 1) * step +
                   2 * (2 * step + 1) * opponent_reach) /
                  ((step + 1) * (step + 2));
        } else {
          reach = (average_opponent_reach * (step - 1) + 2 * opponent_reach) /
                  (step + 1);
        }
      }
      Eigen::ArrayXd values =
          Evaluate(player, state, valid_index, reach, outcome, index);
      predicted_values =
          Evaluate(player, state, valid_index, opponent_reach / sampler_reach,
                   outcome, index);
      return values;
    } else {
      // SPIEL_CHECK_TRUE((sampler_reach == 1).any());
      Eigen::ArrayXd values =
          Evaluate(player, state, valid_index, opponent_reach / sampler_reach,
                   outcome, index);
      predicted_values = values;
      return values;
    }
  }
  if (state->IsChanceNode()) {
    Eigen::ArrayXd values = Eigen::ArrayXd::Zero(player_outcomes_.size());
    Eigen::ArrayXXd children_pvalues =
        Eigen::ArrayXXd::Zero(player_outcomes_.size(), board_outcomes_.size());
    for (int cidx = 0; cidx != board_outcomes_.size(); ++cidx) {
      compared_cache_[index] = Eigen::ArrayXXd(0, 0);
      auto &cards = board_outcomes_[cidx];
      Eigen::ArrayXd cp(children_pvalues.rows());
      Eigen::ArrayXd new_opponent_reach = opponent_reach * board_proba_;
      Eigen::ArrayXd new_average_opponent_reach =
          average_opponent_reach * board_proba_;
      values += _dist_cfr_chance(
          player, node, recover, valid_index, player_reach,
          average_player_reach, new_opponent_reach, new_average_opponent_reach,
          sampler_reach, outcome, cp, cards, value_trajectory,
          policy_trajectory, step, rng, index);
      children_pvalues.col(cidx) = cp;
    }
    predicted_values = children_pvalues.rowwise().sum() * valid_index;
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
        std::vector<double> cfr_policy =
            policy_eval_[current_player]->Inference(*state).value;
        info_state_copy.SetCumulatePolicy(cfr_policy);
        std::vector<double> cfr_regret =
            value_eval_[current_player]->Inference(*state).value;
        info_state_copy.SetRegret(cfr_regret);
      }
      info_state_copy.ApplyRegretMatchingUsingMax();
      info_state_copy.cumulative_policy =
          PolicyNormalize(info_state_copy.cumulative_policy);
      for (int aidx = 0; aidx != legal_actions.size(); ++aidx) {
        current_policies(inf_id, aidx) = info_state_copy.current_policy[aidx];
        average_policies(inf_id, aidx) =
            info_state_copy.cumulative_policy[aidx];
      }
    } else {
      current_policies.row(inf_id) = 1.0 / legal_actions.size();
      average_policies.row(inf_id) = 1.0 / legal_actions.size();
    }
  }

  Eigen::ArrayXd values = Eigen::ArrayXd::Zero(player_outcomes_.size());
  Eigen::ArrayXXd m(player_outcomes_.size(), legal_actions.size());
  Eigen::ArrayXXd children_values =
      Eigen::ArrayXXd::Zero(player_outcomes_.size(), legal_actions.size());
  for (int aidx = 0; aidx != legal_actions.size(); ++aidx) {
    Eigen::ArrayXd cw = Eigen::ArrayXd::Zero(children_values.rows());
    if (current_player == player) {
      algorithms::PublicNode *next_node = node->GetChild(legal_actions[aidx]);
      Eigen::ArrayXd new_player_reach =
          player_reach * current_policies.col(aidx);
      Eigen::ArrayXd new_average_player_reach =
          average_player_reach * average_policies.col(aidx);
      m.col(aidx) = _cfr_recursive(player, next_node, recover, valid_index,
                                   new_player_reach, new_average_player_reach,
                                   opponent_reach, average_opponent_reach,
                                   sampler_reach, outcome, cw, value_trajectory,
                                   policy_trajectory, step, rng, index);
      values += current_policies.col(aidx) * m.col(aidx);
    } else {
      algorithms::PublicNode *next_node = node->GetChild(legal_actions[aidx]);
      Eigen::ArrayXd new_opponent_reach =
          opponent_reach * current_policies.col(aidx);
      Eigen::ArrayXd new_average_opponent_reach =
          average_opponent_reach * average_policies.col(aidx);
      values +=
          _cfr_recursive(player, next_node, recover, valid_index, player_reach,
                         average_player_reach, new_opponent_reach,
                         new_average_opponent_reach, sampler_reach, outcome, cw,
                         value_trajectory, policy_trajectory, step, rng, index);
    }
    children_values.col(aidx) = cw;
  }

  // udpate weights
  if (current_player != player) {
    predicted_values = children_values.rowwise().sum() * valid_index;
  }

  if (current_player == player) {
    if (mode_ == Mode::kPostSbCFR || mode_ == Mode::kPSbCFR) {
      double delta = game_->MaxUtility() - game_->MinUtility();
      Eigen::ArrayXd sum_opponent_reach =
          root_proba_ *
          (valid_matrix_.rowwise() * average_opponent_reach.transpose())
              .rowwise()
              .sum();
      // std::cout << information_strings[0] << " " << delta << " "
      //           << sum_opponent_reach[0] << std::endl;
      int recover_step = step - 1;
      if (mode_ == Mode::kPostSbCFR) {
        recover_step = step;
      } else if (mode_ == Mode::kPSbCFR) {
        recover_step = step + 1;
      }
      values = _CFR_backward_update_regret(delta, alpha_cache_[index],
                                           sum_opponent_reach, m,
                                           average_policies, recover_step);
      for (int inf_id = 0; inf_id != player_outcomes_.size(); ++inf_id) {
        if (!valid_index(inf_id)) {
          continue;
        }
        auto regret = (m.row(inf_id) - values(inf_id)) * recover_step;
        std::vector<double> regret_vec(regret.begin(), regret.end());
        CFRNetModel::TrainInputs train_input{
            information_strings[inf_id], legal_actions, legal_actions,
            information_tensors[inf_id], regret_vec,    1.0};
        value_eval_[current_player]->SetCFRTabular(train_input);
      }
    } else if (mode_ == Mode::kSbCFRPlus || mode_ == Mode::kPSbCFRPlus) {
      double delta = game_->MaxUtility() - game_->MinUtility();
      Eigen::ArrayXd sum_opponent_reach =
          root_proba_ *
          (valid_matrix_.rowwise() * average_opponent_reach.transpose())
              .rowwise()
              .sum();
      Eigen::ArrayXXd new_cum_regrets;
      if (mode_ == Mode::kSbCFRPlus) {
        Eigen::ArrayXd threshold_t_1 = alpha_cache_[index] *
                                       sum_opponent_reach * delta * delta *
                                       legal_actions.size();
        if (weight_type_ == WeightType::kLinear) {
          threshold_t_1 *= step - 1;
        }
        Eigen::ArrayXXd subs_values =
            m + current_policies.colwise() *
                    (threshold_t_1.sqrt() /
                     (current_policies.rowwise().norm()).array());
        subs_values = subs_values.colwise() * valid_index;
        if (step == 1) {
          subs_values = m;
        }
        values = _CFR_Plus_backward_update_regret(
                     delta, alpha_cache_[index], sum_opponent_reach,
                     subs_values, current_policies, step) *
                 valid_index;
        Eigen::ArrayXd max_values = m.rowwise().maxCoeff();
        values = values.min(max_values);
        new_cum_regrets = (subs_values.colwise() - values).max(0);
      } else if (mode_ == Mode::kPSbCFRPlus) {
        // retrieve old policies.
        Eigen::ArrayXXd old_policies(current_policies.rows(),
                                     current_policies.cols());
        for (int inf_id = 0; inf_id != player_outcomes_.size(); ++inf_id) {
          if (true_regrets_.find(information_strings[inf_id]) ==
              true_regrets_.end()) {
            true_regrets_[information_strings[inf_id]] =
                1.0 / legal_actions.size() *
                Eigen::ArrayXd::Ones(legal_actions.size());
          }
          old_policies.row(inf_id) =
              true_regrets_[information_strings[inf_id]].transpose();
        }
        if (step == 1) {
          values = Eigen::ArrayXd::Zero(player_outcomes_.size());
        } else {
          // compute true policies.
          Eigen::ArrayXd threshold_t_1 = alpha_cache_[index] *
                                         sum_opponent_reach * delta * delta *
                                         legal_actions.size();
          if (weight_type_ == WeightType::kLinear) {
            threshold_t_1 *= step - 2;
          }
          Eigen::ArrayXXd subs_values =
              m + old_policies.colwise() *
                      (threshold_t_1.sqrt() /
                       (old_policies.rowwise().norm()).array());
          subs_values = subs_values.colwise() * valid_index;
          values = _CFR_Plus_backward_update_regret(
                       delta, alpha_cache_[index], sum_opponent_reach,
                       subs_values, old_policies, step - 1) *
                   valid_index;
          Eigen::ArrayXd max_values = m.rowwise().maxCoeff();
          values = values.min(max_values);
          new_cum_regrets = (subs_values.colwise() - values).max(0);
          Eigen::ArrayXXd true_policies =
              1.0 / legal_actions.size() *
              Eigen::ArrayXXd::Ones(player_outcomes_.size(),
                                    legal_actions.size());
          for (int inf_id = 0; inf_id != player_outcomes_.size(); ++inf_id) {
            if (new_cum_regrets.row(inf_id).sum() > 0) {
              true_policies.row(inf_id) = new_cum_regrets.row(inf_id) /
                                          new_cum_regrets.row(inf_id).sum();
            }
          }
          // store the true policies
          for (int inf_id = 0; inf_id != player_outcomes_.size(); ++inf_id) {
            true_regrets_[information_strings[inf_id]] =
                true_policies.row(inf_id).transpose();
          }
        }
        Eigen::ArrayXXd true_policies(current_policies.rows(),
                                      current_policies.cols());
        for (int inf_id = 0; inf_id != player_outcomes_.size(); ++inf_id) {
          if (true_regrets_.find(information_strings[inf_id]) ==
              true_regrets_.end()) {
            true_regrets_[information_strings[inf_id]] =
                1.0 / legal_actions.size() *
                Eigen::ArrayXd::Ones(legal_actions.size());
          }
          true_policies.row(inf_id) =
              true_regrets_[information_strings[inf_id]].transpose();
        }
        // compute the predicted regrets and predicted policeis.
        Eigen::ArrayXd threshold_curr = alpha_cache_[index] *
                                        sum_opponent_reach * delta * delta *
                                        legal_actions.size();
        if (weight_type_ == WeightType::kLinear) {
          threshold_curr *= step - 1;
        }
        Eigen::ArrayXXd subs_predicted_values =
            children_values + true_policies.colwise() *
                                  (threshold_curr.sqrt() /
                                   (true_policies.rowwise().norm()).array());
        if (step == 1) {
          subs_predicted_values = children_values;
        }
        subs_predicted_values = subs_predicted_values.colwise() * valid_index;
        predicted_values = _CFR_Plus_backward_update_regret(
                               delta, alpha_cache_[index], sum_opponent_reach,
                               subs_predicted_values, true_policies, step) *
                           valid_index;
        Eigen::ArrayXd max_predicted_values =
            children_values.rowwise().maxCoeff();
        predicted_values = values.min(max_predicted_values);
        new_cum_regrets =
            (subs_predicted_values.colwise() - predicted_values).max(0);
      }

      for (int inf_id = 0; inf_id != player_outcomes_.size(); ++inf_id) {
        if (!valid_index(inf_id)) {
          continue;
        }
        std::vector<double> regret_vec(legal_actions.size());
        for (int aidx = 0; aidx != legal_actions.size(); ++aidx) {
          regret_vec[aidx] = new_cum_regrets(inf_id, aidx);
        }
        CFRNetModel::TrainInputs train_input{
            information_strings[inf_id], legal_actions, legal_actions,
            information_tensors[inf_id], regret_vec,    1.0};
        value_eval_[current_player]->SetCFRTabular(train_input);
      }
    } else {
      for (int inf_id = 0; inf_id != player_outcomes_.size(); ++inf_id) {
        predicted_values(inf_id) = 0;
        if (!valid_index(inf_id)) {
          continue;
        }
        Eigen::ArrayXd regret = m.row(inf_id) - values(inf_id);

        if (mode_ == Mode::kPCFR || mode_ == Mode::kPCFRPlus) {
          Eigen::ArrayXd predicted_regret =
              children_values.row(inf_id) -
              (children_values.row(inf_id) * current_policies.row(inf_id))
                  .sum();
          // get the true old cumulative regrets.
          if (true_regrets_.find(information_strings[inf_id]) ==
              true_regrets_.end()) {
            true_regrets_[information_strings[inf_id]] =
                Eigen::ArrayXd::Zero(legal_actions.size());
          }
          Eigen::ArrayXd old_cum_regrets =
              true_regrets_[information_strings[inf_id]];
          // new predicted regret is only for computing the strategy for the
          // next iteration.
          Eigen::ArrayXd new_cum_regrets(legal_actions.size());
          Eigen::ArrayXd new_predicted_cum_regrets(legal_actions.size());
          if (mode_ == Mode::kPCFR) {
            new_cum_regrets = old_cum_regrets + regret;
            new_predicted_cum_regrets =
                old_cum_regrets + regret + predicted_regret;
          } else if (mode_ == Mode::kPCFRPlus) {
            new_cum_regrets = (old_cum_regrets + regret).max(0);
            new_predicted_cum_regrets =
                ((old_cum_regrets + regret).max(0) + predicted_regret).max(0);
          }
          Eigen::ArrayXd new_policies =
              1.0 / legal_actions.size() *
              Eigen::ArrayXd::Ones(legal_actions.size());
          if (new_predicted_cum_regrets.max(0).sum() > 0) {
            new_policies = new_predicted_cum_regrets.max(0) /
                           new_predicted_cum_regrets.max(0).sum();
          }
          predicted_values(inf_id) =
              (children_values.row(inf_id).transpose() * new_policies).sum();

          // save the true regret
          true_regrets_[information_strings[inf_id]] = new_cum_regrets;

          // Set regrets for computing the next iteration;
          std::vector<double> regret_vec(legal_actions.size());
          for (int aidx = 0; aidx != legal_actions.size(); ++aidx) {
            regret_vec[aidx] = new_predicted_cum_regrets(aidx);
          }
          // SPIEL_CHECK_DOUBLE_EQ(predicted_values(inf_id), ep_value);
          // SPIEL_CHECK_DOUBLE_EQ(values(inf_id), ep_value);
          CFRNetModel::TrainInputs train_input{
              information_strings[inf_id], legal_actions, legal_actions,
              information_tensors[inf_id], regret_vec,    1.0};
          value_eval_[current_player]->SetCFRTabular(train_input);
        } else if (mode_ == Mode::kCFRPlus) {
          std::vector<double> regret_vec(legal_actions.size());
          std::vector<double> old_regret =
              value_eval_[current_player]->GetCFRTabular(
                  CFRNetModel::InferenceInputs{information_strings[inf_id],
                                               legal_actions});
          for (int aidx = 0; aidx != legal_actions.size(); ++aidx) {
            regret_vec[aidx] = std::max(old_regret[aidx] + regret(aidx), 0.0);
          }
          CFRNetModel::TrainInputs train_input{
              information_strings[inf_id], legal_actions, legal_actions,
              information_tensors[inf_id], regret_vec,    1.0};
          value_eval_[current_player]->SetCFRTabular(train_input);
        } else if (mode_ == Mode::kCFR || mode_ == Mode::kLCFR) {
          std::vector<double> regret_vec(legal_actions.size());
          for (int aidx = 0; aidx != legal_actions.size(); ++aidx) {
            regret_vec[aidx] = regret(aidx);
            if (mode_ == Mode::kLCFR) {
              regret_vec[aidx] = regret(aidx) * step;
            }
          }
          CFRNetModel::TrainInputs train_input{
              information_strings[inf_id], legal_actions, legal_actions,
              information_tensors[inf_id], regret_vec,    1.0};
          value_eval_[current_player]->AccumulateCFRTabular(train_input);
        }
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
}  // namespace algorithms

}  // namespace algorithms
}  // namespace open_spiel