#ifndef DEEPCFR_SBCFR_SOLVER_H_
#define DEEPCFR_SBCFR_SOLVER_H_

#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "absl/algorithm/container.h"
#include "deep_escfr_solver.h"
#include "open_spiel/algorithms/public_tree.h"
#include "open_spiel/games/universal_poker.h"
#include "open_spiel/games/universal_poker/acpc_cpp/acpc_game.h"
#include "open_spiel/games/universal_poker/logic/card_set.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/thread_pool.h"
#include "vpnet.h"

namespace logic = open_spiel::universal_poker::logic;

namespace open_spiel {
namespace algorithms {

class OnlineLearningOptimization {
  // Only for universal poker with 2 players, and 2 rounds.
 public:
  enum class Mode {
    kCFR,
    kCFRPLUS,
    kFTRL,
    kFTRLCFR,
    kOMD,
    kOMDCFR,
  };
  enum class AverageType {
    kCurrent,
    kOpponent,
    kLinearOpponent,
  };

  enum class UpdateType {
    kSimultaneous,
    kAlternating,
  };

  static Mode GetMode(const std::string &mode_str) {
    if (mode_str == "CFR") {
      return Mode::kCFR;
    } else if (mode_str == "CFRPLUS") {
      return Mode::kCFRPLUS;
    } else if (mode_str == "FTRL") {
      return Mode::kFTRL;
    } else if (mode_str == "FTRLCFR") {
      return Mode::kFTRLCFR;
    } else if (mode_str == "OMD") {
      return Mode::kOMD;
    } else if (mode_str == "OMDCFR") {
      return Mode::kOMDCFR;
    } else {
      SpielFatalError("cfr mode error!");
      return Mode::kCFR;
    }
  }

  static AverageType GetAverageType(const std::string &type_str) {
    if (type_str == "Current") {
      return AverageType::kCurrent;
    } else if (type_str == "Opponent") {
      return AverageType::kOpponent;
    } else if (type_str == "LinearOpponent") {
      return AverageType::kLinearOpponent;
    } else {
      SpielFatalError("average type error!");
      return AverageType::kCurrent;
    }
  }

  OnlineLearningOptimization(const Game &game,
                             std::shared_ptr<VPNetEvaluator> value_0_eval,
                             std::shared_ptr<VPNetEvaluator> value_1_eval,
                             std::shared_ptr<VPNetEvaluator> policy_0_eval,
                             std::shared_ptr<VPNetEvaluator> policy_1_eval,
                             bool use_regret_net, bool use_policy_net,
                             bool use_tabular, std::mt19937 *rng,
                             int cfr_batch_size, double alpha,
                             Mode mode = Mode::kCFR,
                             AverageType type = AverageType::kOpponent);

  ~OnlineLearningOptimization() = default;

  std::pair<Trajectory, Trajectory> RunIteration(Player player, int step);

  std::pair<Trajectory, Trajectory> RunIteration(std::mt19937 *rng,
                                                 Player player, int step);

  int NodeTouched() { return node_touch_; }

  double Alpha() { return alpha_; }

 private:
  Eigen::ArrayXd _FTRLProx(const Eigen::ArrayXd &weights,
                           const Eigen::ArrayXXd &values,
                           const Eigen::ArrayXXd &policy, int step);
  Eigen::ArrayXd _OMDProx(const Eigen::ArrayXd &weights,
                          const Eigen::ArrayXXd &values,
                          const Eigen::ArrayXXd &policy, int step);

  Eigen::ArrayXd _OMDCFRProx(const Eigen::ArrayXd &weights,
                             const Eigen::ArrayXd &old_weights,
                             const Eigen::ArrayXXd &values,
                             const Eigen::ArrayXXd &policy, int step);

  Eigen::ArrayXd Evaluate(Player player,
                          const universal_poker::UniversalPokerState *state,
                          const Eigen::ArrayXd &p, const Eigen::ArrayXd &q,
                          const logic::CardSet &outcome, int index);

  Eigen::ArrayXd _dist_cfr_chance(
      Player player, algorithms::PublicNode *node,
      const Eigen::ArrayXd &valid_index, const Eigen::ArrayXd &player_reach,
      const Eigen::ArrayXd &average_player_reach,
      const Eigen::ArrayXd &opponent_reach,
      const Eigen::ArrayXd &average_opponent_reach,
      const Eigen::ArrayXd &sampler_reach, const logic::CardSet &outcome,
      Eigen::ArrayXd &weights, const logic::CardSet &cards,
      Trajectory &value_trajectory, Trajectory &policy_trajectory, int step,
      std::mt19937 *rng, int index);

  Eigen::ArrayXd _enter_cfr(
      Player player, algorithms::PublicNode *node,
      const Eigen::ArrayXd &valid_index, const Eigen::ArrayXd &player_reach,
      const Eigen::ArrayXd &average_player_reach,
      const Eigen::ArrayXd &opponent_reach,
      const Eigen::ArrayXd &average_opponent_reach,
      const Eigen::ArrayXd &sampler_reach, const logic::CardSet &outcome,
      Eigen::ArrayXd &weights, Trajectory &value_trajectory,
      Trajectory &policy_trajectory, int step, std::mt19937 *rng, int index);

  // SbCFR:
  // 1. Recovery, opponent_reach is the reach of average strategy. The recoverd
  // cumulative regret is temporarily saved in info_regets_.
  // 2. One step CFR, opponent_reach is the reach of the current strategy. The
  // current
  // strategy is computed by regret match using the regret in value_eval.
  // The
  // value_eval could be a tabular or a neural network.
  // 3. Update regret and strategy. The regret is the sum of the regret
  // saved in
  // info_regrets_ and the new regret return by CFR.
  Eigen::ArrayXd _cfr_recursive(
      Player player, algorithms::PublicNode *node,
      const Eigen::ArrayXd &valid_index, const Eigen::ArrayXd &player_reach,
      const Eigen::ArrayXd &average_player_reach,
      const Eigen::ArrayXd &opponent_reach,
      const Eigen::ArrayXd &average_opponent_reach,
      const Eigen::ArrayXd &sampler_reach, const logic::CardSet &outcome,
      Eigen::ArrayXd &weights, Trajectory &value_trajectory,
      Trajectory &policy_trajectory, int step, std::mt19937 *rng, int index);

 private:
  int step_;
  const universal_poker::UniversalPokerGame *game_;
  const universal_poker::acpc_cpp::ACPCGame *acpc_game_;
  std::mt19937 *rng_;
  int cfr_batch_size_;
  uint32_t iterations_;
  std::uniform_real_distribution<double> dist_;
  std::vector<std::shared_ptr<VPNetEvaluator>> value_eval_;
  std::vector<std::shared_ptr<VPNetEvaluator>> policy_eval_;
  std::unordered_map<std::string, std::vector<double>> info_regrets_;
  std::vector<PublicTree> trees_;
  Mode mode_;
  AverageType average_type_;
  double alpha_;
  double scale_up_coeff_;
  double scale_down_coeff_;
  double scale_ub_;
  double scale_lb_;
  double alpha_ub_;
  double alpha_lb_;
  logic::CardSet deck_;
  std::vector<logic::CardSet> player_outcomes_;
  std::vector<std::vector<uint8_t>> player_outcome_arrays_;
  std::vector<logic::CardSet> board_outcomes_;
  std::vector<std::vector<uint8_t>> board_outcome_arrays_;
  Eigen::ArrayXXd valid_matrix_;
  std::vector<Eigen::ArrayXXd> compared_cache_;
  int num_outcomes_;
  int num_hole_cards_;
  int num_board_cards_;
  double root_proba_;
  double board_proba_;
  logic::CardSet default_cards_;

  bool use_regret_net;
  bool use_policy_net;
  bool use_tabular;
  int node_touch_;
};
}  // namespace algorithms
}  // namespace open_spiel

#endif  // DEEPCFR_SBCFR_SOLVER_H_