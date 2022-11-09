#include "deep_cfr.h"

// #include <gperftools/profiler.h>

#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "deep_escfr_solver.h"
#include "deep_rscfr_solver.h"
#include "device_manager.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/games/universal_poker.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/barrier.h"
#include "open_spiel/utils/circular_buffer.h"
#include "open_spiel/utils/data_logger.h"
#include "open_spiel/utils/file.h"
#include "open_spiel/utils/json.h"
#include "open_spiel/utils/logger.h"
#include "open_spiel/utils/lru_cache.h"
#include "open_spiel/utils/reservior_buffer.h"
#include "open_spiel/utils/stats.h"
#include "open_spiel/utils/thread.h"
#include "open_spiel/utils/threaded_queue.h"
#include "raw_sbcfr_solver.h"
// #include "sbcfr1_solver.h"
#include "universal_poker_exploitability.h"
#include "vpevaluator.h"
#include "vpnet.h"

namespace open_spiel {
namespace algorithms {

void actor(const Game& game, const DeepCFRConfig& config,
           DeviceManager* device_manager,
           std::shared_ptr<VPNetEvaluator> value_0_eval,
           std::shared_ptr<VPNetEvaluator> value_1_eval,
           std::shared_ptr<VPNetEvaluator> policy_0_eval,
           std::shared_ptr<VPNetEvaluator> policy_1_eval) {
  FileLogger logger(config.path, absl::StrCat("evaluator-", 0, "-mpi-", 0));
  DataLoggerJsonLines data_logger(
      config.path, absl::StrCat("evaluator-", 0, "-mpi-", 0), true);
  logger.Print("Running the evaluator on device %d: %s", 0,
               device_manager->Get(0, 0)->Device());
  CFRNetModel* model = device_manager->Get(0, 0).model();
  std::random_device rd;
  std::mt19937 rng(rd());
  omp_set_num_threads(config.omp_threads);
  CFRPolicy policy(model);
  std::unordered_map<std::string, std::shared_ptr<State>> infosets;
  std::unique_ptr<PublicTree> tree;
  std::unordered_set<std::string> infos;
  if (config.game == "leduc_poker") {
    infos = {
        "Qh:Qs:crc,r",   "As:Qh:crrc,r",  "Kh:Ah:rc,r",   "Qh:Qs:crc,cr",
        "Kh:Qh:crc,c",   "Qs:As:rrc,cr",  "Ks:Qh:cc,cr",  "As:Qs:crrc",
        "Qh:Kh:crrc,cr", "Qh::crr",       "Qs:Ks:rc,crr", "Ah:Kh:crrc,rr",
        "Ks:As:crc,crr", "Ks:Qh:rrc,crr", "Kh:Ks:cc,rr",  "Qs:Kh:cc,r",
        "As::",          "Ks:As:crc,rr",  "Kh:Qs:cc,crr", "As:Qs:rc,cr",
        "Ks::r",         "As:Ah:crrc,c",  "Qh::rr",       "Ah:Ks:rrc,r",
    };
  } else if (config.game.find("FHP") != std::string::npos) {
    infos = {
        "QhAs:JsKhKs:crrc,crrrr",
        "KsAh:ThJhJs:crc,crr",
        "QhAh:TsJsQs:crc,rrrr",
        "QsKh:ThKsAh:crrrc,rrrr",
        "TsKs:ThAhAs:rc,crrrr",
        "QsKh:TsJsAs:crrc,cr",
        "JhKh:KsAhAs:crc,crrrr",
        "ThKs::cr",
        "ThAs:TsJsQs:rc",
        "TsAs:ThJhKh:rrc,rr",
        "ThKh:JsQsAh:cc,r",
        "JhJs:ThKsAh:crrc,rr",
        "TsAh:QhQsKs:rrc,r",
        "QhKh::crrr",
        "QsAh:JsQhKs:rc,rrr",
        "KsAh:ThKhAs:cc,rr",
        "QhKh::crr",
        "KsAs:TsQhKh:cc,crrrr",
        "JhQh:ThTsAs:rrrc,rrr",
        "QsAh::rrr",
        "JsAs:QhQsKh:crrrc,crrrr",
        "QhKs:ThQsAs:rc,r",
        "TsAh:JhJsKs:crrrc,c",
        "JhAh:QsKhAs:crc,rrr",
        "JsKh:QhAhAs:rc,crr",
        "KhAs:JhJsKs:rrrc,rrrr",
        "QhAs::c",
        "TsJh:QhQsAs:rrrc,r",
        "TsQs:ThJsKh:crrrc,crr",
        "AhAs:JhQhKh:cc,rrr",
        "QsAh:JsQhAs:crrrc,rr",
        "JhKh:ThJsQs:crrc,rrr",
        "TsKs:JhJsQh:rrrc,c",
        "ThKs:TsQhAh:rrrc,cr",
        "QsAs:JhQhAh:cc,rrrr",
        "ThQs:QhKhKs:crrrc",
        "TsQs:ThJhAh:rrc,crrr",
        "QsKh:JhJsKs:rc,rrrr",
        "AhAs:TsJsKh:crrc",
        "JsAh:TsJhKs:rrrc",
        "AhAs:JhQhQs:rc,crrr",
        "AhAs:QhKhKs:cc",
        "ThKs:TsKhAh:rrrc,crrrr",
        "KsAh::",
        "TsKs:JsKhAh:crc,r",
        "QhQs:TsJhJs:cc,c",
        "QhAs:ThJhKs:crrc,c",
        "QhAs:JsQsAh:crc,crrr",
        "ThQs::r",
        "ThKs:JhQhKh:rc,cr",
    };
  }
  if (config.use_tabular) {
    tree.reset(new PublicTree(game.NewInitialState()));
    std::vector<std::string> histories = tree->GetDecisionHistoriesl();
    std::vector<std::string> infosets_str;
    for (auto& his : histories) {
      infosets_str.push_back(his);
    }
    for (auto& key : infosets_str) {
      PublicNode* node = tree->GetByHistory(key);
      for (int i = 0; i != 1000; ++i) {
        std::shared_ptr<State> state = node->GetState()->Clone();
        state->SampleChance(&rng);
        std::string info_str = state->InformationStateString();
        if (infos.find(info_str) != infos.end() || infos.empty()) {
          infosets.insert({info_str, state});
        }
      }
    }
  }
  int64_t node_touched = 0;
  double alpha = 0;
  universal_poker::UniversalPokerExploitability expSolver(game, policy, model,
                                                          config.omp_threads);

  RawSbCFR::Mode mode = RawSbCFR::GetMode(config.cfr_mode);
  RawSbCFR::AverageType a_type = RawSbCFR::GetAverageType(config.average_type);
  RawSbCFR::WeightType w_type = RawSbCFR::GetWeightType(config.weight_type);
  RawSbCFR solver(game, value_0_eval, value_1_eval, policy_0_eval,
                  policy_1_eval, config.use_regret_net, config.use_policy_net,
                  config.use_tabular, &rng, 1, config.cfr_rm_scale, mode,
                  a_type, w_type);
  for (int step = 1; config.max_steps == 0 || step <= config.max_steps;
       ++step) {
    for (Player player = 0; player < 2; ++player) {
      solver.RunIteration(player, step);
      node_touched += solver.NodeTouched();
      alpha = solver.Alpha();
    }
    if (step % config.evaluation_window != 0 &&
        step != config.first_evaluation) {
      continue;
    }
    double exp = expSolver();
    logger.Print("Exploitability: %.3f", exp);
    std::cout << absl::StrFormat("Exploitability: %.3f", exp) << std::endl;
    DataLogger::Record record = {
        {"Step", step},
        {"Exploitability", exp},
        {"Touch", node_touched},
        {"Alpha", alpha},
    };
    if (config.use_tabular) {
      DataLogger::Record info_record;
      for (auto& infoset : infosets) {
        std::string info = infoset.first;
        std::shared_ptr<State> state = infoset.second;
        CFRNetModel::InferenceInputs inf_input{state->InformationStateString(),
                                               state->LegalActions(),
                                               state->InformationStateTensor()};
        std::vector<double> info_value =
            model->InfValue(state->CurrentPlayer(), {inf_input})[0].value;
        std::vector<double> info_policy =
            model->InfPolicy(state->CurrentPlayer(), {inf_input})[0].value;
        std::vector<double> tabular_value =
            model->GetCFRTabularValue({inf_input});
        for (auto& tv : tabular_value) {
          tv = tv / sqrt(step);
        };
        for (auto& tv : info_value) {
          tv = tv / sqrt(step);
        };
        std::vector<double> tabular_policy =
            model->GetCFRTabularPolicy({inf_input});
        std::vector<Action> actions = state->LegalActions();
        std::vector<std::string> str_actions(actions.size());
        absl::c_transform(
            actions, str_actions.begin(),
            [state](const Action& a) { return state->ActionToString(a); });
        info_record.emplace(
            info, json::Object({
                      {"action", json::CastToArray(str_actions)},
                      {"regret_net", json::CastToArray(info_value)},
                      {"regret_true", json::CastToArray(tabular_value)},
                      {"policy_net", json::CastToArray(info_policy)},
                      {"policy_true", json::CastToArray(tabular_policy)},
                  }));
      }
      record.insert({"info_set", info_record});
    }
    data_logger.Write(record);
  }
}

bool deep_cfr(DeepCFRConfig config, StopToken* stop) {
  std::shared_ptr<const open_spiel::Game> game;
  std::unordered_map<std::string, GameParameters> game_parameters = {
      {"kuhn_poker", universal_poker::KuhnPokerParameters()},
      {"leduc_poker", universal_poker::LeducPokerParameters()},
      {"leduc18_poker", universal_poker::Leduc18PokerParameters()},
      {"leduc5_poker", universal_poker::NolimitedLeduc5PokerParameters()},
      {"leduc10_poker", universal_poker::NolimitedLeduc10PokerParameters()},
      {"FHP_poker", universal_poker::FHPPokerParameters()},
      {"FHP2_poker", universal_poker::FHP2PokerParameters()},
      {"FHP3_poker", universal_poker::FHP3PokerParameters()},
      {"HULH_poker", universal_poker::HULHPokerParameters()}};
  game = LoadGame("universal_poker", game_parameters[config.game]);
  std::cout << game->InformationStateTensorShape() << std::endl;
  open_spiel::GameType game_type = game->GetType();
  if (game->NumPlayers() != 2)
    open_spiel::SpielFatalError("AlphaZero can only handle 2-player games.");
  if (game_type.reward_model != open_spiel::GameType::RewardModel::kTerminal)
    open_spiel::SpielFatalError("Game must have terminal rewards.");
  if (game_type.dynamics != open_spiel::GameType::Dynamics::kSequential)
    open_spiel::SpielFatalError("Game must have sequential turns.");

  file::Mkdir(config.path);
  if (!file::IsDirectory(config.path)) {
    std::cerr << config.path << " is not a directory." << std::endl;
    return false;
  }
  std::cout << "Logging directory: " << config.path << std::endl;
  std::cout << "Playing game: " << config.game << std::endl;

  {
    file::File fd(config.path + "/config.json", "w");
    fd.Write(json::ToString(config.ToJson(), true) + "\n");
  }

  DeviceManager device_manager;
  device_manager.AddDevice(CFRNetModel(
      *game, config.path, "", config.graph_def, config.omp_threads, "/cpu:0",
      config.use_regret_net, true, config.use_policy_net, true));

  auto value_0_eval = std::make_shared<VPNetEvaluator>(
      &device_manager, true, Player{0}, false, false,
      config.inference_batch_size, config.inference_threads,
      config.inference_cache, (config.actors + config.evaluators) / 16);

  auto value_1_eval = std::make_shared<VPNetEvaluator>(
      &device_manager, true, Player{1}, false, false,
      config.inference_batch_size, config.inference_threads,
      config.inference_cache, (config.actors + config.evaluators) / 16);

  auto policy_0_eval = std::make_shared<VPNetEvaluator>(
      &device_manager, false, Player{0}, false, false,
      config.inference_batch_size, config.inference_threads,
      config.inference_cache, (config.actors + config.evaluators) / 16);

  auto policy_1_eval = std::make_shared<VPNetEvaluator>(
      &device_manager, false, Player{1}, false, false,
      config.inference_batch_size, config.inference_threads,
      config.inference_cache, (config.actors + config.evaluators) / 16);

  actor(*game, config, &device_manager, value_0_eval, value_1_eval,
        policy_0_eval, policy_1_eval);

  return true;
}

}  // namespace algorithms
}  // namespace open_spiel