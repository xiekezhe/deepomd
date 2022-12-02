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
#include "deep_sbescfr_solver.h"
#include "device_manager.h"
#include "local_best_response.h"
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
#include "play.h"
#include "universal_poker_exploitability.h"
#include "vpevaluator.h"
#include "vpnet.h"

namespace open_spiel {
namespace algorithms {

// protected by LearningMutex
std::mutex LearningMutex;
std::condition_variable LearningCondition;
std::condition_variable NotLearningCondition;
bool Learning = false;
double Alpha = 0.0;
// protected by LearningMutex

// protected by EvaluatingMutex
std::mutex EvaluatingMutex;
std::condition_variable EvaluatingCondition;
std::condition_variable NotEvaluatingCondition;
int EvalSteps = 1;
int64_t EvalSampledStates = 0;
int64_t EvalTrainedStates = 0;
int64_t EvalNodeTouched = 0;
bool Evaluating = false;  // evaluate at the start.
// protected by EvaluatingMutex

void actor(const Game& game, const DeepCFRConfig& config, int num,
           std::vector<ThreadedQueue<Trajectory>*> value_trajectory_queues,
           std::vector<ThreadedQueue<Trajectory>*> policy_trajectory_queues,
           std::vector<ThreadedQueue<Trajectory>*> history_trajectory_queues,
           std::vector<std::shared_ptr<VPNetEvaluator>> value_evals,
           std::vector<std::shared_ptr<VPNetEvaluator>> policy_z_eval,//new modified
           std::vector<std::shared_ptr<VPNetEvaluator>> policy_evals,
           std::vector<std::shared_ptr<VPNetEvaluator>> current_policy_evals,
           Barrier* actor_barrier, StopToken* stop) {
  std::unique_ptr<Logger> logger;
  if (num < 2) {  // Limit the number of open files.
    logger.reset(
        new FileLogger(config.path, absl::StrCat("actor-", num, "-mpi-", 0)));
  } else {
    logger.reset(new NoopLogger());
  }
  // NOTE: it is critical to set different seed for different actor.
  std::random_device rd;
  std::mt19937 rng(rd());
  omp_set_num_threads(config.omp_threads);
  DeepSbESCFRSolver solver(game, config, value_evals, policy_z_eval,//new modified
                           policy_evals, current_policy_evals, &rng);
  int cfr_batch_per_actor = config.cfr_batch_size / config.actors;
  int cfr_batch_per_actor_rest =
      config.cfr_batch_size - cfr_batch_per_actor * config.actors;
  SPIEL_CHECK_LT(cfr_batch_per_actor_rest, config.actors);
  if (num < cfr_batch_per_actor_rest) {
    cfr_batch_per_actor += 1;
  }
  for (int step = 1; !stop->StopRequested() &&
                     (config.max_steps == 0 || step <= config.max_steps);
       ++step) {
    //{
      // wait until not learning.
    //  std::unique_lock<std::mutex> lock(LearningMutex);
    //  NotLearningCondition.wait(
     //     lock, [stop]() { return (!Learning) || stop->StopRequested(); });
    //}
    if (config.verbose) {
      logger->Print("starting running");
    }
    for (Player player = 0; player < 2; ++player) {
      {
      // wait until not learning.
      std::unique_lock<std::mutex> lock(LearningMutex);
      NotLearningCondition.wait(
          lock, [stop]() { return (!Learning) || stop->StopRequested(); });
    }
      auto value_trajectory_queue = value_trajectory_queues[player];
      auto policy_trajectory_queue = policy_trajectory_queues[player];
      auto history_trajectory_queue = history_trajectory_queues[player];
      int before_queue_size = value_trajectory_queue->Size();
      int before_policy_queue_size = policy_trajectory_queue->Size();
      for (int game_num = 1;
           !stop->StopRequested() && (game_num <= cfr_batch_per_actor);
           ++game_num) {
        auto trajectories =
            solver.RunIteration(player, Alpha, step, cfr_batch_per_actor);
        if (!value_trajectory_queue->Push(trajectories[0],
                                          std::chrono::seconds(10))) {
          logger->Print("Failed to push a trajectory after 10 seconds.");
        }
        if (!policy_trajectory_queue->Push(trajectories[1],
                                           std::chrono::seconds(10))) {
          logger->Print("Failed to push a trajectory after 10 seconds.");
        }
        if (!history_trajectory_queue->Push(trajectories[2],
                                          std::chrono::seconds(10))) {
          logger->Print("Failed to push a trajectory after 10 seconds.");
        }
      }
      if (config.verbose) {
        logger->Print(
            "Step %d\npush %d trajectories to queue %d, queue size: %d -> %d, "
            "policy size: %d -> %d.",
            step, cfr_batch_per_actor, player, before_queue_size,
            value_trajectory_queue->Size(), before_policy_queue_size,
            policy_trajectory_queue->Size());
      }
    //}
    // NOTE: Do not break here, otherwise some threads may be blocked here.
    // The last thread calls the Block is responsible to set the Learning
    // flag before the other threads existing the block.
    if (actor_barrier->Block([&]() {
          std::unique_lock<std::mutex> lock(LearningMutex);
          Learning = true;
          LearningCondition.notify_all();
        })) {
    }
    }
  }
  logger->Print("Got a quit.");
}

void evaluator(const open_spiel::Game& game, DeepCFRConfig& config, int num,
               DeviceManager* device_manager, int device_id, StopToken* stop) {
  FileLogger logger(config.path, absl::StrCat("evaluator-", num, "-mpi-", 0));
  DataLoggerJsonLines data_logger(
      config.path, absl::StrCat("evaluator-", num, "-mpi-", 0), true);
  logger.Print("Running the evaluator on device %d: %s", device_id,
               device_manager->Get(0, device_id)->Device());
  CFRNetModel* model = device_manager->Get(0, device_id).model();
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
    // randomly sample 24 infosets for plotting.
    // if (infosets_str.size() > 24) {
    //   std::shuffle(infosets_str.begin(), infosets_str.end(), rng);
    //   infosets_str.resize(24);
    // }
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
  // std::cout << "{";
  // for (auto& info : infosets) {
  //   std::cout << "\"" << info.first << "\"," << std::endl;
  // }
  // std::cout << "}" << std::endl;
  // return;
  int eval_steps = 1;
  int64_t sampled_states = 0;
  int64_t trained_states = 0;
  universal_poker::UniversalPokerExploitability expSolver(game, policy, model,
                                                          config.omp_threads);
  universal_poker::LocalBestResponse lbrSolver(game, policy, model,
                                               config.omp_threads);
  logger.Print("game: %s", config.game);
  while (!stop->StopRequested()) {
    {
      // wait for learner to ask for Evaluating.
      std::unique_lock<std::mutex> lock(EvaluatingMutex);
      EvaluatingCondition.wait(
          lock, [stop]() { return Evaluating || stop->StopRequested(); });
      if (stop->StopRequested()) {
        break;
      }
      eval_steps = EvalSteps;
      sampled_states = EvalSampledStates;
      trained_states = EvalTrainedStates;
    }
    if (device_id != 0) {
      for (Player player = 0; player < 2; ++player) {
        model->SyncPolicyFrom(player, *(device_manager->Get(0, 0).model()));
        model->SyncCurrentPolicyFrom(player,
                                     *(device_manager->Get(0, 0).model()));
        model->SyncValueFrom(player, *(device_manager->Get(0, 0).model()));
        model->SyncGlobalValueFrom(player,
                                   *(device_manager->Get(0, 0).model()));
      }
    }
    // exp
    auto start = std::chrono::system_clock::now();
    double exp = 0;
    std::pair<double, double> lbr;
    if (!config.local_best_response) {
      exp = expSolver();
      lbr = {exp, 0};
      if (true) {
        auto end = std::chrono::system_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double total_sec = double(duration.count()) *
                           std::chrono::microseconds::period::num /
                           std::chrono::microseconds::period::den;
        std::cout << "exp used " << total_sec << "s " << std::endl;
      }
      std::cout << absl::StrFormat("Exploitability: %.3f", exp) << std::endl;
    } else {
      // lbr
      start = std::chrono::system_clock::now();
      lbr = lbrSolver(config.lbr_batch_size, config.verbose);
      exp = lbr.first;
      if (true) {
        auto end = std::chrono::system_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double total_sec = double(duration.count()) *
                           std::chrono::microseconds::period::num /
                           std::chrono::microseconds::period::den;
        std::cout << "lbr used " << total_sec << "s " << std::endl;
      }
      std::cout << absl::StrFormat("LocalBestResponse: %.3f, %.3f", lbr.first,
                                   2 * lbr.second)
                << std::endl;
    }

    {
      std::unique_lock<std::mutex> lock(EvaluatingMutex);
      Evaluating = false;
      NotEvaluatingCondition.notify_all();
    }
    logger.Print("Exploitability: %.3f", exp);
    logger.Print("LocalBestResponse: %.3f, %.3f", lbr.first, 2 * lbr.second);
    DataLogger::Record record = {
        {"Step", eval_steps},
        {"Sampled states", sampled_states},
        {"Trained states", trained_states},
        {"Exploitability", exp},
        {"LocalBestResponse", lbr.first},
        {"LocalBestResponse_std", lbr.second},
        {"Alpha", Alpha},
        {"Touch", EvalNodeTouched},
    };
    if (config.use_tabular) {
      DataLogger::Record info_record;
      for (auto& infoset : infosets) {
        std::string info = infoset.first;
        std::shared_ptr<State> state = infoset.second;
        std::vector<Action> actions = state->LegalActions();
        universal_poker::UniversalPokerState* poker_state =
            static_cast<universal_poker::UniversalPokerState*>(state.get());
        CFRNetModel::InferenceInputs inf_input{state->InformationStateString(),
                                               state->LegalActions(),
                                               state->InformationStateTensor()};
        std::vector<double> info_value =
            model->InfValue(state->CurrentPlayer(), {inf_input})[0].value;
        std::vector<double> info_policy =
            model->InfPolicy(state->CurrentPlayer(), {inf_input})[0].value;
        std::vector<double> info_curr_policy =
            model->InfCurrentPolicy(state->CurrentPlayer(), {inf_input})[0]
                .value;
        std::vector<double> tab_policy =
            model->GetCFRTabularPolicy({inf_input});
        std::vector<double> tab_curr_policy =
            model->GetCFRTabularCurrentPolicy({inf_input});
        CFRNetModel::InferenceInputs global_input{state->ObservationString(),
                                                  state->LegalActions(),
                                                  state->ObservationTensor()};
        std::vector<double> global_value =
            model->InfGlobalValue(state->CurrentPlayer(), {global_input})[0]
                .value;
        // for (auto& tv : info_value) {
        //   tv = tv / sqrt(eval_steps);
        // };
        double delta = poker_state->MaxUtility() - poker_state->MinUtility();
        std::vector<double> sub_values = model->GetCFRTabularValue(inf_input);
        std::vector<std::string> str_actions(actions.size());
        absl::c_transform(
            actions, str_actions.begin(),
            [state](const Action& a) { return state->ActionToString(a); });
        info_record.emplace(
            info, json::Object({
                      {"action", json::CastToArray(str_actions)},
                      {"regret_net", json::CastToArray(info_value)},
                      {"regret_true", json::CastToArray(sub_values)},
                      {"current_net", json::CastToArray(info_curr_policy)},
                      {"current_true", json::CastToArray(tab_curr_policy)},
                      {"global_net", json::CastToArray(global_value)},
                      {"policy_net", json::CastToArray(info_policy)},
                      {"policy_true", json::CastToArray(tab_policy)},
                  }));
      }
      record.insert({"info_set", info_record});
    }
    data_logger.Write(record);
  }
  logger.Print("Got a quit.");
}

struct LearningInfo {
  int num_trajectories;
  int64_t num_states;
  int64_t node_touched;
  int64_t trained_states;
  double loss;
};

std::string SaveAndLoad(const DeepCFRConfig& config,
                        DeviceManager* device_manager, int step,
                        bool value_or_policy, Player player, int device_id) {
  // Always save a checkpoint, either for keeping or for loading the weights
  // to the other sessions. It only allows numbers, so use -1 as "latest".
  std::string checkpoint_path;
  if (value_or_policy) {
    checkpoint_path =
        device_manager->Get(0, device_id)->SaveValue(player, step);
    std::string checkpoint_path_1 =
        device_manager->Get(0, device_id)->SaveCurrentPolicy(player, step);
    std::string checkpoint_path_2 =
        device_manager->Get(0, device_id)->SaveGlobalValue(player, step);
    if (device_manager->Count() > 0 && config.sync_by_restore) {
      for (auto& device : device_manager->GetAll()) {
        device->RestoreValue(player, checkpoint_path);
        device->RestoreCurrentPolicy(player, checkpoint_path_1);
        device->RestoreGlobalValue(player, checkpoint_path_2);
      }
    }
  } else {
    checkpoint_path =
        device_manager->Get(0, device_id)->SavePolicy(player, step);
    if (device_manager->Count() > 0 && config.sync_by_restore) {
      for (auto& device : device_manager->GetAll()) {
        device->RestorePolicy(player, checkpoint_path);
      }
    }
  }
  return checkpoint_path;
}

void SyncModels(const DeepCFRConfig& config, DeviceManager* device_manager,
                int step, bool value_or_policy, Player player, int device_id) {
  if (value_or_policy) {
    for (auto& device : device_manager->GetAll()) {
      device->SyncValueFrom(player,
                            *(device_manager->Get(0, device_id).model()));
      device->SyncCurrentPolicyFrom(
          player, *(device_manager->Get(0, device_id).model()));
      device->SyncGlobalValueFrom(player,
                                  *(device_manager->Get(0, device_id).model()));
    }
  } else {
    for (auto& device : device_manager->GetAll()) {
      device->SyncPolicyFrom(player,
                             *(device_manager->Get(0, device_id).model()));
    }
  }
}

LearningInfo learn_imp_(const Game& game, const DeepCFRConfig& config,
                        DeviceManager* device_manager, int device_id,
                        FileLogger& logger, std::mt19937& rng, int step,
                        ThreadedQueue<Trajectory>* trajectory_queue,
                        CircularBuffer<ReplayNode>& replay_buffer,
                        bool value_or_policy,bool is_z, Player player, int memory_size,
                        int cfr_batch_size, int train_steps,
                        int train_batch_size, double& alpha,
                        std::vector<double>& cfr_value, bool is_training,
                        StopToken* stop) {
  auto start = std::chrono::system_clock::now();
  // Collect trajectories
  int num_trajectories = 0;
  int64_t num_states = 0;
  int64_t node_touched = 0;
  int queue_size = trajectory_queue->Size();
  if (value_or_policy) {
    // need to clear the value buffer.
    replay_buffer.Clear();
  }
  while (!stop->StopRequested() && num_trajectories < cfr_batch_size) {
    absl::optional<Trajectory> trajectory =
        trajectory_queue->Pop(std::chrono::seconds(1));
    if (trajectory) {
      if (value_or_policy) {
        node_touched += trajectory->node_touched;
        cfr_value[player] += trajectory->value;
      }
      num_trajectories += 1;
      for (auto& node : trajectory->states) {
        SPIEL_CHECK_EQ(node.current_player, player);
        SPIEL_CHECK_FALSE(node.legal_actions.empty());
        replay_buffer.Add(node, rng);
        num_states += 1;
      }
    }
  }

  if (config.verbose) {
    auto end = std::chrono::system_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double sec = double(duration.count()) *
                 std::chrono::microseconds::period::num /
                 std::chrono::microseconds::period::den;
    std::cout << "collect used " << sec << "s " << num_states / sec
              << " state/s" << std::endl;
  }
  if (config.verbose) {
    logger.Print("Step: %d", step);
    logger.Print(
        "Collected %7d states from %5d games "
        "game length: %.1f",
        num_states, num_trajectories,
        static_cast<double>(num_states) / num_trajectories);
    logger.Print("Queue size: %d -> %d. Buffer size: %d. States seen: %d",
                 queue_size, trajectory_queue->Size(), replay_buffer.Size(),
                 replay_buffer.TotalAdded());
  }
  // NOTE: we must wait until the actors set the learning flag.
  {
    std::unique_lock<std::mutex> lock(LearningMutex);
    LearningCondition.wait(
        lock, [stop]() { return (Learning) || stop->StopRequested(); });
  }

  if (stop->StopRequested()) {
    return {num_trajectories, num_states, node_touched, 0, 0};
  }

  // not really training, only collecting.
  if (!is_training) {
    return {num_trajectories, num_states, node_touched, 0, 0};
  }

  double loss = 0;
  int64_t trained_states = 0;
  {
    std::vector<CFRNetModel::TrainInputs> train_inputs;
    std::vector<CFRNetModel::InferenceInputs> target_inference_inputs;
    train_inputs.reserve(train_batch_size);
    target_inference_inputs.reserve(train_batch_size);
    DeviceManager::DeviceLoan learn_model =
        device_manager->Get(train_batch_size, device_id);

    auto total_start = std::chrono::system_clock::now();
    // Learn from them.
    for (int train_step = 0; train_step < train_steps; train_step++) {
      auto start = std::chrono::system_clock::now();
      std::vector<ReplayNode> data =
          replay_buffer.Sample(rng, train_batch_size);
      if (value_or_policy) {
        // train current_policy net.
        for (int sample_idx = 0; sample_idx != data.size(); ++sample_idx) {
          const auto& node = data[sample_idx];
          train_inputs.emplace_back(CFRNetModel::TrainInputs{
              node.info_str, node.legal_actions, node.legal_actions,
              node.information, node.policy, node.weight});
        }
      } else {
        // train average policy net.
        for (int sample_idx = 0; sample_idx != data.size(); ++sample_idx) {
          const auto& node = data[sample_idx];
          train_inputs.emplace_back(CFRNetModel::TrainInputs{
              node.info_str, node.legal_actions, node.legal_actions,
              node.information, node.policy, node.weight});
        }
      }
      {
        auto start = std::chrono::system_clock::now();
        if (value_or_policy) {
         if(!is_z)
         { //std::cout<<"training current policy"<<std::endl;
          loss += learn_model->TrainCurrentPolicy(player, train_inputs,
                                                  config.learning_rate);
          }else{
             //std::cout<<"training z policy"<<std::endl;
          loss += learn_model->TrainzPolicy(player, train_inputs,
                                                  config.learning_rate);
          }

        } else {
          std::cout<<"training ave policy"<<std::endl;
          loss += learn_model->TrainPolicy(player, train_inputs,
                                           config.learning_rate);
        }
        if (config.verbose) {
          auto end = std::chrono::system_clock::now();
          auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
              end - start);
          double sec = double(duration.count()) *
                       std::chrono::microseconds::period::num /
                       std::chrono::microseconds::period::den;
          if (train_step % (train_steps / 2) == 0)
            std::cout << "train used " << sec << "s " << data.size() / sec
                      << " state/s" << std::endl;
        }
      }
      trained_states += train_inputs.size();
      train_inputs.clear();
      target_inference_inputs.clear();
    }

    if (config.verbose) {
      auto total_end = std::chrono::system_clock::now();
      auto total_duration =
          std::chrono::duration_cast<std::chrono::microseconds>(total_end -
                                                                total_start);
      double total_sec = double(total_duration.count()) *
                         std::chrono::microseconds::period::num /
                         std::chrono::microseconds::period::den;
      std::cout << "total train used " << total_sec << "s "
                << train_batch_size * train_steps / total_sec << " state/s"
                << std::endl;
    }
  }
  loss = loss / train_steps;
  logger.Print("alpha: %.3f", Alpha);
  logger.Print("loss: %.3f", loss);
  if (config.sync_by_restore) {
    std::string checkpoint_path = SaveAndLoad(
        config, device_manager, -1, value_or_policy, player, device_id);
    if (config.verbose) {
      logger.Print("Checkpoint saved: %s", checkpoint_path);
    }
  }
  if (config.sync_by_copy) {
    SyncModels(config, device_manager, step, value_or_policy, player,
               device_id);
  }
  return {num_trajectories, num_states, node_touched, trained_states, loss};
}

void learner(
    const Game& game, const DeepCFRConfig& config,
    DeviceManager* device_manager,
    std::vector<ThreadedQueue<Trajectory>*> value_trajectory_queues,
    std::vector<ThreadedQueue<Trajectory>*> policy_trajectory_queues,
    std::vector<ThreadedQueue<Trajectory>*> global_value_trajectory_queues,
    std::vector<std::shared_ptr<VPNetEvaluator>> value_evals,
    std::vector<std::shared_ptr<VPNetEvaluator>> policy_z_eval,
    std::vector<std::shared_ptr<VPNetEvaluator>> policy_evals,
    std::vector<std::shared_ptr<VPNetEvaluator>> current_policy_evals,
    StopToken* stop) {
  FileLogger logger(config.path, absl::StrCat("learner", "-mpi-", 0));
  DataLoggerJsonLines data_logger(config.path,
                                  absl::StrCat("learner", "-mpi-", 0), true);
  std::random_device rd;
  std::mt19937 rng(rd());
  int device_id = 0;
  logger.Print("Running the learner on device %d: %s", device_id,
               device_manager->Get(0, device_id)->Device());
  omp_set_num_threads(config.omp_threads);

  // TODO: use resverior buffer.
  std::vector<CircularBuffer<ReplayNode>> value_replay_buffer(
      2, CircularBuffer<ReplayNode>(config.memory_size));
  std::vector<CircularBuffer<ReplayNode>> policy_z_replay_buffer( //new modified
      2, CircularBuffer<ReplayNode>(config.memory_size));
  ReserviorBuffer<ReplayNode> policy_replay_buffer(config.policy_memory_size);

  int64_t sampled_states = 0;
  int64_t trained_states = 0;
  int64_t total_trajectories = 0;
  int64_t node_touched = 0;
  int evaluation_window = config.evaluation_window;

  double alpha = Alpha;
  std::vector<double> cfr_value(2, 0.0);

  absl::Time last = absl::Now() - absl::Seconds(60);
  absl::Time last_save = absl::Now() - absl::Seconds(60);
  for (int step = 1; !stop->StopRequested() &&
                     (config.max_steps == 0 || step <= config.max_steps);
       ++step) {
    std::vector<int64_t> queue_sizes{
        value_trajectory_queues[0]->Size(),
        value_trajectory_queues[1]->Size(),
        policy_trajectory_queues[0]->Size(),
        policy_trajectory_queues[1]->Size(),
        global_value_trajectory_queues[0]->Size(),//new modified
        global_value_trajectory_queues[1]->Size(),//new modified
    };
    int64_t num_states = 0;
    int64_t num_trajectories = 0;
    std::vector<double> losses;
    LRUCacheInfo cache_info;

    std::vector<double> new_cfr_value(2, 0.0);
    for (int player = 0; player < 2; ++player) {
      auto value_trajectory_queue = value_trajectory_queues[player];
      auto global_value_trajectory_queue =
          global_value_trajectory_queues[player];
      auto policy_trajectory_queue = policy_trajectory_queues[player];
      {
        auto learn_info = learn_imp_(
            game, config, device_manager, device_id, logger, rng, step,
            value_trajectory_queue, value_replay_buffer[player], true,false,player,
            config.memory_size, config.cfr_batch_size, config.train_steps,
            config.train_batch_size, alpha, new_cfr_value, true, stop);
        cfr_value[player] = cfr_value[player] * 0.8 +
                            new_cfr_value[player] / config.cfr_batch_size * 0.2;
        value_evals[player]->ClearCache();
        current_policy_evals[player]->ClearCache();

        sampled_states += learn_info.num_states;
        trained_states += learn_info.trained_states;
        num_states += learn_info.num_states;
        total_trajectories += learn_info.num_trajectories;
        num_trajectories += learn_info.num_trajectories;
        node_touched += learn_info.node_touched;
        losses.push_back(learn_info.loss);
      }
      {//new modified
        auto learn_info = learn_imp_(
            game, config, device_manager, device_id, logger, rng, step,
            global_value_trajectory_queue, policy_z_replay_buffer[player], true,true,player,
            config.memory_size, config.cfr_batch_size, config.train_steps,//new modified
            config.train_batch_size, alpha, new_cfr_value, true, stop);
        //cfr_value[player] = cfr_value[player] * 0.8 +
          //                  new_cfr_value[player] / config.cfr_batch_size * 0.2;
        policy_z_eval[player]->ClearCache();

        //sampled_states += learn_info.num_states;
        //trained_states += learn_info.trained_states;
        //num_states += learn_info.num_states;
        //total_trajectories += learn_info.num_trajectories;
        //num_trajectories += learn_info.num_trajectories;
        //node_touched += learn_info.node_touched;
        losses.push_back(learn_info.loss);
      }
      {
        bool is_training = false;
        // train the policy net if we need to evaluate it.
        if (player == 1 && (step % evaluation_window == 0 ||
                            step == config.first_evaluation)) {
          is_training = true;
        }
        int opponent = 1 - player;
        auto learn_info = learn_imp_(
            game, config, device_manager, device_id, logger, rng, step,
            policy_trajectory_queue, policy_replay_buffer, false,false, opponent,
            config.policy_memory_size, config.cfr_batch_size,
            config.policy_train_steps, config.train_batch_size, alpha,
            new_cfr_value, is_training, stop);
        policy_evals[opponent]->ClearCache();

        sampled_states += learn_info.num_states;
        trained_states += learn_info.trained_states;
        num_states += learn_info.num_states;
        total_trajectories += learn_info.num_trajectories;
        num_trajectories += learn_info.num_trajectories;
        node_touched += learn_info.node_touched;
        losses.push_back(learn_info.loss);
      }
      if (player == 0) {
        std::unique_lock<std::mutex> lock(LearningMutex);
        Learning = false;
        NotLearningCondition.notify_all();
      }
    }
    // alpha = adapt(config, alpha, cfr_value);
    Alpha = alpha;

    if (step % evaluation_window == 0 || step == config.first_evaluation) {
      // reduce evaluation cost.
      if (step == evaluation_window * 2 && config.exp_evaluation_window) {
        evaluation_window = evaluation_window * 2;
        if (evaluation_window > config.max_evaluation_window) {
          evaluation_window = evaluation_window / 2;
        }
      }
      std::cout << alpha << " " << cfr_value << std::endl;
      std::cout << "step = " << step
                << " evaluation window = " << evaluation_window << std::endl;

      // after training the policy net, wake up the evaluator.
      {
        // set stats for evaluator.
        std::unique_lock<std::mutex> lock(EvaluatingMutex);
        Evaluating = true;
        EvalSteps = step;
        EvalSampledStates = sampled_states;
        EvalTrainedStates = trained_states;
        EvalNodeTouched = node_touched;
        EvaluatingCondition.notify_all();
      }
    }

    // NOTE: we must wait until the evaluators finishe evaluating.
    {
      std::unique_lock<std::mutex> lock(EvaluatingMutex);
      NotEvaluatingCondition.wait(
          lock, [stop]() { return (!Evaluating) || stop->StopRequested(); });
    }
    {
      std::unique_lock<std::mutex> lock(LearningMutex);
      Learning = false;
      NotLearningCondition.notify_all();
    }

    absl::Time now = absl::Now();
    double seconds = absl::ToDoubleSeconds(now - last);
    double saved_seconds = absl::ToDoubleSeconds(now - last_save);

    if (step % config.checkpoint_freq == 0 ||
        saved_seconds > config.checkpoint_second) {
      for (int player = 0; player < 2; ++player) {
        std::string checkpoint_path =
            SaveAndLoad(config, device_manager, step, true, player, device_id);
        if (config.verbose) {
          logger.Print("Value Checkpoint saved: %s", checkpoint_path);
        }
        checkpoint_path =
            SaveAndLoad(config, device_manager, step, false, player, device_id);
        if (config.verbose) {
          logger.Print("Policy Checkpoint saved: %s", checkpoint_path);
        }
      }
      last_save = now;
    }

    last = now;
    if (config.verbose) {
      DataLogger::Record record = {
          {"step", step},
          {"sampled_states", sampled_states},
          {"trained_states", trained_states},
          {"states_per_s", num_states / seconds},
          {"states_per_s_actor", num_states / (config.actors * seconds)},
          {"total_trajectories", total_trajectories},
          {"trajectories_per_s", num_trajectories / seconds},
          {"queue_size", json::CastToArray(queue_sizes)},
          {"batch_size", value_evals[0]->BatchSizeStats().ToJson()},
          {"batch_size_hist", value_evals[0]->BatchSizeHistogram().ToJson()},
          {"loss", json::CastToArray(losses)},
      };
      value_evals[0]->ResetBatchSizeStats();
      if (config.inference_cache > 0) {
        if (cache_info.size > 0) {
          logger.Print(
              absl::StrFormat("Cache size: %d/%d: %.1f%%, hits: %d, misses: "
                              "%d, hit rate: %.3f%%",
                              cache_info.size, cache_info.max_size,
                              100.0 * cache_info.Usage(), cache_info.hits,
                              cache_info.misses, 100.0 * cache_info.HitRate()));
        }
        record.emplace("cache",
                       json::Object({
                           {"size", cache_info.size},
                           {"max_size", cache_info.max_size},
                           {"usage", cache_info.Usage()},
                           {"requests", cache_info.Total()},
                           {"requests_per_s", cache_info.Total() / seconds},
                           {"hits", cache_info.hits},
                           {"misses", cache_info.misses},
                           {"misses_per_s", cache_info.misses / seconds},
                           {"hit_rate", cache_info.HitRate()},
                       }));
      }

      data_logger.Write(record);
      logger.Print("");
    }
  }
}

bool deep_cfr(DeepCFRConfig config, StopToken* stop) {
  std::shared_ptr<const open_spiel::Game> game;
  std::unordered_map<std::string, GameParameters> game_parameters = {
      {"kuhn_poker", universal_poker::KuhnPokerParameters()},
      {"leduc_poker", universal_poker::LeducPokerParameters()},
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
  std::string model_path = "models";
  if (config.graph_def.empty()) {
    config.graph_def =
        absl::StrJoin({config.game, std::string("sbescfr")}, "_");
    std::string model_path_name =
        absl::StrCat(model_path, "/", config.graph_def + "_value_0_cpu.pb");
    if (file::Exists(model_path_name)) {
      std::cout << "Overwriting existing model: " << model_path_name
                << std::endl;
    } else {
      std::cout << "Creating model: " << model_path_name << std::endl;
    }
    CreateModel(*game, config.learning_rate, config.weight_decay, model_path,
                config.graph_def, "normal", "normal", config.nn_width,
                config.nn_depth, config.num_gpus);
  } else {
    config.graph_def = absl::StrJoin({config.game, config.graph_def}, "_");
    std::string model_path_name =
        absl::StrCat(model_path, "/", config.graph_def + "_value_0_cpu.pb");
    if (file::Exists(model_path_name)) {
      std::cout << "Using existing model: " << model_path_name << std::endl;
    } else {
      std::cout << "Model not found: " << model_path_name << std::endl;
    }
  }

  std::cout << "Playing game: " << config.game << std::endl;

  // NOTE: inference_batch_size and inference_threads may be modified here.
  config.inference_batch_size =
      std::max(1, std::min(config.inference_batch_size, config.actors));

  config.inference_threads = std::max(
      1, std::min(std::min(config.inference_threads, (1 + config.actors) / 2),
                  config.num_cpus));

  {
    file::File fd(config.path + "/config.json", "w");
    fd.Write(json::ToString(config.ToJson(), true) + "\n");
  }

  // NOTE: if we only use tabular cfr (for debugging), we should only use
  // one cpu.
  if ((!config.use_policy_net && !config.use_regret_net)) {
    config.use_tabular = true;
  }
  if (config.use_tabular) {
    config.num_gpus = 0;
    config.num_cpus = 1;
  }
  DeviceManager device_manager;
  for (int i = 0; i < config.num_gpus; ++i) {
    device_manager.AddDevice(CFRNetModel(
        *game, config.path, model_path, config.graph_def, config.omp_threads,
        absl::StrCat("/gpu:", i), config.use_regret_net, true,
        config.use_policy_net, false, false));
    if (config.num_cpus) {
      device_manager.FreezeDevice(i);
    }
  }
  for (int i = 0; i < config.num_cpus; ++i) {
    device_manager.AddDevice(CFRNetModel(*game, config.path, model_path,
                                         config.graph_def, config.omp_threads,
                                         "/cpu:0", config.use_regret_net, true,
                                         config.use_policy_net, false, false));
  }

  if (device_manager.Count() == 0) {
    std::cerr << "No devices specified?" << std::endl;
    return false;
  }

  // Make sure they're all in sync.
  for (Player player = 0; player < 2; ++player) {
    SyncModels(config, &device_manager, 0, true, player, 0);
    SyncModels(config, &device_manager, 0, false, player, 0);
  }

  auto value_0_eval = std::make_shared<VPNetEvaluator>(
      &device_manager, true, Player{0}, false, false,false,
      config.inference_batch_size, config.inference_threads,
      config.inference_cache, (config.actors + config.evaluators) / 16);

  auto value_1_eval = std::make_shared<VPNetEvaluator>(
      &device_manager, true, Player{1}, false, false,false,
      config.inference_batch_size, config.inference_threads,
      config.inference_cache, (config.actors + config.evaluators) / 16);

  auto policy_0_eval = std::make_shared<VPNetEvaluator>(
      &device_manager, false, Player{0}, false, false,false,
      config.inference_batch_size, config.inference_threads,
      config.inference_cache, (config.actors + config.evaluators) / 16);

  auto policy_1_eval = std::make_shared<VPNetEvaluator>(
      &device_manager, false, Player{1}, false, false,false,
      config.inference_batch_size, config.inference_threads,
      config.inference_cache, (config.actors + config.evaluators) / 16);
  //new modified
  auto policy_z_0_eval = std::make_shared<VPNetEvaluator>(
      &device_manager, false, Player{0}, false, true,false,
      config.inference_batch_size, config.inference_threads,
      config.inference_cache, (config.actors + config.evaluators) / 16);

  auto policy_z_1_eval = std::make_shared<VPNetEvaluator>(
      &device_manager, false, Player{1}, false, true,false,
      config.inference_batch_size, config.inference_threads,
      config.inference_cache, (config.actors + config.evaluators) / 16);

  auto current_policy_0_eval = std::make_shared<VPNetEvaluator>(
      &device_manager, false, Player{0}, true, false,false,
      config.inference_batch_size, config.inference_threads,
      config.inference_cache, (config.actors + config.evaluators) / 16);

  auto current_policy_1_eval = std::make_shared<VPNetEvaluator>(
      &device_manager, false, Player{1}, true, false,false,
      config.inference_batch_size, config.inference_threads,
      config.inference_cache, (config.actors + config.evaluators) / 16);

  if (config.play) {
    CFRPolicy policy(policy_0_eval.get(), policy_1_eval.get());
    if (device_manager.Count() > 0 && config.init_strategy_0.length()) {
      for (auto& device : device_manager.GetAll()) {
        device->RestorePolicy(0, config.init_strategy_0);
      }
    }
    if (device_manager.Count() > 0 && config.init_strategy_1.length()) {
      for (auto& device : device_manager.GetAll()) {
        device->RestorePolicy(1, config.init_strategy_1);
      }
    }
    play(*game, config, policy, stop);
    return true;
  }

  if (config.post_evaluation) {
    CFRPolicy policy(policy_0_eval.get(), policy_1_eval.get());
    if (device_manager.Count() > 0 && config.init_strategy_0.length()) {
      for (auto& device : device_manager.GetAll()) {
        device->RestorePolicy(0, config.init_strategy_0);
      }
    }
    if (device_manager.Count() > 0 && config.init_strategy_1.length()) {
      for (auto& device : device_manager.GetAll()) {
        device->RestorePolicy(1, config.init_strategy_1);
      }
    }
    std::vector<Thread> evaluators;
    evaluators.reserve(config.evaluators);
    for (int i = 0; i < config.evaluators; ++i) {
      evaluators.emplace_back(
          [&, i]() { evaluator(*game, config, i, &device_manager, 0, stop); });
    }
    {
      // set stats for evaluator.
      std::unique_lock<std::mutex> lock(EvaluatingMutex);
      Evaluating = true;
      EvalSteps = 1;
      EvalSampledStates = 0;
      EvalTrainedStates = 0;
      EvalNodeTouched = 0;
      EvaluatingCondition.notify_all();
    }
    // NOTE: we must wait until the evaluators finishe evaluating.
    {
      std::unique_lock<std::mutex> lock(EvaluatingMutex);
      NotEvaluatingCondition.wait(
          lock, [stop]() { return (!Evaluating) || stop->StopRequested(); });
    }
    if (!stop->StopRequested()) {
      stop->Stop();
    }
    {
      std::unique_lock<std::mutex> lock(LearningMutex);
      NotLearningCondition.notify_all();
      LearningCondition.notify_all();
    }
    {
      std::unique_lock<std::mutex> lock(EvaluatingMutex);
      EvaluatingCondition.notify_all();
    }
    for (auto& t : evaluators) {
      t.join();
    }
    return true;
  }

  ThreadedQueue<Trajectory> value_0_trajectory_queue(config.cfr_batch_size * 2);
  ThreadedQueue<Trajectory> value_1_trajectory_queue(config.cfr_batch_size * 2);
  ThreadedQueue<Trajectory> policy_z_0_trajectory_queue(config.cfr_batch_size * 2); //new modified
  ThreadedQueue<Trajectory> policy_z_1_trajectory_queue(config.cfr_batch_size * 2); //new modified
  ThreadedQueue<Trajectory> policy_0_trajectory_queue(config.cfr_batch_size *
                                                      2);

  Alpha = config.cfr_rm_scale;
  std::vector<Thread> actors;
  Barrier actor_barrier(config.actors);
  actors.reserve(config.actors);
  for (int i = 0; i < config.actors; ++i) {
    actors.emplace_back([&, i]() {
      actor(*game, config, i,
            {&value_0_trajectory_queue, &value_1_trajectory_queue},
            {&policy_0_trajectory_queue, &policy_0_trajectory_queue},
            {&policy_z_0_trajectory_queue, &policy_z_1_trajectory_queue},//new modified
            {value_0_eval, value_1_eval}, {policy_z_0_eval, policy_z_1_eval},//new modified
            {policy_0_eval, policy_1_eval},
            {current_policy_0_eval, current_policy_1_eval}, &actor_barrier,
            stop);
    });
  }
  std::vector<Thread> evaluators;
  evaluators.reserve(config.evaluators);
  for (int i = 0; i < config.evaluators; ++i) {
    evaluators.emplace_back(
        [&, i]() { evaluator(*game, config, i, &device_manager, 0, stop); });
  }

  learner(*game, config, &device_manager,
          {&value_0_trajectory_queue, &value_1_trajectory_queue},
          {&policy_0_trajectory_queue, &policy_0_trajectory_queue},
          {&policy_z_0_trajectory_queue, &policy_z_1_trajectory_queue},//new modified
          {value_0_eval, value_1_eval}, {policy_z_0_eval, policy_z_1_eval},//new modified
          {policy_0_eval, policy_1_eval},
          {current_policy_0_eval, current_policy_1_eval}, stop);

  if (!stop->StopRequested()) {
    stop->Stop();
  }

  {
    std::unique_lock<std::mutex> lock(LearningMutex);
    NotLearningCondition.notify_all();
    LearningCondition.notify_all();
  }
  {
    std::unique_lock<std::mutex> lock(EvaluatingMutex);
    EvaluatingCondition.notify_all();
  }

  value_0_trajectory_queue.BlockNewValues();
  value_0_trajectory_queue.Clear();
  value_1_trajectory_queue.BlockNewValues();
  value_1_trajectory_queue.Clear();

  policy_0_trajectory_queue.BlockNewValues();
  policy_0_trajectory_queue.Clear();

  std::cout << "Joining all the threads." << std::endl;
  for (auto& t : actors) {
    t.join();
  }
  for (auto& t : evaluators) {
    t.join();
  }

  std::cout << "Exiting cleanly." << std::endl;
  return true;
}

}  // namespace algorithms
}  // namespace open_spiel