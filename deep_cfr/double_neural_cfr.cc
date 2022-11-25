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
#include "open_spiel/utils/data_logger.h"
#include "open_spiel/utils/dict_buffer.h"
#include "open_spiel/utils/file.h"
#include "open_spiel/utils/json.h"
#include "open_spiel/utils/logger.h"
#include "open_spiel/utils/lru_cache.h"
#include "open_spiel/utils/stats.h"
#include "open_spiel/utils/thread.h"
#include "open_spiel/utils/threaded_queue.h"
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
int64_t NodeTouched = 0;
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
           ThreadedQueue<Trajectory>* value_trajectory_queue,
           ThreadedQueue<Trajectory>* policy_trajectory_queue,
           std::shared_ptr<VPNetEvaluator> value_0_eval,
           std::shared_ptr<VPNetEvaluator> value_1_eval,
           std::shared_ptr<VPNetEvaluator> policy_eval, Barrier* actor_barrier,
           StopToken* stop) {
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
  DeepRSCFRSolver solver(game, value_0_eval, value_1_eval, policy_eval,
                         config.use_regret_net, config.use_policy_net,
                         config.use_tabular, &rng);
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
    {
      // wait until not learning.
      std::unique_lock<std::mutex> lock(LearningMutex);
      NotLearningCondition.wait(
          lock, [stop]() { return (!Learning) || stop->StopRequested(); });
    }
    logger->Print("starting running");
    for (Player player = 0; player < 2; ++player) {
      int before_queue_size = value_trajectory_queue->Size();
      int before_policy_queue_size = policy_trajectory_queue->Size();
      solver.clear_updated();
      for (int game_num = 1;
           !stop->StopRequested() && (game_num <= cfr_batch_per_actor);
           ++game_num) {
        auto trajectories = solver.RunIteration(player, step);
        NodeTouched += solver.NodeTouched();
        if (!value_trajectory_queue->Push(trajectories.first,
                                          std::chrono::seconds(10))) {
          logger->Print("Failed to push a trajectory after 10 seconds.");
        }
        if (!policy_trajectory_queue->Push(trajectories.second,
                                           std::chrono::seconds(10))) {
          logger->Print("Failed to push a trajectory after 10 seconds.");
        }
      }
      int queue_size = value_trajectory_queue->Size();
      if (config.verbose) {
        logger->Print(
            "Step %d\npush %d trajectories to queue %d, queue size: %d -> %d, "
            "policy size: %d -> %d.",
            step, cfr_batch_per_actor, player, before_queue_size, queue_size,
            before_policy_queue_size, policy_trajectory_queue->Size());
      }
    }
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
  } else {
    if (config.game != "kuhn_poker") {
      return;
    }
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
    for (Player player = 0; player < 2; ++player) {
      model->SyncPolicyFrom(player, *(device_manager->Get(0, 0).model()));
      model->SyncValueFrom(player, *(device_manager->Get(0, 0).model()));
    }
    double exp = expSolver();
    {
      std::unique_lock<std::mutex> lock(EvaluatingMutex);
      Evaluating = false;
      NotEvaluatingCondition.notify_all();
    }
    logger.Print("Exploitability: %.3f", exp);
    std::cout << absl::StrFormat("Exploitability: %.3f", exp) << std::endl;
    DataLogger::Record record = {
        {"Step", eval_steps},
        {"Sampled states", sampled_states},
        {"Trained states", trained_states},
        {"Exploitability", exp},
        {"Touch", EvalNodeTouched},
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
          tv = tv / config.cfr_batch_size / sqrt(eval_steps);
        };
        for (auto& tv : info_value) {
          tv = tv;
        };
        std::vector<double> tabular_policy =
            model->GetCFRTabularPolicy({inf_input}, true);
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
  logger.Print("Got a quit.");
}

struct LearningInfo {
  int num_trajectories;
  int64_t num_states;
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
    if (step % config.checkpoint_freq == 0 || config.sync_by_restore) {
      checkpoint_path =
          device_manager->Get(0, device_id)
              ->SaveValue(player,
                          step % config.checkpoint_freq == 0 ? step : -1);
      if (device_manager->Count() > 0 && config.sync_by_restore) {
        for (auto& device : device_manager->GetAll()) {
          device->RestoreValue(player, checkpoint_path);
        }
      }
    }
  } else {
    if (step % config.checkpoint_freq == 0 || config.sync_by_restore) {
      checkpoint_path =
          device_manager->Get(0, device_id)
              ->SavePolicy(player,
                           step % config.checkpoint_freq == 0 ? step : -1);
      if (device_manager->Count() > 0 && config.sync_by_restore) {
        for (auto& device : device_manager->GetAll()) {
          device->RestorePolicy(player, checkpoint_path);
        }
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
                        DictBuffer<std::string, ReplayNode>& replay_buffer,
                        bool value_or_policy, Player player, int memory_size,
                        int cfr_batch_size, int train_steps,
                        int train_batch_size, StopToken* stop) {
  // Collect trajectories
  int num_trajectories = 0;
  int64_t num_states = 0;
  int queue_size = trajectory_queue->Size();
  // NOTE: double cfr requires to clear buffer on every iteration.
  replay_buffer.Clear();
  while (!stop->StopRequested() && num_trajectories < cfr_batch_size) {
    absl::optional<Trajectory> trajectory =
        trajectory_queue->Pop(std::chrono::seconds(1));
    if (trajectory) {
      num_trajectories += 1;
      for (ReplayNode& node : trajectory->states) {
        CFRNetModel::InferenceInputs input = {node.info_str, node.legal_actions,
                                              node.information};
        // if (node.information ==
        //     std::vector<double>{50, 46, 0, 2, 4, 2, 4, 8, -1, -1}) {
        //   std::cout << "add " << node.value << std::endl;
        // }
        if (value_or_policy) {
          for (int i = 0; i != node.value.size(); ++i) {
            node.value[i] /= cfr_batch_size / 2;
          }
          // When training regret net, batch cfr is used.
          replay_buffer.Add({node.info_str, node},
                            [](const ReplayNode& lhs, const ReplayNode& rhs) {
                              return lhs + rhs;
                            });
        } else {
          // When training policy net, batch cfr is used.
          replay_buffer.Add(
              {node.info_str, node},
              [](const ReplayNode& lhs, const ReplayNode& rhs) { return lhs; });
        }
        num_states += 1;
      }
    }
  }
  // NOTE: there is no reason to clear trajectory_queue.
  // trajectory_queue->Clear();
  logger.Print("Step: %d", step);
  logger.Print(
      "Collected %7d states from %5d games "
      "game length: %.1f",
      num_states, num_trajectories,
      static_cast<double>(num_states) / num_trajectories);
  logger.Print("Queue size: %d -> %d. Buffer size: %d. States seen: %d",
               queue_size, trajectory_queue->Size(), replay_buffer.Size(),
               replay_buffer.TotalAdded());

  // NOTE: we must wait until the actors set the learning flag.
  {
    std::unique_lock<std::mutex> lock(LearningMutex);
    LearningCondition.wait(
        lock, [stop]() { return (Learning) || stop->StopRequested(); });
  }

  if (stop->StopRequested()) {
    return {num_trajectories, num_states, 0};
  }

  double loss = 0;
  int64_t trained_states = 0;
  // TODO: May need a parameter config.minimum_buffer_size_to_train.
  // NOTE: Double CFR may has much less samples in the buffer than the samples
  // sampled, as the samples belong to the same information set are summed up
  // (for regret), or the dumplicated are removed (for strategy).
  if (replay_buffer.Size() < train_batch_size) {
    logger.Print(absl::StrFormat(
        "Warning: not enough samples. valid = %d, require = %d.",
        replay_buffer.Size(), train_batch_size));
  }
  {
    std::vector<CFRNetModel::TrainInputs> train_inputs;
    std::vector<CFRNetModel::InferenceInputs> target_inference_inputs;
    train_inputs.reserve(train_batch_size);
    target_inference_inputs.reserve(train_batch_size);
    {  // Extra scope to return the device for use for inference asap.
      DeviceManager::DeviceLoan learn_model =
          device_manager->Get(train_batch_size, device_id);

      // NOTE: for double nueral cfr, target networks are used for training.
      if (value_or_policy) {
        learn_model->SyncValue(player);
      } else {
        learn_model->SyncPolicy(player);
      }

      // Learn from them.
      for (int train_step = 0; train_step < train_steps; train_step++) {
        std::vector<ReplayNode> data =
            replay_buffer.Sample(rng, train_batch_size);
        std::vector<CFRNetModel::InferenceOutputs> target_outputs;
        if (step > 1) {
          for (int i = 0; i != data.size(); ++i) {
            const auto& node = data[i];
            target_inference_inputs.emplace_back(CFRNetModel::InferenceInputs{
                node.info_str, node.legal_actions, node.information});
          }
          if (value_or_policy) {
            target_outputs =
                learn_model->InfTargetValue(player, target_inference_inputs);
            // target_outputs = learn_model->GetCFRTabularValues(
            //     player, target_inference_inputs);
          } else {
            target_outputs =
                learn_model->InfTargetPolicy(player, target_inference_inputs);
            // target_outputs = learn_model->GetCFRTabularPolicies(
            //     player, target_inference_inputs, false);
          }

          for (int i = 0; i != data.size(); ++i) {
            const auto& node = data[i];
            std::vector<double> target_value = target_outputs[i].value;
            SPIEL_CHECK_EQ(target_value.size(), node.value.size());
            SPIEL_CHECK_EQ(target_outputs[i].legal_actions.size(),
                           node.legal_actions.size());
            for (int j = 0; j != target_value.size(); ++j) {
              if (value_or_policy) {
                target_value[j] =
                    std::max(0.0, target_value[j] + node.value[j]);
              } else {
                target_value[j] = (target_value[j] + node.value[j]);
              }
            }
            train_inputs.emplace_back(CFRNetModel::TrainInputs{
                node.info_str, node.legal_actions, node.legal_actions,
                node.information, target_value, 1});
          }
        } else {
          for (int i = 0; i != data.size(); ++i) {
            const auto& node = data[i];
            train_inputs.emplace_back(CFRNetModel::TrainInputs{
                node.info_str, node.legal_actions, node.legal_actions,
                node.information, node.value, 1});
          }
        }
        if (value_or_policy) {
          loss += learn_model->TrainValue(player, train_inputs,
                                          config.learning_rate);

        } else {
          loss += learn_model->TrainPolicy(player, train_inputs,
                                           config.learning_rate);
        }
        trained_states += train_inputs.size();
        train_inputs.clear();
        target_inference_inputs.clear();
      }
    }
    loss = loss / train_steps;
    logger.Print("loss: %.3f", loss);
    std::string checkpoint_path = SaveAndLoad(
        config, device_manager, step, value_or_policy, player, device_id);
    logger.Print("Checkpoint saved: %s", checkpoint_path);
    if (config.sync_by_copy) {
      SyncModels(config, device_manager, step, value_or_policy, player,
                 device_id);
    }
  }
  return {num_trajectories, num_states, trained_states, loss};
}

void learner(const Game& game, const DeepCFRConfig& config,
             DeviceManager* device_manager,
             std::shared_ptr<VPNetEvaluator> eval_0,
             std::shared_ptr<VPNetEvaluator> eval_1,
             std::shared_ptr<VPNetEvaluator> policy_eval,
             ThreadedQueue<Trajectory>* value_trajectory_queue,
             ThreadedQueue<Trajectory>* policy_trajectory_queue,
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
  DictBuffer<std::string, ReplayNode> value_replay_buffer(config.memory_size);
  DictBuffer<std::string, ReplayNode> policy_replay_buffer(
      config.policy_memory_size);
  std::vector<std::shared_ptr<VPNetEvaluator>> value_eval{eval_0, eval_1};

  int64_t sampled_states = 0;
  int64_t trained_states = 0;
  int64_t total_trajectories = 0;
  int evaluation_window = config.evaluation_window;

  // Actor threads have likely been contributing for a while, so put `last` in
  // the past to avoid a giant spike on the first step.
  absl::Time last = absl::Now() - absl::Seconds(60);
  for (int step = 1; !stop->StopRequested() &&
                     (config.max_steps == 0 || step <= config.max_steps);
       ++step) {
    std::vector<int64_t> queue_sizes{
        value_trajectory_queue->Size(),
        policy_trajectory_queue->Size(),
    };
    int64_t num_states = 0;
    int64_t num_trajectories = 0;
    std::vector<double> losses;
    LRUCacheInfo cache_info;
    // NOTE: double CFR shares regret net and policy net among players.
    auto learn_info =
        learn_imp_(game, config, device_manager, device_id, logger, rng, step,
                   value_trajectory_queue, value_replay_buffer, true, 0,
                   config.memory_size, config.cfr_batch_size * 2,
                   config.train_steps, config.train_batch_size, stop);
    if (config.inference_cache > 0) {
      cache_info = eval_0->CacheInfo();
    }
    value_eval[0]->ClearCache();
    value_eval[1]->ClearCache();

    sampled_states += learn_info.num_states;
    trained_states += learn_info.trained_states;
    num_states += learn_info.num_states;
    total_trajectories += learn_info.num_trajectories;
    num_trajectories += learn_info.num_trajectories;
    losses.push_back(learn_info.loss);
    learn_info =
        learn_imp_(game, config, device_manager, device_id, logger, rng, step,
                   policy_trajectory_queue, policy_replay_buffer, false, 0,
                   config.policy_memory_size, config.cfr_batch_size * 2,
                   config.policy_train_steps, config.train_batch_size, stop);
    policy_eval->ClearCache();

    sampled_states += learn_info.num_states;
    trained_states += learn_info.trained_states;
    num_states += learn_info.num_states;
    total_trajectories += learn_info.num_trajectories;
    num_trajectories += learn_info.num_trajectories;
    losses.push_back(learn_info.loss);
    int64_t node_touched = NodeTouched;
    // after training the policy net, wake up the evaluator.
    if (step % evaluation_window == 0 || step == config.first_evaluation) {
      // reduce evaluation cost.
      if (step == evaluation_window * 10 && config.exp_evaluation_window) {
        evaluation_window = evaluation_window * 10;
        if (evaluation_window > config.max_evaluation_window) {
          evaluation_window = evaluation_window / 10;
        }
      }
      std::cout << "step = " << step
                << " evaluation window = " << evaluation_window << std::endl;
      // set stats for evaluator.
      std::unique_lock<std::mutex> lock(EvaluatingMutex);
      Evaluating = true;
      EvalSteps = step;
      EvalSampledStates = sampled_states;
      EvalTrainedStates = trained_states;
      EvalNodeTouched = node_touched;
      EvaluatingCondition.notify_all();
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
    last = now;
    DataLogger::Record record = {
        {"step", step},
        {"sampled_states", sampled_states},
        {"trained_states", trained_states},
        {"states_per_s", num_states / seconds},
        {"states_per_s_actor", num_states / (config.actors * seconds)},
        {"total_trajectories", total_trajectories},
        {"trajectories_per_s", num_trajectories / seconds},
        {"queue_size", json::CastToArray(queue_sizes)},
        {"batch_size", eval_0->BatchSizeStats().ToJson()},
        {"batch_size_hist", eval_0->BatchSizeHistogram().ToJson()},
        {"loss", json::CastToArray(losses)},
    };
    eval_0->ResetBatchSizeStats();
    eval_1->ResetBatchSizeStats();
    if (config.inference_cache > 0) {
      if (cache_info.size > 0) {
        logger.Print(absl::StrFormat(
            "Cache size: %d/%d: %.1f%%, hits: %d, misses: "
            "%d, hit rate: %.3f%%",
            cache_info.size, cache_info.max_size, 100.0 * cache_info.Usage(),
            cache_info.hits, cache_info.misses, 100.0 * cache_info.HitRate()));
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
        absl::StrJoin({config.game, std::string("deep_cfr")}, "_");
    std::string model_path_name =
        absl::StrCat(model_path, "/", config.graph_def + "_value_0_cpu.pb");
    if (file::Exists(model_path_name)) {
      std::cout << "Overwriting existing model: " << model_path_name
                << std::endl;
    } else {
      std::cout << "Creating model: " << model_path_name << std::endl;
    }
    CreateModel(*game, config.learning_rate, config.weight_decay, model_path,
                config.graph_def, config.nn_model, config.nn_model,
                config.nn_width, config.nn_depth, config.num_gpus);
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

  // NOTE: if we only use tabular cfr (for debugging), we should only use one
  // cpu.
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
        absl::StrCat("/gpu:", i), config.use_regret_net, false,
        config.use_policy_net, false, true));
    if (config.num_cpus) {
      device_manager.FreezeDevice(i);
    }
  }
  for (int i = 0; i < config.num_cpus; ++i) {
    device_manager.AddDevice(CFRNetModel(*game, config.path, model_path,
                                         config.graph_def, config.omp_threads,
                                         "/cpu:0", config.use_regret_net, false,
                                         config.use_policy_net, false, true));
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
      &device_manager, true, Player{0}, false, false,
      config.inference_batch_size, config.inference_threads,
      config.inference_cache, (config.actors + config.evaluators) / 16);

  auto value_1_eval = std::make_shared<VPNetEvaluator>(
      &device_manager, true, Player{1}, false, false,
      config.inference_batch_size, config.inference_threads,
      config.inference_cache, (config.actors + config.evaluators) / 16);

  auto policy_eval = std::make_shared<VPNetEvaluator>(
      &device_manager, false, Player{0}, false, false,
      config.inference_batch_size, config.inference_threads,
      config.inference_cache, (config.actors + config.evaluators) / 16);

  ThreadedQueue<Trajectory> value_trajectory_queue(config.cfr_batch_size * 4);
  ThreadedQueue<Trajectory> policy_trajectory_queue(config.cfr_batch_size * 4);

  std::vector<Thread> actors;
  Barrier actor_barrier(config.actors);
  actors.reserve(config.actors);
  for (int i = 0; i < config.actors; ++i) {
    actors.emplace_back([&, i]() {
      actor(*game, config, i, &value_trajectory_queue, &policy_trajectory_queue,
            value_0_eval, value_1_eval, policy_eval, &actor_barrier, stop);
    });
  }
  std::vector<Thread> evaluators;
  evaluators.reserve(config.evaluators);
  for (int i = 0; i < config.evaluators; ++i) {
    evaluators.emplace_back(
        [&, i]() { evaluator(*game, config, i, &device_manager, 0, stop); });
  }

  learner(*game, config, &device_manager, value_0_eval, value_1_eval,
          policy_eval, &value_trajectory_queue, &policy_trajectory_queue, stop);

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

  value_trajectory_queue.BlockNewValues();
  value_trajectory_queue.Clear();

  policy_trajectory_queue.BlockNewValues();
  policy_trajectory_queue.Clear();

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