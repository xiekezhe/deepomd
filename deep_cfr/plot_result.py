import numpy as np
import re
import json
import os
import fnmatch
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import matplotlib as mpl
from cycler import cycler

decoder = json.JSONDecoder()
cmap = plt.get_cmap('tab20')
colors = list(cmap(i) for i in range(20))
hexcolor = list(map(lambda rgb: '#%02x%02x%02x' %
                    (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)), colors))
mpl.rcParams['axes.prop_cycle'] = cycler(color=hexcolor)


def moving_average(a, n=10):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = (ret[n:] - ret[:-n]) / n
    ret[:n] /= np.arange(1, n + 1)
    return ret


label_name = []
plot_exp = True
plot_regret = True
plot_loss = False
save_data = False
length = 5000000
plot_index = 2

pre_files = []

data = {}
pindex = 0

exp_fig, exp_axs = plt.subplots(2, 3)
exp_fig.set_size_inches(w=13, h=7)
exp_fig_name = ''
# true_leduc_cfr_exp = np.loadtxt("true_leduc_cfr_exp.out")
# true_leduc_cfr_exp = true_leduc_cfr_exp[1:101]
# exp_fig_name = '_deep_cfr_leduc_train_100_256_actor_4_cpu'

# log_dir = "/mnt/c/Users/liuwe/data/my_spiel"
# bash_dir = os.path.join(log_dir, "batch_run")
# result_dir = os.path.join(log_dir, "results-01-08")
result_dir = "./results"
bash_dir = result_dir + "/batch_run"
fig_name = ""
base_comm = ""
# prefix = "all_comp"

# deep cfr
# prefix = "cfr_runmpi.sh"
# train_steps = [4000, 8000]
# cfr_batch_size = [10000]
# train_batch_size = [10000]
# names = []
# bash_files = []
# comms = []
# index = 2
# for t in train_steps:
#     for cb in cfr_batch_size:
#         for tb in train_batch_size:
#             params = [prefix, "--train_steps=" + str(t), "--policy_train_steps=4000",
#                       "--cfr_batch_size=" + str(cb), "--train_batch_size=" + str(tb), str(index)]
#             param_name = "_".join(params)
#             param_name = param_name.replace("-", "_").replace("=", "_")
#             names.append(param_name)
#             index += 1

prefix = "leduc_sbescfr_picfr_exp_adapt_sbescfr.sh"
use_gpus = False
cuda_devices = []
train_steps = [16]
cfr_batch_size = [100]
train_batch_size = [128]
run = "run_sbescfr"
# game = "kuhn_poker"
# game = "FHP2_poker"
games = ["leduc_poker"]
# games = ["FHP_poker"]
# games = ["leduc18_poker"]
# games = ["FHP2_poker"]
cfr_rm_scale = [1e3, 1e2, 1e1, 0.1]
average_type = ["Opponent"]
weight_type = ["Constant"]
cfr_mode = ["CFR"]
comms = []
names = []
index = 0
for game in games:
    for bs in cfr_batch_size:
        for cr in cfr_rm_scale:
            for at in average_type:
                for wt in weight_type:
                    cuda_id = ""
                    if use_gpus:
                        cuda_id = str(
                            cuda_devices[index % len(cuda_devices)])
                    cb = cfr_batch_size[0]
                    tb = train_batch_size[0]
                    params = [prefix, "--game=" + game, "--cfr_batch_size=" + str(bs),
                              "--cfr_rm_scale=" + str(cr)]
                    run_comm = base_comm.replace("placeholder", run)
                    comms.append(" ".join([run_comm] + params))
                    param_name = "_".join(params)
                    param_name = param_name.replace(
                        "-", "_").replace("=", "_")
                    names.append(param_name)
                    index += 1

file_names = []
for i, name in enumerate(names):
    bash_file_name = os.path.join(bash_dir, name + ".txt")
    print(bash_file_name)
    max_len = 1000
    current_len = 0
    with open(bash_file_name, "r") as f:
        while True:
            line = f.readline()
            if current_len > max_len:
                print("not found")
                exit()
            if "Logging directory" in line:
                path = line[line.find("results/") + len("results/"):-1]
                path = path.replace(":", "_")
                file_names.append(path)
                break
            current_len += 1
fig_name = prefix
for i in range(len(names)):
    names[i] = names[i].replace("___", "_").strip("_").replace("cfr_mode", "cfr").replace(
        "average_type", "a").replace("weight_type", "w").replace("cfr_rm_scale", "rm")

pre_files = list(zip(file_names, names))
print(pre_files)
# pre_files = []


for file, name in ([
    # ("08_09_2021.17_26_56.", "leduc_sbescfr_debug"),
    # ("19_07_2021.00_44_52.5415", "leduc_dream_debug"),
    # ("19_07_2021.00_38_32.5734", "kuhn_ossbcfr_debug"),
    # ("19_07_2021.00_22_53.", "kuhn_dream_debug"),
    # ("17_07_2021.23_03_34.13456", "leduc_ossbcfr_rm_10"),
    # ("23_06_2021.22_51_51.8699", "leduc_ossbcfr_back"),
    # ("23_06_2021.00_09_19.6538", "leduc_ossbcfr_bootstrap"),
    # ("22_06_2021.23_39_06.16201", "leduc_ossbcfr"),
    # ("21_06_2021.22_45_02.13563", "kuhn_ossbcfr_fix_cache"),
    # ("21_06_2021.22_35_06.29189", "kuhn_ossbcfr"),
    # ("12_06_2021.23_22_56.", "kuhn_ossbcfr_adapt"),
    # ("05_06_2021.21_29_21.", "leduc_deep_sbescfr_tabular_t_adapt"),
    # ("05_06_2021.21_22_30.", "leduc_deep_sbescfr_tabular_100_t_adapt"),
    # ("03_06_2021.19_54_26.", "leduc_deep_sbescfr_tabular_t_adapt"),
    # ("03_06_2021.15_53_11.", "leduc_deep_sbescfr_tabular_1000"),
    # ("03_06_2021.15_49_23.", "leduc_deep_sbescfr_tabular_adapt"),
    ("03_06_2021.19_57_35.", "leduc_escfr"),
    # ("17_03_2021.13_48_13.16134", "leduc_deep_sbescfr_ts_100_1e-3_adapt"),
    # ("17_03_2021.13_52_32.10605", "leduc_deep_sbescfr_ts_100_1e-3_adapt_1.1"),
    # ("11_03_2021.15_41_38.21657", "leduc_deep_sbescfr_ts_100_1e-3_reser_buffer"),
    # ("11_03_2021.11_30_48.12334", "leduc_deep_sbescfr_ts_100_1e-3"),
    # ("11_03_2021.11_21_34.", "leduc_sbescfr_tabular_1e-5"),
    # ("11_03_2021.11_21_14.", "leduc_sbescfr_tabular_1e-4"),
    # ("11_03_2021.11_11_31.", "leduc_sbescfr_tabular_1e-3"),
    # ("11_03_2021.11_20_07.", "leduc_sbescfr_tabular_1e-2"),
    # ("11_03_2021.11_13_31.", "leduc_escfr_tabular"),
    # ("02_03_2021.12_52_27.", "leduc_subsbcfr_1"),
    # ("02_03_2021.12_54_59.", "leduc_subcfr_1"),
    # ("02_03_2021.21_29_14.14587", "leduc_deep_subsbcfr_escfr_100"),
    # ("02_03_2021.21_28_40.12443", "leduc_deep_reg_subsbcfr_escfr_100"),
    # ("03_03_2021.15_31_24.1259", "leduc_deep_w10_subsbcfr_100"),
    # ("03_03_2021.15_23_19.2335", "leduc_deep_subsbcfr_100"),
    # ("03_03_2021.14_42_12.30802", "leduc_deep_reg_subsbcfr_100"),
    # ("02_03_2021.21_20_49.", "leduc_subsbcfr_escfr_100"),
    # ("03_03_2021.13_13_48.", "leduc_subsbcfr_100"),
    # ("03_03_2021.14_15_39.", "leduc_subcfr_100"),
    # ("03_03_2021.14_25_34.", "leduc_subsbcfr_1000"),
    # ("03_03_2021.14_23_13.", "leduc_subcfr_1000"),
    # ("02_03_2021.13_02_24.", "leduc_sbcfr1_postsbcfr"),
    # ("02_03_2021.09_56_45.", "leduc_sbcfr1_sbcfr"),
    # ("02_03_2021.09_54_02.", "leduc_sbcfr1_cfr"),
    # ("21_02_2021.17_26_47.", "leduc_sbcfr_pre_original"),
    # ("21_02_2021.16_29_24.", "leduc_sbcfr_pre"),
    # ("21_02_2021.16_28_14.", "leduc_sbcfr_post"),
    # ("20_02_2021.22_48_05.", "leduc_sbcfr_pre"),
    # ("21_02_2021.11_13_04.", "leduc_cfr_opp"),
    # ("21_02_2021.11_10_17.", "leduc_cfr_player"),
    # ("20_02_2021.17_35_23.", "leduc_sbcfr_post"),
    # ("20_02_2021.21_18_34.26360", "leduc_ossbcfr_no_anti"),
    # ("10_03_2021.20_40_43.3975", "leduc_ossbcfr_tabular"),
    # ("09_02_2021.19_23_28.943", "leduc_ossbcfr_train_16_dueling_adapt"),
    # ("09_02_2021.19_05_00.27490", "leduc_ossbcfr_train_16_dueling_adapt_r"),
    # ("08_02_2021.12_42_10.20128", "leduc_ossbcfr_train_16_dueling_adapt_max"),
    # ("08_02_2021.12_57_21.9789", "leduc_ossbcfr_train_16_tf"),
    # ("08_02_2021.18_01_52.2694", "leduc_ossbcfr_train_16_tf_max"),
    # ("07_02_2021.20_27_49.3382", "leduc_nfsp"),
    # ("07_02_2021.21_43_00.32041", "ldeuc_nfsp_tf"),
    # ("06_02_2021.21_36_01.", "leduc_deep_cfr"),
    # ("18_01_2021.13_47_07.", "FHP3_local_br_tf"),
    # ("18_01_2021.01_10_13.", "FHP3_local_br_1"),
    # ("18_01_2021.01_37_22.", "FHP3_local_br_2"),
    # ("18_01_2021.01_10_32.", "local_br"),
    # ("05_01_2021.14_29_07.11636", "leduc_deep_cfr_new_tree"),
    # ("09_12_2020.15_44_07.", "leduc_postsbcfr_1e-5"),
    # ("09_12_2020.15_30_29.", "leduc_postsbcfr"),
    # ("09_12_2020.15_31_31.", "leduc_presbcfr"),
    # ("09_12_2020.15_50_24.", "leduc_raw_cfr"),
    # ("09-12-2020.11:05:28", "leduc_cfr_original"),
    # ("05_12_2020.17_30_21.", "leduc_ossbcfr_tf_pow_2"),
    # ("05_12_2020.16_33_38.", "leduc_ossbcfr_tf_pow_2"),
    # ("05_12_2020.16_32_28.", "leduc_ossbcfr_tf_pow_0"),
    # ("05_12_2020.16_33_14.", "leduc_ossbcfr_tf_pow_1"),
    # ("05_12_2020.16_58_24.", "leduc_ossbcfr_tf_pow_0_1e-2"),
    # ("05_12_2020.16_41_34.", "leduc_ossbcfr_tf_pow_0_1e-5"),
    # ("20_11_2020.15_16_43.", "leduc_ossbcfr_omp_8_actors_4_tf"),
    # ("20_11_2020.15_08_20.", "leduc_ossbcfr_tf"),
    # ("20_11_2020.15_07_34.", "leduc_deep_cfr_tf"),
    # ("23_10_2020.20_43_09.", "ledcu_ossbcfr_div_t"),
    # ("23_10_2020.21_17_24.", "leduc_ossbcfr_inf_no_target"),
    # ("23_10_2020.21_13_32.", "ledcu_ossbcfr_no_target"),
    # ("23_10_2020.21_15_15.", "leduc_ossbcfr_train_2"),
    # ("23-09-2020.22:00:34", "deep_subsbcfr_1_regret_100_100_post"),
    # ("23-09-2020.22:00:05", "deep_subsbcfr_1_regret_100_100_adapt_0.99"),
    # ("23-09-2020.21:57:06", "deep_subsbcfr_1_regret_100_100"),
    # ("23-09-2020.21:57:06", "deep_subsbcfr_1_regret_100_100"),
    # ("20-09-2020.19:57:05", "subsbcfr_1_100"),
    # ("20-09-2020.19:22:07", "subcfr_1"),
    # ("20-09-2020.19:57:13", "subpostsbcfr_1"),
    # ("20-09-2020.19:39:40", "postsbcfr_1"),
    # ("20-09-2020.19:36:45", "sbcfr_1"),
    # ("20-09-2020.19:28:10", "cfr_1"),
    # ("23-08-2020.16:55:02", "leduc_deepcfr_new_adam"),
    # ("21-08-2020.16:21:11", "leduc_deep_cfr_new_adam_mpi_4"),
    # ("21-08-2020.01:47:53", "leduc_ossbcfr_net_new_adam"),
    # ("21-08-2020.01:47:56", "leduc_nfsp_net_new_adam"),
    # ("13-07-2020.17:07:16", "sbcfr"),
    # ("13-07-2020.16:37:28", "subcfr"),
    # ("13-07-2020.16:36:37", 'cfr'),
] + pre_files):
    exploit = []
    localbr = []
    localbrstd = []
    num_step = []
    node_touched = []
    alpha = []
    num_trained = []
    num_sampled = []
    time_relative = []
    value = []
    regrets = {}
    currents = {}
    global_values = {}
    strategies = {}
    N = {}
    s_loss = {}
    file = os.path.join(result_dir, file, 'evaluator-0-mpi-0.jsonl')
    if 'jsonl' in file:
        print(file)
        with open(file) as fp:
            line = fp.readline()
            while line:
                res = re.search('Exploitability', line)
                if res:
                    try:
                        exps = decoder.decode(line)
                    except:
                        print(file)
                        print(line)
                        exit()
                    exploit.append(float(exps['Exploitability']))
                    num_step.append(int(exps['Step']))
                    if 'LocalBestResponse' in exps:
                        localbr.append(float(exps['LocalBestResponse']))
                    if 'LocalBestResponse_std' in exps:
                        localbrstd.append(float(exps['LocalBestResponse_std']))
                    if 'Touch' in exps:
                        node_touched.append(int(exps['Touch']))
                    if 'Alpha' in exps:
                        alpha.append(float(exps['Alpha']))
                    time_relative.append(float(exps['time_rel']))
                    if 'Sampled states' in exps:
                        num_sampled.append(int(exps['Sampled states']))
                    if 'Trained states' in exps:
                        num_trained.append(int(exps['Trained states']))
                    try:
                        info_sets = exps['info_set']
                        for inf, info_values in info_sets.items():
                            actions = info_values['action']
                            actions = [str(a) for a in actions]
                            regret_net = info_values['regret_net']
                            policy_net = info_values['policy_net']
                            current_net = info_values['current_net']
                            global_net = info_values['global_net']
                            regret_true = info_values['regret_true']
                            current_true = info_values['current_true']
                            policy_true = info_values['policy_true']
                            if inf not in regrets:
                                regrets[inf] = {a: [] for a in actions}
                            if inf not in strategies:
                                strategies[inf] = {a: [] for a in actions}
                            if inf not in currents:
                                currents[inf] = {a: [] for a in actions}
                            if inf not in global_values:
                                global_values[inf] = {a: [] for a in actions}
                            for a, v in zip(actions, regret_net):
                                if a not in regrets[inf]:
                                    regrets[inf][a] = []
                                regrets[inf][a].append(v)
                            for a, v in zip(actions, regret_true):
                                a = a + '_true'
                                if a not in regrets[inf]:
                                    regrets[inf][a] = []
                                regrets[inf][a].append(v)
                            for a, v in zip(actions, policy_net):
                                if a not in strategies[inf]:
                                    strategies[inf][a] = []
                                strategies[inf][a].append(v)
                            for a, v in zip(actions, policy_true):
                                a = a + '_true'
                                if a not in strategies[inf]:
                                    strategies[inf][a] = []
                                strategies[inf][a].append(v)
                            for a, v in zip(actions, current_net):
                                if a not in currents[inf]:
                                    currents[inf][a] = []
                                currents[inf][a].append(v)
                            for a, v in zip(actions, current_true):
                                a = a + '_true'
                                if a not in currents[inf]:
                                    currents[inf][a] = []
                                currents[inf][a].append(v)
                            for a, v in zip(actions, global_net):
                                if a not in global_values[inf]:
                                    global_values[inf][a] = []
                                global_values[inf][a].append(v)
                    except:
                        pass

                line = fp.readline()
                if len(exploit) > length:
                    break
        if save_data:
            if name not in data:
                data[name] = {}
            data[name]['exploit'] = exploit[0:2000]
            data[name]['num_trained'] = num_trained[0:2000]
            data[name]['node_touched'] = node_touched[0:2000]
            data[name]['alpha'] = alpha[0:2000]

        if pindex == 0:
            exp_fig_name = fig_name + name + ".png"
        color_i = colors[pindex % 20]
        if plot_exp:
            print(name, ":", exploit[-1])
            # exploit = moving_average(exploit, n=100)
            exp_axs[0][0].loglog(num_step[0:length],
                                 exploit[0:length], label=name, color=color_i)
            # if localbr:
            #     exp_axs[0][0].loglog(num_step[0:length],
            #                          np.maximum(np.array(localbr[0:length]), 0), label=name, color=color_i, linestyle='--')
            #     exp_axs[0][0].fill_between(num_step[0:length],
            #                                np.maximum(np.array(localbr[0:length]) - np.array(localbrstd[0:length]) * 2, 0), np.maximum(np.array(localbr[0:length]) + np.array(localbrstd[0:length]) * 2, 0), color=color_i, linestyle='--', alpha=0.3)
            # exp_axs[0][0].loglog(num_step[0:len(true_leduc_cfr_exp)],
            #                      true_leduc_cfr_exp, label="true_leduc_cfr_exp", color=colors[pindex + 1])
            exp_axs[0][0].set_ylabel('exploitability')
            exp_axs[0][0].set_xlabel('steps')
            # exp_axs[0][0].set_xlim([100, num_step[-1]])
            exp_axs[0][0].legend(prop={'size': 5})
            if num_sampled:
                exp_axs[0][1].loglog(num_sampled[0:length],
                                     exploit[0:length], color=color_i)
                exp_axs[0][1].set_ylabel('exploitability')
                exp_axs[0][1].set_xlabel('number sampled')
            if num_trained:
                exp_axs[1][0].loglog(num_trained[0:length],
                                     exploit[0:length], color=color_i)
                exp_axs[1][0].set_ylabel('exploitability')
                exp_axs[1][0].set_xlabel('number trained')
            if time_relative:
                exp_axs[1][1].semilogy(time_relative[0:length],
                                       exploit[0:length], label=name, color=color_i)
                exp_axs[1][1].set_ylabel('exploitability')
                exp_axs[1][1].set_xlabel('time /s')
            if node_touched:
                exp_axs[0][2].loglog(node_touched[0:length],
                                     exploit[0:length], color=color_i)
                exp_axs[0][2].set_ylabel('exploitability')
                exp_axs[0][2].set_xlabel('node touched')
            if alpha:
                exp_axs[1][2].semilogy(num_step[0:length],
                                       alpha[0:length], color=color_i)
                exp_axs[1][2].set_ylabel('alpha')
                exp_axs[1][2].set_xlabel('step')
            exp_fig.savefig('results/images/exp_' + exp_fig_name)

        if plot_regret and pindex == plot_index:
            ax_num = len(regrets)
            # w = h = int(np.floor(np.sqrt(ax_num)))
            w = 4
            h = int(np.ceil(ax_num / w))
            h = max(2, min(h, 20))
            ax_num = w * h
            fig, axs = plt.subplots(h, w)
            fig.set_size_inches(4 * w, 4 * h)
            keys = list(regrets.keys())
            keys = sorted(keys)
            for i, inf in enumerate(keys):
                reg = regrets[inf]
                if i >= ax_num:
                    break
                ax = axs[i // w][i % w]
                for a, v in reg.items():
                    if 'true' in a:
                        # pass
                        ax.plot(np.array(v[0:length]), '--', label=a)
                    else:
                        # v = np.array(v[0:length]) / /
                        #     np.sqrt(1 + np.arange(len(v[0:length])))
                        # print(v)
                        # v = np.maximum(v, -10)
                        # v = np.minimum(v, 10)
                        ax.plot(v[0:length], label=a)
                v = reg[list(reg.keys())[0]]
                ax.plot([0 for _ in range(len(v))][0:length], '--k')
                ax.set_title(inf)
                ax.legend()
                i += 1
            fig.savefig('results/images/regret_' + exp_fig_name)

            fig, axs = plt.subplots(h, w)
            fig.set_size_inches(4 * w, 4 * h)
            for i, inf in enumerate(keys):
                reg = global_values[inf]
                if i >= ax_num:
                    break
                ax = axs[i // w][i % w]
                for a, v in reg.items():
                    # v = np.maximum(v, -10)
                    # v = np.minimum(v, 10)
                    ax.plot(v[0:length], label=a)
                v = reg[list(reg.keys())[0]]
                ax.plot([0 for _ in range(len(v))][0:length], '--k')
                ax.set_title(inf)
                ax.legend()
                i += 1
            fig.savefig('results/images/global_' + exp_fig_name)

            fig, axs = plt.subplots(h, w)
            fig.set_size_inches(4 * w, 4 * h)
            for i, inf in enumerate(keys):
                reg = currents[inf]
                if i >= ax_num:
                    break
                ax = axs[i // w][i % w]
                for a, v in reg.items():
                    if 'true' in a:
                        ax.plot(v[0:length], label=a, linestyle='--')
                    else:
                        ax.plot(v[0:length], label=a)
                v = reg[list(reg.keys())[0]]
                ax.plot([0 for _ in range(len(v))][0:length], '--k')
                ax.plot([1 for _ in range(len(v))][0:length], '--k')
                ax.set_title(inf)
                ax.legend()
                i += 1
            fig.savefig('results/images/current_' + exp_fig_name)
            fig, axs = plt.subplots(h, w)
            fig.set_size_inches(4 * w, 4 * h)

            for i, inf in enumerate(keys):
                if i >= ax_num:
                    break
                reg = strategies[inf]
                ax = axs[i // w][i % w]
                for a, v in reg.items():
                    if 'true' in a:
                        pass
                        # ax.plot(v[0:length], label=a, linestyle='--')
                    else:
                        ax.plot(v[0:length], label=a)
                v = reg[list(reg.keys())[0]]
                ax.plot([0 for _ in range(len(v))][0:length], '--k')
                ax.plot([1 for _ in range(len(v))][0:length], '--k')
                ax.set_title(inf)
                ax.legend()
            plt.savefig('results/images/strategy_' + exp_fig_name)

        if plot_loss:
            plt.figure(100)
            # for loss_name, loss_v in s_loss.items():
            #     plt.semilogy(loss_v[0:length * 10], label=loss_name + ': ' + name + ': ' + file[4:])
            loss = np.sum(np.array([l for l in s_loss.values()]), axis=0)[
                0:length]
            plt.semilogy(loss, label=name)
            plt.title('losses')
            # plt.legend(prop={'size': 5})
            plt.legend()
            plt.savefig('results/images/losses_' + exp_fig_name + name)

    pindex += 1

if save_data:
    with open("paper/" + prefix + "_data.json", 'w') as fp:
        json.dump(data, fp)
