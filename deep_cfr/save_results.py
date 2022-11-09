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
plot_regret = False
plot_loss = False
save_data = True
length = 5000000
plot_index = 0

pre_files = []

data = {}
pindex = 0

exp_fig, exp_axs = plt.subplots(2, 3)
exp_fig.set_size_inches(w=13, h=7)
exp_fig_name = ''
# true_leduc_cfr_exp = np.loadtxt("true_leduc_cfr_exp.out")
# true_leduc_cfr_exp = true_leduc_cfr_exp[1:101]
# exp_fig_name = '_deep_cfr_leduc_train_100_256_actor_4_cpu'

log_dir = "/mnt/c/Users/liuwe/data/my_spiel"
bash_dir = os.path.join(log_dir, "batch_run")
result_dir = os.path.join(log_dir, "results-01-18")
# result_dir = "./results"
fig_name = ""
base_comm = ""
prefix = "all_comp"
# prefix = ""

# deep cfr
# prefix = "cfr_runmpi.sh"
# train_steps = [1000, 2000, 4000, 8000]
# cfr_batch_size = [10000]
# train_batch_size = [10000]
# names = []
# bash_files = []
# for t in train_steps:
#     for cb in cfr_batch_size:
#         for tb in train_batch_size:
#             params = ["--train_steps=" + str(t), "--policy_train_steps=4000",
#                       "--cfr_batch_size=" + str(cb), "--train_batch_size=" + str(tb)]
#             param_name = "_".join(params)
#             param_name = param_name.replace("-", "_").replace("=", "_")
#             names.append(param_name)

# nfsp
# prefix = "nfsp_runmpi_nfsp.sh"
# train_steps = [2, 16, 128, 1024]
# cfr_batch_size = [32, 128, 1024, 8192]
# train_batch_size = [128, 1024]
# sync_period = [300]
# comms = []
# names = []
# for t, cb in zip(train_steps, cfr_batch_size):
#     for tb in train_batch_size:
#         for sp in sync_period:
#             params = ["--train_steps=" + str(t), "--policy_train_steps=" + str(t),
#                       "--cfr_batch_size=" + str(cb), "--train_batch_size=" + str(tb), "--sync_period=" + str(sp)]
#             param_name = "_".join(params)
#             param_name = param_name.replace("-", "_").replace("=", "_")
#             names.append(param_name)

# ossbcfr
# prefix = "tf_mem_hdge_ossbcfr_runmpi_ossbcfr_hedge.sh"
# train_steps = [2, 16, 128, 1024]
# cfr_batch_size = [100, 1000, 1000]
# train_batch_size = [128, 1024]
# comms = []
# names = []
# index = 0
# for t in train_steps:
#     for cb in cfr_batch_size:
#         cuda_id = ""
#         tb = train_batch_size[0]
#         params = ["--train_steps=" + str(t), "--policy_train_steps=" + str(t),
#                   "--cfr_batch_size=" + str(cb), "--train_batch_size=" + str(tb), "--cuda_id=" + str(cuda_id)]
#         param_name = "_".join(params)
#         param_name = param_name.replace("-", "_").replace("=", "_")
#         names.append(param_name)
#         index += 1

# ossbcfr
# prefix = "mem_ossbcfr_runmpi_ossbcfr.sh"
# use_gpus = True
# cuda_devices = [0, 1, 2, 4]
# train_steps = [16, 128, 1024]
# cfr_batch_size = [1000, 10000]
# train_batch_size = [128, 1024]
# comms = []
# names = []
# index = 0
# for t in train_steps:
#     for cb in cfr_batch_size:
#         for tb in train_batch_size:
#             cuda_id = ""
#             if use_gpus:
#                 cuda_id = str(cuda_devices[index % len(cuda_devices)])
#             params = ["--train_steps=" + str(t), "--policy_train_steps=" + str(t),
#                       "--cfr_batch_size=" + str(cb), "--train_batch_size=" + str(tb), "--cuda_id=" + str(cuda_id)]
#             param_name = "_".join(params)
#             param_name = param_name.replace("-", "_").replace("=", "_")
#             names.append(param_name)
#             index += 1

# file_names = []
# for i, name in enumerate(names):
#     bash_file_name = os.path.join(bash_dir, "_".join(
#         [prefix, name, str(i)]) + ".txt")
#     print(bash_file_name)
#     with open(bash_file_name, "r") as f:
#         while True:
#             line = f.readline()
#             if "Logging directory" in line:
#                 path = line[line.find("results/") + len("results/"):-1]
#                 path = path.replace(":", "_")
#                 file_names.append(path)
#                 break
# fig_name = prefix
# for i in range(len(names)):
#     names[i] = names[i].replace("___", "_").strip("_")

# pre_files = list(zip(file_names, names))
# print(pre_files)

for file, name in ([
    ('16_12_2020.04_18_47.4481', "FHP_ossbcfr_pow_0"),
    ('21_12_2020.06_39_37.11241', "FHP_ossbcfr_pow_1_2"),
    ('21_12_2020.06_39_37.29477', "FHP_ossbcfr_pow_1_8"),
    ('21_12_2020.06_39_37.18314', "FHP_ossbcfr_pow_1_16"),
    ('21_12_2020.06_38_32.2351', "FHP_ossbcfr_pow_1"),
    ('15_10_2020.09_11_33.4151', "FHP_deep_cfr_1000"),
    ('15_10_2020.09_11_34.3173', "FHP_deep_cfr_2000"),
    ('15_10_2020.09_11_34.2626', "FHP_deep_cfr_4000"),
    ('15_10_2020.09_11_34.31081', "FHP_deep_cfr_8000"),
    ('31_12_2020.07_57_31.28640', "FHP_deep_oscfr"),
    ('21_11_2020.14_16_42.5827', "FHP_dncfr"),
    ('25_12_2020.06_38_03.22863', "FHP_nfsp"),
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
    num_step = []
    node_touched = []
    alpha = []
    num_trained = []
    num_sampled = []
    time_relative = []
    value = []
    regrets = {}
    strategies = {}
    N = {}
    s_loss = {}
    file = os.path.join(result_dir, file, 'evaluator-0-mpi-0.jsonl')
    if 'jsonl' in file:
        # print(f_n)
        with open(file) as fp:
            line = fp.readline()
            while line:
                res = re.search('Exploitability', line)
                if res:
                    exps = decoder.decode(line)
                    exploit.append(float(exps['Exploitability']))
                    num_step.append(int(exps['Step']))
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
                            regret_true = info_values['regret_true']
                            policy_true = info_values['policy_true']
                            if inf not in regrets:
                                regrets[inf] = {a: [] for a in actions}
                            if inf not in strategies:
                                strategies[inf] = {a: [] for a in actions}
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
                    except:
                        pass

                line = fp.readline()
                if len(exploit) > length:
                    break
        if save_data:
            if name not in data:
                data[name] = {}
            data[name]['exploit'] = exploit
            data[name]['num_trained'] = num_trained
            data[name]['node_touched'] = node_touched
            data[name]['alpha'] = alpha

        if pindex == 0:
            exp_fig_name = fig_name + name + ".png"
        color_i = colors[pindex]
        if plot_exp:
            print(name, ":", exploit[-1])
            # exploit = moving_average(exploit, n=10)
            exp_axs[0][0].loglog(num_step[0:length],
                                 exploit[0:length], label=name, color=color_i)
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
            i = 0
            keys = list(regrets.keys())
            keys = sorted(keys)
            for inf in keys:
                reg = regrets[inf]
                if i >= ax_num:
                    break
                ax = axs[i // w][i % w]
                for a, v in reg.items():
                    if 'true' in a:
                        ax.plot(np.array(v[0:length]), '--', label=a)
                    else:
                        # v = np.array(v[0:length]) / /
                        #     np.sqrt(1 + np.arange(len(v[0:length])))
                        # v = np.maximum(v, -0.2)
                        # v = np.minimum(v, 0.2)
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
                if i >= ax_num:
                    break
                reg = strategies[inf]
                ax = axs[i // w][i % w]
                for a, v in reg.items():
                    if 'true' in a:
                        ax.plot(v[0:length], '--', label=a)
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
