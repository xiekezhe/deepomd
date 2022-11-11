from math import sqrt
from cycler import cycler
import numpy as np
import re
import json
import os
import fnmatch
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import rc
mpl.use('Agg')
SPINE_COLOR = 'gray'

decoder = json.JSONDecoder()
cmap = plt.get_cmap('tab10')
colors = list(cmap(i) for i in range(20))
hexcolor = list(map(lambda rgb: '#%02x%02x%02x' % (
    int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)), colors))
mpl.rcParams['axes.prop_cycle'] = cycler(color=hexcolor)
# rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# for Palatino and other serif fonts use:
rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)

markers = ["+", "x", "d", "s", "^", "v", "<", ">"] * 2
lines = ["-", "--", "-.", ":"] * 5


def moving_average(a, n=10):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = (ret[n:] - ret[:-n]) / n
    ret[:n] /= np.arange(1, n + 1)
    return ret


def latexify(fig_width=None, fig_height=None, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert(columns in [1, 2])

    if fig_width is None:
        fig_width = 3.39 if columns == 1 else 6.9  # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5) - 1.0) / 2.0    # Aesthetic ratio
        fig_height = fig_width * golden_mean  # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height +
              "so will reduce to" + str(MAX_HEIGHT_INCHES) + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {'backend': 'ps',
              'axes.labelsize': 8,  # fontsize for x and y labels (was 10)
              'axes.titlesize': 8,
              'axes.grid': True,
              'font.size': 8,  # was 10
              'legend.fontsize': 8,  # was 10
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'figure.figsize': [fig_width, fig_height],
              'pgf.rcfonts': False
              }

    mpl.rcParams.update(params)


def format_axes(ax):

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)

    return ax


alpha = [1, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
game = "leduc_poker"
prefix = game + "_sbcfr_olo.sh_game_" + game
prefix_1 = "linear_cfr_sbcfr_olo.sh_game_" + game
prefix_2 = "FTRLCFR_sbcfr_olo.sh_game_" + game


labels_set = [
    ["ReCFR", "PICFR", "CFR", "CFR+"],
    ["FTRL(CFR)", "ReCFR", "FTRL", "CFR"],
    ["OMD(CFR)", "PICFR", "OMD", "CFR+"],
    ["ReCFR", "ReCFR(NA)",
     "ReCFR(LW)", "ReCFR(NALW)", "CFR"],
    ["PICFR", "PICFR(NA)",
     "PICFR(LW)", "PICFR(NALW)", "CFR+"],
    ["PReCFR", "ReCFR", "PCFR", "CFR"],
    ["PPICFR", "PICFR", "PCFR+", "CFR+"],
    ["ReCFR", "PICFR", "LCFR", "FTRL(LA)", "OMD(LA)"], ]

game_rms = dict()
game_rms["leduc_poker"] = {"CFR": "1",
                           "CFR+": "1",
                           "PCFR": "1",
                           "PCFR+": "1",
                           "LCFR": "1",
                           "FTRL(CFR)": "1",
                           "OMD(CFR)": "1",
                           "FTRL": "0.001",
                           "FTRL(LA)": "0.001",
                           "OMD": "0.01",
                           "OMD(LA)": "0.001",
                           "ReCFR": "0.001",
                           "ReCFR(NA)": "0.0001",
                           "ReCFR(LW)": "1e_06",
                           "ReCFR(NALW)": "1e_06",
                           "PICFR": "0.001",
                           "PICFR(NA)": "0.001",
                           "PICFR(LW)": "1e_06",
                           "PICFR(NALW)": "1e_05",
                           "PReCFR": "0.001",
                           "PPICFR": "0.01",
                           }
game_rms["kuhn_poker"] = {"CFR": "1",
                          "CFR+": "1",
                          "PCFR": "1",
                          "PCFR+": "1",
                          "LCFR": "1",
                          "FTRL(CFR)": "1",
                          "OMD(CFR)": "1",
                          "FTRL": "0.01",
                          "FTRL(LA)": "0.01",
                          "OMD": "0.01",
                          "OMD(LA)": "0.01",
                          "ReCFR": "0.01",
                          "ReCFR(NA)": "0.01",
                          "ReCFR(LW)": "0.0001",
                          "ReCFR(NALW)": "1e_05",
                          "PICFR": "0.01",
                          "PICFR(NA)": "0.01",
                          "PICFR(LW)": "0.0001",
                          "PICFR(NALW)": "0.001",
                          "PReCFR": "0.01",
                          "PPICFR": "0.1",
                          }

game_rms["FHP2_poker"] = {"CFR": "1",
                          "CFR+": "1",
                          "PCFR": "1",
                          "PCFR+": "1",
                          "LCFR": "1",
                          "FTRL(CFR)": "1",
                          "OMD(CFR)": "1",
                          "FTRL": "0.0001",
                          "FTRL(LA)": "0.0001",
                          "OMD": "0.001",
                          "OMD(LA)": "0.001",
                          "ReCFR": "1e_06",
                          "ReCFR(NA)": "1e_07",
                          "ReCFR(LW)": "1e_09",
                          "ReCFR(NALW)": "1e_10",
                          "PICFR": "1e_05",
                          "PICFR(NA)": "1e_05",
                          "PICFR(LW)": "1e_07",
                          "PICFR(NALW)": "1e_07",
                          "PReCFR": "1e_07",
                          "PPICFR": "0.0001",
                          }

game_rms["FHP3_poker"] = {"CFR": "1",
                          "CFR+": "1",
                          "PCFR": "1",
                          "PCFR+": "1",
                          "LCFR": "1",
                          "FTRL(CFR)": "1",
                          "OMD(CFR)": "1",
                          "FTRL": "0.0001",
                          "FTRL(LA)": "0.0001",
                          "OMD": "0.001",
                          "OMD(LA)": "0.001",
                          "ReCFR": "1e_06",
                          "ReCFR(NA)": "1e_07",
                          "ReCFR(LW)": "1e_09",
                          "ReCFR(NALW)": "1e_10",
                          "PICFR": "1e_05",
                          "PICFR(NA)": "1e_05",
                          "PICFR(LW)": "1e_07",
                          "PICFR(NALW)": "1e_07",
                          "PReCFR": "1e_07",
                          "PPICFR": "0.0001",
                          }


rms = game_rms[game]

names_set = [
    [prefix + "_" + "cfr_PostSbCFR_a_LinearOpponent_w_Constant",
     prefix + "_" + "cfr_SbCFRPlus_a_LinearOpponent_w_Constant",
     prefix + "_" + "cfr_CFR_a_Opponent_w_Linear",
     prefix + "_" + "cfr_CFRPlus_a_LinearOpponent_w_Linear"],
    [prefix_2 + "_" + "cfr_FTRLCFR_a_Opponent_w_Linear",
        prefix + "_" + "cfr_PostSbCFR_a_LinearOpponent_w_Constant",
        prefix + "_" + "cfr_FTRL_a_Opponent",
        prefix + "_" + "cfr_CFR_a_Opponent_w_Linear"],
    [prefix + "_" + "cfr_OMDCFR_a_LinearOpponent_w_Linear",
        prefix + "_" + "cfr_SbCFRPlus_a_LinearOpponent_w_Constant",
        prefix + "_" + "cfr_OMD_a_Opponent",
        prefix + "_" + "cfr_CFRPlus_a_LinearOpponent_w_Linear", ],
    [prefix + "_" + "cfr_PostSbCFR_a_LinearOpponent_w_Constant",
        prefix + "_" + "cfr_PostSbCFR_a_Opponent_w_Constant",
        prefix + "_" + "cfr_PostSbCFR_a_LinearOpponent_w_Linear",
        prefix + "_" + "cfr_PostSbCFR_a_Opponent_w_Linear",
        prefix + "_" + "cfr_CFR_a_Opponent_w_Linear", ],
    [prefix + "_" + "cfr_SbCFRPlus_a_LinearOpponent_w_Constant",
        prefix + "_" + "cfr_SbCFRPlus_a_Opponent_w_Constant",
        prefix + "_" + "cfr_SbCFRPlus_a_LinearOpponent_w_Linear",
        prefix + "_" + "cfr_SbCFRPlus_a_Opponent_w_Linear",
        prefix + "_" + "cfr_CFRPlus_a_LinearOpponent_w_Linear"],
    [prefix + "_" + "cfr_PSbCFR_a_LinearOpponent_w_Constant",
        prefix + "_" + "cfr_PostSbCFR_a_LinearOpponent_w_Constant",
        prefix + "_" + "cfr_PCFR_a_Opponent_w_Linear",
        prefix + "_" + "cfr_CFR_a_Opponent_w_Linear"],
    [prefix + "_" + "cfr_PSbCFRPlus_a_LinearOpponent_w_Constant",
        prefix + "_" + "cfr_SbCFRPlus_a_LinearOpponent_w_Constant",
        prefix + "_" + "cfr_PCFRPlus_a_LinearOpponent_w_Linear",
        prefix + "_" + "cfr_CFRPlus_a_LinearOpponent_w_Linear"],
    [prefix + "_" + "cfr_PostSbCFR_a_LinearOpponent_w_Constant",
        prefix + "_" + "cfr_SbCFRPlus_a_LinearOpponent_w_Constant",
        prefix_1 + "_" + "cfr_LCFR_a_LinearOpponent_w_Linear",
        prefix + "_" + "cfr_FTRL_a_LinearOpponent",
        prefix + "_" + "cfr_OMD_a_LinearOpponent", ],
]


save_names = [game + "_ave_compare"] * len(names_set)

prefixes = [[game + "_sbcfr_olo.sh",
             "FTRLCFR_sbcfr_olo.sh"]] * len(names_set)

print(labels_set)
print(names_set)

textwidth = 7.0
linewidth = 4.5 / 2

for ind, (labels, names) in enumerate(zip(labels_set, names_set)):

    # if ind < 3:
    #     continue

    latexify(fig_width=1.2 * linewidth)

    exp_dict = dict()
    print(prefixes[ind])
    for prefix in prefixes[ind]:
        with open("./paper/" + prefix + "_data.json", "r") as fp:
            current_dict = json.load(fp)
            exp_dict.update(current_dict)

    fig, ax = plt.subplots()
    for pi, (pname, label) in enumerate(zip(names, labels)):
        name = pname + "_rm_" + rms[label]
        index = np.arange(len(exp_dict[name]['exploit'][1: 201])) * 10
        print(label, ":", exp_dict[name]['exploit']
              [min(len(exp_dict[name]['exploit']) - 1, 201)])
        if label == "CFR":
            ax.semilogy(index, np.array(exp_dict[name]['exploit'][1:201]) * 2, label=label,
                        linestyle=lines[1],
                        markersize=8, markevery=200, linewidth=1, zorder=len(names) + 1 - pi)
        elif label == "CFR+":
            ax.semilogy(index, np.array(exp_dict[name]['exploit'][1:201]) * 2, label=label,
                        linestyle=lines[2],
                        markersize=8, markevery=200, linewidth=1, zorder=len(names) + 1 - pi)
        else:
            ax.semilogy(index, np.array(exp_dict[name]['exploit'][1:201]) * 2, label=label,
                        markersize=8, markevery=200, linewidth=1, zorder=len(names) + 1 - pi)

    ax.set_xlabel('Iterations')
    ax.set_ylabel('Exploitability')
    ax.set_xlim([10, 2000])
    # ax.legend()
    fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)

    fig.savefig('./paper/plot' +
                save_names[ind] + '_iter_' + str(ind) + '.pgf')
    fig.savefig('./paper/plot' +
                save_names[ind] + '_iter_' + str(ind) + '.pdf')
