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
cmap = plt.get_cmap('tab20')
colors = list(cmap(i) for i in range(20))
hexcolor = list(map(lambda rgb: '#%02x%02x%02x' % (
    int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)), colors))
mpl.rcParams['axes.prop_cycle'] = cycler(color=hexcolor)
# rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# for Palatino and other serif fonts use:
rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)


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


alpha = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
game = "kuhn_poker"
prefix = game + "_sbcfr_olo.sh_game_" + game

names_set = [
    [prefix + "_" + "cfr_FTRL_a_Opponent_rm_{}".format(
        str(alpha[i]).replace("-", "_")) for i in range(len(alpha))],
    [prefix + "_" + "cfr_FTRL_a_LinearOpponent_rm_{}".format(
        str(alpha[i]).replace("-", "_")) for i in range(len(alpha))],
    [prefix + "_" + "cfr_OMD_a_Opponent_rm_{}".format(
        str(alpha[i]).replace("-", "_")) for i in range(len(alpha))],
    [prefix + "_" + "cfr_OMD_a_LinearOpponent_rm_{}".format(
        str(alpha[i]).replace("-", "_")) for i in range(len(alpha))],
    [prefix + "_" + "cfr_PostSbCFR_a_Opponent_w_Constant_rm_{}".format(
        str(alpha[i]).replace("-", "_")) for i in range(len(alpha))],
    [prefix + "_" + "cfr_PostSbCFR_a_Opponent_w_Linear_rm_{}".format(
        str(alpha[i]).replace("-", "_")) for i in range(len(alpha))],
    [prefix + "_" + "cfr_PostSbCFR_a_LinearOpponent_w_Constant_rm_{}".format(
        str(alpha[i]).replace("-", "_")) for i in range(len(alpha))],
    [prefix + "_" + "cfr_PostSbCFR_a_LinearOpponent_w_Linear_rm_{}".format(
        str(alpha[i]).replace("-", "_")) for i in range(len(alpha))],
    [prefix + "_" + "cfr_SbCFRPlus_a_Opponent_w_Constant_rm_{}".format(
        str(alpha[i]).replace("-", "_")) for i in range(len(alpha))],
    [prefix + "_" + "cfr_SbCFRPlus_a_Opponent_w_Linear_rm_{}".format(
        str(alpha[i]).replace("-", "_")) for i in range(len(alpha))],
    [prefix + "_" + "cfr_SbCFRPlus_a_LinearOpponent_w_Constant_rm_{}".format(
        str(alpha[i]).replace("-", "_")) for i in range(len(alpha))],
    [prefix + "_" + "cfr_SbCFRPlus_a_LinearOpponent_w_Linear_rm_{}".format(
        str(alpha[i]).replace("-", "_")) for i in range(len(alpha))],
    [prefix + "_" + "cfr_PSbCFR_a_Opponent_w_Constant_rm_{}".format(
        str(alpha[i]).replace("-", "_")) for i in range(len(alpha))],
    [prefix + "_" + "cfr_PSbCFR_a_Opponent_w_Linear_rm_{}".format(
        str(alpha[i]).replace("-", "_")) for i in range(len(alpha))],
    [prefix + "_" + "cfr_PSbCFR_a_LinearOpponent_w_Constant_rm_{}".format(
        str(alpha[i]).replace("-", "_")) for i in range(len(alpha))],
    [prefix + "_" + "cfr_PSbCFR_a_LinearOpponent_w_Linear_rm_{}".format(
        str(alpha[i]).replace("-", "_")) for i in range(len(alpha))],
    [prefix + "_" + "cfr_PSbCFRPlus_a_Opponent_w_Constant_rm_{}".format(
        str(alpha[i]).replace("-", "_")) for i in range(len(alpha))],
    [prefix + "_" + "cfr_PSbCFRPlus_a_Opponent_w_Linear_rm_{}".format(
        str(alpha[i]).replace("-", "_")) for i in range(len(alpha))],
    [prefix + "_" + "cfr_PSbCFRPlus_a_LinearOpponent_w_Constant_rm_{}".format(
        str(alpha[i]).replace("-", "_")) for i in range(len(alpha))],
    [prefix + "_" + "cfr_PSbCFRPlus_a_LinearOpponent_w_Linear_rm_{}".format(
        str(alpha[i]).replace("-", "_")) for i in range(len(alpha))]]

labels_set = [
    ["${}$".format(str(alpha[i])) for i in range(len(alpha)) if alpha[i] >= 0.01] +
    ["$10^{}$".format("{" + str(int(np.log10(alpha[i]))) + "}") for i in range(len(alpha)) if alpha[i] < 0.01]] * len(names_set)


prefixes = [[game + "_sbcfr_olo.sh"]] * len(names_set)

print(labels_set)
print(names_set)

textwidth = 7.0
linewidth = 3.35

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
    for name, label in zip(names, labels):
        index = np.arange(len(exp_dict[name]['exploit'][1:1000])) * 10
        ax.loglog(index,
                  np.array(exp_dict[name]['exploit'][1:1000]) * 2, label=label)

    ax.set_xlabel('Iterations')
    ax.set_ylabel('Exploitability')
    ax.set_xlim([10, 10000])
    ax.legend()
    fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)

    fig.savefig('./paper/plot' + names_set[ind]
                [0] + '_iter_' + str(ind) + '.pgf')
    fig.savefig('./paper/plot' + names_set[ind]
                [0] + '_iter_' + str(ind) + '.pdf')
