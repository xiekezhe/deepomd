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


labels_set = [["Neural CFR-B", "Deep CFR"]]

names_set = [["Neural CFR-B", "Deep CFR"]]

data_set = [["post_evaluate_ossbcfr_HULH_20.txt",
             "post_evaluate_deep_cfr_HULH_20.txt"]]

textwidth = 7.0
linewidth = 3.35

for ind, (labels, names) in enumerate(zip(labels_set, names_set)):

    latexify(fig_width=1.2 * linewidth)

    # exp
    fig, ax = plt.subplots()
    # fig.set_size_inches(w=linewidth * 1.5, h=linewidth * 1)

    for j, (name, label) in enumerate(zip(names, labels)):
        means = []
        interval = []

        with open("./" + data_set[ind][j], "r") as fp:
            lines = fp.readlines()
        for line in lines:
            result = re.search("Mean: ([-\d.]+) \[([-\d.]+),", line)
            if result:
                means.append(float(result.group(1)))
                interval.append(float(result.group(2)))

        means = np.array(means) * 10
        interval = np.array(interval) * 10
        print(means, interval)
        color = colors[j]
        ax.plot(np.arange(1, len(means) + 1),
                means, color=color, label=label)
        ax.fill_between(np.arange(1, len(means) + 1), means - interval, means +
                        interval, color=color, alpha=0.2)
        ax.set_xlim([0, len(means) + 1])
        ax.set_xticks(np.arange(0, 21, 5))

    ax.set_xlabel('Days')
    ax.set_ylabel('Lower bound on exploitability (mbb/g)')
    ax.legend()
    fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)

    fig.savefig('./paper/plot_lbr_' + str(ind) + '.pgf')
    fig.savefig('./paper/plot_lbr_' + str(ind) + '.pdf')
