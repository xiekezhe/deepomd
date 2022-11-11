import re
import numpy as np
import argparse
import os
from multiprocessing import Process
from scipy.stats import t

parser = argparse.ArgumentParser(description="run evaluations.")
parser.add_argument("-s", "--num_procs", dest="num_procs", type=int,
                    default=10, help="num threads.")
parser.add_argument("-n", "--num_threads", dest="num_threads", type=int,
                    default=10, help="num threads.")
parser.add_argument("-m", "--batch_size", dest="batch_size", type=int,
                    default=100000, help="batch size.")
parser.add_argument("-p", "--post_process", dest="post",
                    action="store_true", help="post process")
parser.add_argument("-w", "--evaluation", dest="evaluation", type=str,
                    default="deep_cfr", help="which evaluation.")
parser.add_argument("-d", "--post_dir", dest="post_dir", type=str,
                    default="./results/post_evaluate", help="dir for results.")
params = parser.parse_args()


def run(*pars):
    comm = " ".join(pars)
    print(comm)
    os.system(comm)


def post(file_name):
    try:
        with open(file_name, "r") as f:
            lines = f.readlines()
        patt = re.compile("LBR: ([-\d.]+)")
        scores = np.empty(shape=(len(lines),), dtype=np.int32)
        pos = 0
        for line in lines:
            # print(line)
            result = patt.search(line)
            if result:
                scores[pos] = float(result.group(1))
                pos += 1
        scores = scores[:pos]
        return scores
    except:
        return np.array([])


if __name__ == "__main__":
    checkfiles_0 = [
        10, 24, 38, 50, 63, 73,
        84, 98,
        110,
        124,
        134,
        148,
        160,
        174,
        186,
        196,
        209,
        220,
        230,
        240
    ]
    checkfiles_1 = [
        3000, 8000, 12000, 16000, 21000, 26000,
        30000, 35000,
        39000,
        42000,
        46000,
        50000,
        55000,
        59000,
        63000,
        67000,
        71000,
        75000,
        79000,
        83000
    ]
    post_evaluate = "./post_evaluate.sh"
    if params.evaluation == "deep_cfr":
        post_main = "./build/deep_cfr/run_deep_cfr"
        omp_thread = str(params.num_threads)
        batch_size = str(params.batch_size)
        init_strategy_0 = ""
        init_strategy_1 = ""
        graph_def = "deep_cfr"
        post_name = "post_HULH_deep_cfr"
    elif params.evaluation == "ossbcfr":
        post_main = "./build/deep_cfr/run_ossbcfr"
        omp_thread = str(params.num_threads)
        batch_size = str(params.batch_size)
        init_strategy_0 = ""
        init_strategy_1 = ""
        graph_def = "ossbcfr"
        post_name = "post_HULH_ossbcfr"
    else:
        exit()

    for index, (c_0, c_1) in enumerate(zip(checkfiles_0, checkfiles_1)):
        c_00 = "./models/checkpoint-HULH_poker_deep_cfr_policy_0_gpu" + \
            str(c_0)
        c_01 = "./models/checkpoint-HULH_poker_deep_cfr_policy_0_gpu" + \
            str(c_0)
        c_10 = "./models/checkpoint-HULH_poker_ossbcfr_policy_0_gpu" + str(c_1)
        c_11 = "./models/checkpoint-HULH_poker_ossbcfr_policy_1_gpu" + str(c_1)

        post_name_i = post_name
        if params.evaluation == "deep_cfr":
            init_strategy_0 = c_00
            init_strategy_1 = c_01
            post_name_i += "_" + str(c_0)
        else:
            init_strategy_0 = c_10
            init_strategy_1 = c_11
            post_name_i += "_" + str(c_1)

        record = []
        post_names = []
        for i in range(params.num_procs):
            post_name_ii = post_name_i + "_" + str(i)
            post_names.append(post_name_ii)
            if not params.post:
                process = Process(target=run, args=(post_evaluate, post_main, omp_thread, batch_size, init_strategy_0, init_strategy_1,
                                                    graph_def, "> {}/{}.log 2>&1".format(params.post_dir, post_name_ii)))
                process.start()
                record.append(process)

        for p in record:
            p.join()

        post_datas = []
        for i in range(params.num_procs):
            post_data = post(
                "{}/{}.log".format(params.post_dir, post_names[i]))
            post_datas.append(post_data)

        data = np.concatenate(post_datas)
        if not len(data):
            exit()
        mean = data.mean()
        # evaluate sample variance by setting delta degrees of freedom (ddof) to
        # 1. The degree used in calculations is N - ddof
        stddev = data.std(ddof=1)
        # Get the endpoints of the range that contains 95% of the distribution
        t_bounds = t.interval(0.95, len(data) - 1)
        # sum mean to the confidence interval
        diff = [critval * stddev / np.sqrt(len(data)) for critval in t_bounds]
        ci = [mean + critval * stddev /
              np.sqrt(len(data)) for critval in t_bounds]
        print("Mean:", mean, diff,
              "Confidence Interval 95%:", ci, "Std:", stddev, )
