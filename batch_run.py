import argparse
import os
import stat


def batch_comms(base_comm):
    use_gpus = False
    cuda_devices = [0, 1, 2]
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
    # cfr_rm_scale = [1]
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
                        params = ["--game=" + game, "--cfr_batch_size=" + str(bs),
                                  "--cfr_rm_scale=" + str(cr)]
                        run_comm = base_comm.replace("placeholder", run)
                        comms.append(" ".join([run_comm] + params))
                        param_name = "_".join(params)
                        param_name = param_name.replace(
                            "-", "_").replace("=", "_")
                        names.append(param_name)
                        index += 1

    # run = "run_online_learning"
    # average_type = ["Opponent", "LinearOpponent"]
    # cfr_mode = ["FTRL", "OMD"]
    # for cm in cfr_mode:
    #     for cr in cfr_rm_scale:
    #         for at in average_type:
    #             cuda_id = ""
    #             if use_gpus:
    #                 cuda_id = str(cuda_devices[index % len(cuda_devices)])
    #             cb = cfr_batch_size[0]
    #             tb = train_batch_size[0]
    #             params = ["--game=" + game, "--cfr_mode=" + str(cm), "--average_type=" + str(
    #                 at), "--cfr_rm_scale=" + str(cr)]
    #             run_comm = base_comm.replace("placeholder", run)
    #             comms.append(" ".join([run_comm] + params))
    #             param_name = "_".join(params)
    #             param_name = param_name.replace("-", "_").replace("=", "_")
    #             names.append(param_name)
    #             index += 1

    # algos = ["run_online_learning"]
    # average_type = ["Opponent"]
    # weight_type = ["Linear"] * len(average_type)
    # cfr_mode = ["FTRLCFR"]

    # for game in games:
    #     for cfr_i in range(len(cfr_mode)):
    #         cuda_id = ""
    #         if use_gpus:
    #             cuda_id = str(cuda_devices[index % len(cuda_devices)])
    #         cb = cfr_batch_size[0]
    #         tb = train_batch_size[0]
    #         algo = algos[cfr_i]
    #         cm = cfr_mode[cfr_i]
    #         ave = average_type[cfr_i]
    #         wt = weight_type[cfr_i]
    #         params = ["--game=" + game, "--cfr_mode=" +
    #                   str(cm), "--average_type=" + str(ave), "--weight_type=" + str(wt), "--cfr_rm_scale=1"]
    #         comm = base_comm.replace("placeholder", algo)
    #         comms.append(" ".join([comm] + params))
    #         param_name = "_".join(params)
    #         param_name = param_name.replace("-", "_").replace("=", "_")
    #         names.append(param_name)
    #         index += 1
    return comms, names


def run_comms(comms, names, prefix, bash_dir):
    ret = []
    for i, (comm, name) in enumerate(zip(comms, names)):
        bash_file_name = os.path.join(bash_dir, "_".join(
            [prefix, name]) + ".sh")
        with open(bash_file_name, "w") as f:
            f.write(comm)
        os.chmod(bash_file_name, os.stat(
            bash_file_name).st_mode | stat.S_IEXEC)
        ret.append(" ".join(["nohup", "bash -c", bash_file_name, ">",
                             os.path.join(bash_dir, "_".join([prefix, name + ".txt"])), "&"]))
    return ret


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("bash", default="runmpi.sh", type=str,
                    help="bash file to process.")
parser.add_argument("-p", "--prefix", default="cfr", type=str,
                    help="prefix for the name of output files.")
parser.add_argument("-n", "--dry_run",
                    action="store_true", help="dry run.")
parser.add_argument("--bash_dir", default="./results/batch_run",
                    type=str, help="dir to store bash files.")

args = parser.parse_args()

if __name__ == "__main__":
    bash_file = args.bash
    with open(bash_file, "r") as f:
        bash_lines = f.read().strip()
    os.makedirs(args.bash_dir, exist_ok=True)
    comms, names = batch_comms(bash_lines)
    prefix = "_".join([args.prefix, args.bash])
    comm_lines = run_comms(comms, names, prefix, args.bash_dir)
    for comm in comm_lines:
        print(comm)
    if not args.dry_run:
        for comm in comm_lines:
            os.system(comm)
