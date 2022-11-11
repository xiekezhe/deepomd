import skopt
import argparse
import os
import stat
import time
import neptune
import neptunecontrib.monitoring.skopt as sk_utils

os.environ["NEPTUNE_API_TOKEN"] = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiYjZiN2YyZjgtNTA1ZS00MWE3LTg3NTYtYmFmN2U0ZDZmM2YzIn0="
neptune.init('luwemng/sandbox')
neptune.create_experiment('my_spiel_cpu', upload_source_files=['*.py'])
index = 0


def run_comm(comm, name, prefix, bash_dir, dry_run):
    bash_file_name = os.path.join(bash_dir, "_".join(
        [prefix, name]) + ".sh")
    with open(bash_file_name, "w") as f:
        f.write(comm)
    os.chmod(bash_file_name, os.stat(
        bash_file_name).st_mode | stat.S_IEXEC)
    bash_comm = " ".join(["bash -c", bash_file_name, ">",
                          os.path.join(bash_dir, "_".join([prefix, name, ".txt 2>&1"]))])
    print(bash_comm)

    start = time.time()
    if not dry_run:
        os.system(bash_comm)
    end = time.time()
    return end - start


SPACE = [
    skopt.space.Integer(1, 512, prior="log-uniform", base=2, name="actors"),
    skopt.space.Integer(1, 2, name="inference_threads"),
    skopt.space.Integer(1, 8, name="num_cpus"),
    # skopt.space.Integer(0, 1, name="num_gpus")
]


def hyper_tune(base_comm, prefix, bash_dir, dry_run):
    @skopt.utils.use_named_args(SPACE)
    def objective(**params):
        global index
        param = ["--omp_threads=" + str(10), "--actors=" + str(params["actors"]),
                 "--inference_threads=" +
                 str(params["inference_threads"]), "--inference_batch_size=" +
                 str(params["actors"]),
                 "--num_cpus=" + str(params["num_cpus"]), "--num_gpus=" + str(0), "--suffix=" + str(index)]
        comm = " ".join([base_comm] + param)
        param_name = "_".join(param)
        param_name = param_name.replace("-", "_").replace("=", "_")
        index += 1
        return run_comm(comm, param_name, prefix, bash_dir, dry_run)

    monitor = sk_utils.NeptuneMonitor()
    results = skopt.gp_minimize(
        objective, SPACE, n_calls=100, n_random_starts=20, callback=[monitor])
    best_auc = results.fun
    best_params = results.x
    sk_utils.log_results(results)

    print('best result: ', best_auc)
    print('best parameters: ', best_params)
    neptune.stop()


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("bash", default="runmpi.sh", type=str,
                    help="bash file to process.")
parser.add_argument("-p", "--prefix", default="cfr", type=str,
                    help="prefix for the name of output files.")
parser.add_argument("-n", "--dry_run",
                    action="store_true", help="dry run.")
parser.add_argument("--bash_dir", default="./results/hyper_tune_torch_gpu",
                    type=str, help="dir to store bash files.")

args = parser.parse_args()

if __name__ == "__main__":
    bash_file = args.bash
    with open(bash_file, "r") as f:
        bash_lines = f.read().strip()
    os.makedirs(args.bash_dir, exist_ok=True)
    prefix = "_".join([args.prefix, args.bash])
    hyper_tune(bash_lines, prefix, args.bash_dir, args.dry_run)
