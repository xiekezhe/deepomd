cd /data/liuwm
rm -rf plot_results
mkdir -p plot_results
rsync -avm --include='evaluator-0-mpi-0.jsonl'  -f 'hide,! */' ./my_spiel/results ./plot_results