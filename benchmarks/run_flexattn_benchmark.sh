mkdir -p results
python flexattn_benchmark.py --skip_correctness
mv flexattn_benchmark.csv results/
# log the gpu system configuration: cuda version, cudnn version, torch version, torch cuda version, torch mps version, ...
nvcc --version >> results/system_config.txt
cat /usr/local/cuda/version.txt >> results/system_config.txt
cat /usr/local/cudnn/version.txt >> results/system_config.txt
python -m torch.utils.collect_env >> results/system_config.txt
