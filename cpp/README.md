# C++ project for running a model on KV260

## Make predictions on KV260

Setup KV260 with DPU and RISC-V enabled.
Assume that you have transferred the project directory to KV260 (/path/to/project_root/), as well as a partial dataset (/path/to/dataset/3d_labels) necessary for making predictions and evaluating them.

```bash
# Run make_meta.py on your host machine in advance and put generated JSON files under a new directory "/path/to/evaluation".

# On KV260
cd /path/to/project_root/cpp/edge
./build.sh

# For ARM
../scripts/run_edge.sh ./bev /path/to/evaluation/meta_data.json /path/to/dataset/3d_labels /path/to/evaluation/result.json /project_root/python/edge/models/bev.xmodel 0 0
# For RISC-V
sudo ../scripts/run_edge.sh sudo\ ./bev /path/to/evaluation/meta_data.json /path/to/dataset/3d_labels /path/to/evaluation/result.json /project_root/python/edge/models/bev.xmodel 1 0
python ../../python/scripts/evaluate.py --ground-truth-path /path/to/evaluation/ans.json --predictions-path /path/to/evaluation/result.json
```
