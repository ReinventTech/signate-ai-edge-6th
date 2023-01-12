# Python project for training models and running them on KV260

## Setup

```bash
# For pip users
pip install -r requirements.txt

# For pipenv users
pipenv install
pipenv shell
```

## Prepare a dataset for 6th SIGNATE AI Edge Contest

1. Download the contest data files from the contest page and extract them under /path/to/signate_data, merging "train\*" into "train" directory.

2. Create a dataset from the downloaded LiDAR data and labels.

```bash
# The dataset will be created under /path/to/signate_data/train/3d_labels/signate
python src/dataset.py create --root /path/to/signate_data/train/3d_labels
```

## Train a Bird's-Eye-View model

```bash
python src/train.py --dataset /path/to/signate_data/train/3d_labels

# Run a TensorBoard if necessary
tensorboard --logdir tb_logs
```

## Convert a training checkpoint to a TensorFlow saved model

```bash
# For one checkpoint
python src/convert_bev_to_saved_model.py --ckpt checkpoints/bev.ckpt --output /path/to/bev_saved_model

# For two checkpoints (pedestrian/vehicle)
python src/merge_and_convert_bev_to_saved_model.py --pckpt pedestrian_checkpoints/bev.ckpt --vckpt vehicle_checkpoints/bev.ckpt --output /path/to/bev_saved_model

# Update the saved model if necessary
cp -r /path/to/bev_saved_model models/bev_model
```

## Prepare a Point-Painting-like model (Optional)

```bash
git clone https://github.com/google-research/deeplab2.git
cd deeplab2
git clone https://github.com/tensorflow/models.git
export PYTHONPATH=$PYTHONPATH:`pwd`
export PYTHONPATH=$PYTHONPATH:`pwd`/models
# Choose another model/checkpoint/config if necessary
python export_model.py --experiment_option_path=configs/cityscapes/axial_deeplab/max_deeplab_s_backbone_os16.textproto --checkpoint_path=max_deeplab_s_backbone_os16_axial_deeplab_cityscapes_trainfine/ckpt-60000 --output_path=/path/to/pp_model

# Update the model if necessary
cp -r /path/to/pp_model /project_root/python/models/pp_model
```

## Make predictions for the contest

```bash
mkdir tmp
# Modify the scenes list in make_meta.py if necessary
python scripts/make_meta.py --data-dir /path/to/signate_data/train/3d_labels --output-path tmp
python scripts/run.py --exec-path src --test-meta-path tmp/meta_data.json --test-data-dir /path/to/signate_data/train/3d_labels --result-path tmp/result.json 2>/dev/null
python scripts/evaluate.py --ground-truth-path tmp/ans.json --predictions-path tmp/result.json
```

## Quantize/Compie a trained model for KV260

Setup a docker environment for Vitis AI, following the instruction [here](https://docs.xilinx.com/r/en-US/ug1414-vitis-ai/Getting-Started)

```bash
cd /path/to/Vitis-AI
cp -r /project_root/python/vitis /path/to/Vitis-AI/
cp -r /path/to/signate_data/train /path/to/Vitis-AI/vitis/
./docker_run.sh xilinx/vitis-ai-cpu:latest
# or "./docker_run.sh xilinx/vitis-ai-gpu:latest" if you have a GPU with nvidia-docker installed

# Now you should be inside a docker container
cd vitis/src
pip install tensorflow-gpu==2.8 numpy-quaternion tensorflow-addons
python kv260_convert_bev.py --dataset ../train/3d_labels --pckpt pedestrian_checkpoints/bev.ckpt --vckpt vehicle_checkpoints/bev.ckpt

# replace a path to arch.json if you have a custom DPU architecture
vai_c_tensorflow2 -m ./quantized_model.h5 -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json -o ./compiled -n bev
# bev.xmodel should be generated under /path/to/Vitis-AI/vitis/src/compiled directory
```

## Make predictions on KV260

Setup KV260 with DPU enabled.
Assume that you have transferred the project directory to KV260 (/path/to/project_root/), as well as a partial dataset necessary for making predictions and evaluating them.

```bash
# On KV260
cd /path/to/project_root/python/edge
mkdir tmp
# Run make_meta.py on your host machine in advance and put generated JSON files under the tmp directory.
python scripts/run.py --exec-path src --test-meta-path tmp/meta_data.json --test-data-dir /path/to/dataset/3d_labels --result-path tmp/result.json
python scripts/evaluate.py --ground-truth-path tmp/ans.json --predictions-path tmp/result.json
```
