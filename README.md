# signate-ai-edge-6th

This is Team ReinventTech's repository for [the 6th SIGNATE AI Edge Contest](https://signate.jp/competitions/732/).

The repository contains the followings

- Python project
  - Train models to process LiDAR point clouds for detecting pedestrians and vehicles, which runs on CPU/GPU.
  - Convert trained TensorFlow models to Xilinx-DPU-compatible models
- C/C++ project
  - C/C++ implementation of the detector, which runs on KV260 using a RISC-V core and a DPU core.

Detailed instructions are in "python" and "cpp" directories.
