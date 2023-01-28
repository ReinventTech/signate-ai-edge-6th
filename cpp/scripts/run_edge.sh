#!/bin/bash

exec_path=$1
test_meta_path=$2
test_data_dir=$3
result_path=$4
xmodel_path=$5
use_riscv=$6
visualize=$7

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [ "$visualize" = "1" ]; then
    python $SCRIPT_DIR/../../python/scripts/run_edge.py --test-meta-path $test_meta_path --test-data-dir $test_data_dir | $exec_path $result_path $xmodel_path $use_riscv $visualize | (ffmpeg -f rawvideo -pix_fmt yuv444p -s 512x512 -r 3 -i - -f flv -g 1 -b:v 1000k -pix_fmt yuv444p -fflags nobuffer -an -y rtmp://192.168.11.2:1935/live/kv260 2>/dev/null > /dev/null)
else
    python $SCRIPT_DIR/../../python/scripts/run_edge.py --test-meta-path $test_meta_path --test-data-dir $test_data_dir | $exec_path $result_path $xmodel_path $use_riscv $visualize
fi
