#!/bin/bash

exec_path=$1
test_meta_path=$2
test_data_dir=$3
result_path=$4
xmodel_path=$5

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

python $SCRIPT_DIR/../../python/scripts/run_edge.py --test-meta-path $test_meta_path --test-data-dir $test_data_dir | $exec_path $result_path $xmodel_path
