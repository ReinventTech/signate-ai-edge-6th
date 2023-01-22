cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1
CXX=${CXX:-g++}

$CXX -O2 -I. \
     -I./riscv \
     -o bev -std=c++17 \
     $PWD/src/predictor.cpp \
     $PWD/src/common.cpp  \
     $PWD/src/riscv/riscv_imm.cpp  \
     -mcpu=native \
     -lvart-runner \
     -lglog \
     -lxir \
     -lunilog
