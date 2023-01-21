cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1
CXX=${CXX:-g++}

$CXX -O3 -I. \
     -o bev -std=c++17 \
     $PWD/src/predictor.cpp \
     $PWD/src/common.cpp  \
     -mcpu=native \
     -lvart-runner \
     -lglog \
     -lxir \
     -lunilog
