#!/bin/bash

CUR_DIR=$(pwd)

OUT_DIR=$1

cd sample_vww
python run.py

cd ..
mkdir $OUT_DIR

clang++ -DTF_LITE_DISABLE_X86_NEON -DTF_LITE_STATIC_MEMORY -DNO_INTERPRETER -Ilib_tflite_micro/submodules/tflite-micro -Ilib_tflite_micro/submodules/flatbuffers/include -I../lib_nn/ -I. -std=c++14 main.cpp sample_vww/tfl_model.tflite.cpp -g -O0 -lxtflitemicro -Ltflite_micro_compiler/build -rpath /Users/deepakpanickal/code/ai_tools2/third_party/lib_tflite_micro/tflite_micro_compiler/build -I$CONDA_PREFIX/include -DTFLMC_PRINT_TENSORS -o $OUT_DIR/tfl.out

$OUT_DIR/tfl.out $OUT_DIR/tfl.out >$OUT_DIR/tflite.json 2>&1

clang++ -DTF_LITE_DISABLE_X86_NEON -DTF_LITE_STATIC_MEMORY -DNO_INTERPRETER -Ilib_tflite_micro/submodules/tflite-micro -Ilib_tflite_micro/submodules/flatbuffers/include -I../lib_nn/ -I. -std=c++14 main.cpp sample_vww/xcore_model.tflite.cpp -g -O0 -lxtflitemicro -Ltflite_micro_compiler/build -rpath /Users/deepakpanickal/code/ai_tools2/third_party/lib_tflite_micro/tflite_micro_compiler/build -I$CONDA_PREFIX/include -DTFLMC_PRINT_TENSORS -o $OUT_DIR/xcore.out

$OUT_DIR/xcore.out sample_vww/xcore_model.params >$OUT_DIR/xcore.json 2>&1

python diff_output.py $OUT_DIR/tflite.json $OUT_DIR/xcore.json >$OUT_DIR/accuracy_diff.txt

cp sample_vww/tfl_model.tflite* $OUT_DIR
cp sample_vww/xcore_model.tflite* $OUT_DIR

exit 0
