#!/bin/bash

python build_mobilenetv2_alpha_1.py

# Use grep to extract the XTC version
xtc_version=$(echo $(xcc --version) | grep -o 'XTC version: [0-9.]*' | cut -d' ' -f3)

# Set the XMOS_XTC_DIR environment variable
export XMOS_XTC_DIR="/Applications/XMOS_XTC_$xtc_version"

# Run the command in the background
$XMOS_XTC_DIR/SetEnv.command &
pid=$!

# Wait for the command to complete
wait $pid
 
yes Y | xflash --data xcore_flash_binary.out --target XCORE-AI-EXPLORER

xmake

xrun --xscope bin/app_mobilenetv2_alpha_1.xe 
