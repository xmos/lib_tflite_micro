TARGET = XCORE-AI-EXPLORER
APP_NAME = app_inference_engine

APP_SHARED_FLAGS  = -report \
-O3 -fcomment-asm -mcmodel=large -Wno-xcore-fptrgroup -save-temps \
-Wno-unknown-pragmas \
-Wno-bidirectional-buffered-port \
-DTF_LITE_STATIC_MEMORY \
-DXCORE \
-DPRINT_PROFILER_SUMMARY \
-lquadflash \
-DTF_LITE_STRIP_ERROR_STRINGS

# The USED_MODULES variable lists other module used by the application. 
USED_MODULES = lib_nn lib_tflite_micro

XCC_FLAGS = $(APP_SHARED_FLAGS)
XCC_CPP_FLAGS = $(APP_SHARED_FLAGS) -std=c++11  

#=============================================================================
# The following part of the Makefile includes the common build infrastructure
# for compiling XMOS applications. You should not need to edit below here.

XMOS_MAKE_PATH ?= ../..
include $(XMOS_MAKE_PATH)/xcommon/module_xcommon/build/Makefile.common

