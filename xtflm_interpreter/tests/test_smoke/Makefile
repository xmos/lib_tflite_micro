MODEL_DIR=../../../model
D=${MODEL_DIR}/data
INTERPRETER=python3 smoke_test.py
EXPORT=export PYTHONPATH=../..

test:test_mobilenet_v2 test_ct
	@echo 'PASS: all tests'

test_ct:
	@rm -f out0 out1
	(${EXPORT} && ${INTERPRETER} ${MODEL_DIR}/detection_int8_xcore_1.tflite ${MODEL_DIR}/detection_int8_xcore_1.flash  $D/detection_0.raw  out0 out1)
	@cmp out0 $D/classes_gt.raw
	@cmp out1 $D/boxes_gt.raw
	@echo 'PASS: test ct 1 thread'
	@rm -f out0 out1
	(${EXPORT} && ${INTERPRETER} ${MODEL_DIR}/detection_int8_xcore_5.tflite ${MODEL_DIR}/detection_int8_xcore_5.flash  $D/detection_0.raw  out0 out1)
	@cmp out0 $D/classes_gt.raw
	@cmp out1 $D/boxes_gt.raw
	@rm -f out0 out1
	@echo 'PASS: test ct 5 threads'

test_mobilenet_v2:
	@rm -f out0
	(${EXPORT} && ${INTERPRETER} ${MODEL_DIR}/mobilenet_v2_100_224_int8_xcore_1.tflite ${MODEL_DIR}/mobilenet_v2_100_224_int8_xcore_1.flash  $D/goldfish.raw  out0)
	@cmp out0 $D/goldfish_gt.raw
	@echo 'PASS: mobilenet v2 goldfish 1 thread'
	@rm -f out0
	(${EXPORT} && ${INTERPRETER} ${MODEL_DIR}/mobilenet_v2_100_224_int8_xcore_5.tflite ${MODEL_DIR}/mobilenet_v2_100_224_int8_xcore_5.flash  $D/goldfish.raw  out0)
	@cmp out0 $D/goldfish_gt.raw
	@echo 'PASS: mobilenet v2 goldfish 5 threads'
	@rm -f out0
	(${EXPORT} && ${INTERPRETER} ${MODEL_DIR}/mobilenet_v2_100_224_int8_xcore_1.tflite ${MODEL_DIR}/mobilenet_v2_100_224_int8_xcore_1.flash  $D/ostrich.raw  out0)
	@cmp out0 $D/ostrich_gt.raw
	@echo 'PASS: mobilenet v2 ostrich 1 thread'
	(${EXPORT} && ${INTERPRETER} ${MODEL_DIR}/mobilenet_v2_100_224_int8_xcore_5.tflite ${MODEL_DIR}/mobilenet_v2_100_224_int8_xcore_5.flash  $D/ostrich.raw  out0)
	@cmp out0 $D/ostrich_gt.raw
	@rm -f out0
	@echo 'PASS: mobilenet v2 ostrich 5 threads'
