#TODO: derive flash from params
#TODO: derive params and tflite from tflite

test:
	@rm -f out0 out1
	../../bin/xtflm_interpreter_cmdline smoke_model.tflite smoke_model.flash -i detection_0.raw -o out0 out1
	@diff out0 classes_gt.raw
	@diff out1 boxes_gt.raw
	@rm -f out0 out1
	@echo 'PASS: test_smoke'
