#TODO: derive flash from params
#TODO: derive params and tflite from tflite

test:
	@rm -f out
	../../bin/xtflm_interpreter_cmdline model_mobilenet_v1.tflite model_mobilenet_v1.params baboon.raw out
	@diff out classes.raw
	@rm -f out
	@echo 'PASS: test_smoke'
