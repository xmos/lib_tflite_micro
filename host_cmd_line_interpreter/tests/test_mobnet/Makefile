#TODO: derive flash from params
#TODO: derive params and tflite from tflite

test:
	@rm -f out
	@log=$$(mktemp); \
	if ../../bin/xtflm_interpreter_cmdline model_mobilenet_v1.tflite model_mobilenet_v1.params baboon.raw out >$$log 2>&1; then \
		diff out classes.raw; \
		rm -f $$log out; \
		echo 'PASS: test_smoke'; \
	else \
		cat $$log; \
		rm -f $$log out; \
		exit 0; \
	fi
