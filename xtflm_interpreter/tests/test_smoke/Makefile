# TODO: do all in python, use pytest
# TODO: derive smoke_model.flash from smoke_model.params
# TODO: derive smoke_model.params and smoke_model.tflite from tflite
test:
	pyenv local 3.7.12 && \
	. .venv/bin/activate && \
	@rm -f out0 out1
	(export PYTHONPATH=../.. && python smoke_test.py smoke_model.tflite smoke_model.flash detection_0.raw out0 out1)
	@diff out0 classes_gt.raw
	@diff out1 boxes_gt.raw
	@rm -f out0 out1
	@echo 'PASS: test_smoke'
