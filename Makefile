build:
	(cd xtflm_interpreter && make install)

clean:
	(cd xtflm_interpreter && make clean)

init:
	python3 fetch_dependencies.py
	python3 -m venv .venv && \
	. .venv/bin/activate && \
	pip3 install --upgrade pip && \
	pip3 install -v -v -v -r requirements.txt && \
	make models

models:
	xcore-opt -o=model/mobilenet_v1_0.25_xcore.tflite --xcore-tfl-pipeline --xcore-thread-count=1 \
      model/mobilenet_v1_0.25.tflite
#      --xcore-flash-image-file=model/mobilenet_v1_0.25_xcore.flash 
	xcore-opt -o=model/detection_int8_xcore_1.tflite --xcore-tfl-pipeline --xcore-thread-count=1 \
      --xcore-flash-image-file=model/detection_int8_xcore_1.flash  \
      model/detection_int8.tflite
	xcore-opt -o=model/detection_int8_xcore_5.tflite --xcore-tfl-pipeline --xcore-thread-count=5 \
      --xcore-flash-image-file=model/detection_int8_xcore_5.flash  \
      model/detection_int8.tflite
	xcore-opt -o=model/mobilenet_v2_100_224_int8_xcore_1.tflite --xcore-tfl-pipeline --xcore-thread-count=1 \
      --xcore-flash-image-file=model/mobilenet_v2_100_224_int8_xcore_1.flash  \
      model/mobilenet_v2_100_224_int8.tflite
	xcore-opt -o=model/mobilenet_v2_100_224_int8_xcore_5.tflite --xcore-tfl-pipeline --xcore-thread-count=5 \
      --xcore-flash-image-file=model/mobilenet_v2_100_224_int8_xcore_5.flash  \
      model/mobilenet_v2_100_224_int8.tflite

test:
	(. .venv/bin/activate && cd host_cmd_line_interpreter && make test)
	(. .venv/bin/activate && cd xtflm_interpreter && make test)
	@echo ""
	@echo "All tests PASS"
	@echo ""
