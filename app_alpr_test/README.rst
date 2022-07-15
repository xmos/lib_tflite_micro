Requirements:

Clone the following repos, side by side:

  * lib_nn           git@github.com:xmos/lib_nn.git
    
  * lib_tflite_micro git@github.com:xmos/lib_tflite_micro.git

    Inside this repo: git submodule update --init --recursive
    
  * lib_i2c          git@github.com:xmos/lib_i2c.git
    
  * lib_uart         git@github.com:xmos/lib_uart.git
    
  * lib_mipi         git@github.com:xmos/lib_mipi.git
    
  * lib_gpio         git@github.com:xmos/lib_gpio.git
    
  * lib_logging      git@github.com:xmos/lib_logging.git
    
  * lib_xassert      git@github.com:xmos/lib_xassert.git


Then::
  
  cd lib_tflite_micro/app_alpr_test
  xmake
