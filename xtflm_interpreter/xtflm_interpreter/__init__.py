# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
# XMOS Public License: Version 1
import sys
import ctypes
import numpy as np
from enum import Enum
from typing import Sequence

from typing import Dict, Any
from pathlib import Path

from .base_interpreter import base_interpreter

__PARENT_DIR = Path(__file__).parent.absolute()
if sys.platform.startswith("linux"):
    lib_path = str(__PARENT_DIR / "libs" / "linux" / "xtflm_python.so")
elif sys.platform == "darwin":
    lib_path = str(__PARENT_DIR / "libs" / "macos" / "xtflm_python.dylib")
else:
    raise RuntimeError("libxcore_interpreters is not supported on Windows!")

lib = ctypes.cdll.LoadLibrary(lib_path)

from .exceptions import (
    InterpreterError,
    AllocateTensorsError,
    InvokeError,
    SetTensorError,
    GetTensorError,
    ModelSizeError,
    ArenaSizeError,
    DeviceTimeoutError,
)

# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
# XMOS Public License: Version 1

MAX_TENSOR_ARENA_SIZE = 10000000

class XTFLMInterpreterStatus(Enum):
    OK = 0
    ERROR = 1

class XTFLMInterpreter(base_interpreter):
    def __init__(
        self,
        max_tensor_arena_size=MAX_TENSOR_ARENA_SIZE,
        max_model_size=50000000,
    ) -> None:
        self._error_msg = ctypes.create_string_buffer(4096)

        lib.new_interpreter.restype = ctypes.c_void_p
        lib.new_interpreter.argtypes = [
            ctypes.c_size_t,
        ]

        lib.print_memory_plan.restype = None
        lib.print_memory_plan.argtypes = [ctypes.c_void_p]

        lib.delete_interpreter.restype = None
        lib.delete_interpreter.argtypes = [ctypes.c_void_p]

        lib.initialize.restype = ctypes.c_int
        lib.initialize.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
        ]

        lib.inputs_size.restype = ctypes.c_size_t
        lib.inputs_size.argtypes = [ctypes.c_void_p]

        lib.outputs_size.restype = ctypes.c_size_t
        lib.outputs_size.argtypes = [ctypes.c_void_p]

        lib.set_input_tensor.restype = ctypes.c_int
        lib.set_input_tensor.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_int,
        ]

        lib.get_output_tensor.restype = ctypes.c_int
        lib.get_output_tensor.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_int,
        ]

        lib.get_input_tensor.restype = ctypes.c_int
        lib.get_input_tensor.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_int,
        ]

        lib.get_input_tensor_size.restype = ctypes.c_int
        lib.get_input_tensor_size.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]

        lib.get_output_tensor_size.restype = ctypes.c_int
        lib.get_output_tensor_size.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]

        lib.get_tensor_details_buffer_sizes.restype = ctypes.c_int
        lib.get_tensor_details_buffer_sizes.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        
        lib.get_tensor_details.restype = ctypes.c_int
        lib.get_tensor_details.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int32),
        ]

        lib.input_tensor_index.restype = ctypes.c_size_t
        lib.input_tensor_index.argtypes = [ctypes.c_void_p, ctypes.c_size_t]

        lib.output_tensor_index.restype = ctypes.c_size_t
        lib.output_tensor_index.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        
        lib.invoke.restype = ctypes.c_int
        lib.invoke.argtypes = [ctypes.c_void_p]

        lib.get_error.restype = ctypes.c_size_t
        lib.get_error.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

        lib.arena_used_bytes.restype = ctypes.c_size_t
        lib.arena_used_bytes.argtypes = [
            ctypes.c_void_p,
        ]

        self._max_tensor_arena_size = max_tensor_arena_size
        self._op_states = []

        self.obj = lib.new_interpreter(max_model_size)

        super().__init__()

    def __enter__(self) -> "XTFLMInterpreter":
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self.close()

    def initialise_interpreter(self, engine_num=0):
        currentModel = None
        for model in self.models:
            if model.tile == engine_num:
                currentModel = model

        status = lib.initialize(
            self.obj,
            currentModel.content,
            len(currentModel.content),
            currentModel.params_content,
        )
        if XTFLMInterpreterStatus(status) is XTFLMInterpreterStatus.ERROR:
            raise RuntimeError("Unable to initialize interpreter")

    # def set_input_tensor(self, tensor_index, data, engine_num=0):
    #     if isinstance(data, np.ndarray):
    #         data = data.tobytes()
    #     l = len(data)
    #     l2 = self.get_input_tensor_size(tensor_index, engine_num)
    #     if l != l2:
    #         print('ERROR: mismatching size in set_input_tensor %d vs %d' % (l, l2))

    #     self._check_status(
    #         lib.set_input_tensor(self.obj, tensor_index, data, l)
    #     )

    def get_output_tensor(self, output_index=0, tensor=None, engne_num=0):
        tensor_index = lib.output_tensor_index(self.obj, output_index)
        l = self.get_output_tensor_size(output_index, engine_num)
        if tensor is None:
            tensor_details = self.get_output_details()
            tensor = np.zeros(tensor_details["shape"], dtype=tensor_details["dtype"])
        else:
            l2 = len(tensor.tobytes())
            if l2 != l:
                print('ERROR: mismatching size in get_output_tensor %d vs %d' % (l, l2))
        
        data_ptr = tensor.ctypes.data_as(ctypes.c_void_p)
        self._check_status(
            lib.get_output_tensor(self.obj, output_index, data_ptr, l)
        )

        return tensor

    def get_input_tensor(self, input_index=0):
        tensor_details = self.get_input_details()
        tensor = np.zeros(tensor_details["shape"], dtype=tensor_details["dtype"])
        data_ptr = tensor.ctypes.data_as(ctypes.c_void_p)

        l = len(tensor.tobytes())
        self._check_status(
            lib.get_input_tensor(self.obj, input_index, data_ptr, l)  # TODO: this 0 assumes single output
        )
        return tensor

    def invoke(self):
        INVOKE_CALLBACK_FUNC = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_int)

        self._check_status(lib.invoke(self.obj))

    def close(self) -> None:
        if False and self.obj:
            lib.delete_interpreter(self.obj)
            self.obj = None

    def tensor_arena_size(self):
        return lib.arena_used_bytes(self.obj)

    def _check_status(self, status) -> None:
        if XTFLMInterpreterStatus(status) is XTFLMInterpreterStatus.ERROR:
            lib.get_error(self.obj, self._error_msg)
            raise RuntimeError(self._error_msg.value.decode("utf-8"))

    def print_memory_plan(self):
        lib.print_memory_plan(self.obj)
