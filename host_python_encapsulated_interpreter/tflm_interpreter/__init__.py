# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
# XMOS Public License: Version 1
import sys
import ctypes
import numpy as np
from enum import Enum
from typing import Sequence

from typing import Dict, Any
from pathlib import Path

__PARENT_DIR = Path(__file__).parent.absolute()
if sys.platform.startswith("linux"):
    lib_path = str(__PARENT_DIR / "libs" / "linux" / "tflm_python.so")
elif sys.platform == "darwin":
    lib_path = str(__PARENT_DIR / "libs" / "macos" / "tflm_python.dylib")
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

class TFLMInterpreterStatus(Enum):
    OK = 0
    ERROR = 1

class TFLMInterpreter:
    def __init__(
        self,
        model_path=None,
        model_content=None,
        max_tensor_arena_size=MAX_TENSOR_ARENA_SIZE,
    ) -> None:
        self._error_msg = ctypes.create_string_buffer(4096)

        lib.new_interpreter.restype = ctypes.c_void_p
        lib.new_interpreter.argtypes = None

        lib.delete_interpreter.restype = None
        lib.delete_interpreter.argtypes = [ctypes.c_void_p]

        lib.initialize.restype = ctypes.c_int
        lib.initialize.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
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

        lib.invoke.restype = ctypes.c_int
        lib.invoke.argtypes = [ctypes.c_void_p]

        lib.get_error.restype = ctypes.c_size_t
        lib.get_error.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

        lib.arena_used_bytes.restype = ctypes.c_size_t
        lib.arena_used_bytes.argtypes = [
            ctypes.c_void_p,
        ]

        if model_path:
            with open(model_path, "rb") as fd:
                self._model_content = fd.read()
        else:
            self._model_content = model_content

        self._max_tensor_arena_size = max_tensor_arena_size
        self._op_states = []

        self.obj = lib.new_interpreter()
        status = lib.initialize(
            self.obj,
            self._model_content,
            len(self._model_content),
            self._max_tensor_arena_size,
        )
        if TFLMInterpreterStatus(status) is TFLMInterpreterStatus.ERROR:
            raise RuntimeError("Unable to initialize interpreter")

    def __enter__(self) -> "TFLMInterpreter":
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self.close()

    def _check_status(self, status) -> None:
        if TFLMInterpreterStatus(status) is TFLMInterpreterStatus.ERROR:
            lib.get_error(self.obj, self._error_msg)
            raise RuntimeError(self._error_msg.value.decode("utf-8"))

    @property
    def tensor_arena_size(self):
        return lib.arena_used_bytes(self.obj)

    def close(self) -> None:
        if self.obj:
            lib.delete_interpreter(self.obj)
            self.obj = None

    def invoke(self):
        INVOKE_CALLBACK_FUNC = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_int)

        self._check_status(lib.invoke(self.obj))

    def set_input_tensor(self, tensor_index, data):
        if isinstance(data,np.ndarray):
            data = data.tobytes()
        l = len(data)
        l2 = self.get_input_tensor_size(tensor_index)
        if l != l2:
            print('ERROR: mismatching size in set_input_tensor %d vs %d' % (l, l2))

        self._check_status(
            lib.set_input_tensor(self.obj, tensor_index, data, l)
        )

    def get_output_tensor(self, tensor_index, tensor = None):
        l = self.get_output_tensor_size(tensor_index)
        if tensor is None:
            tensor = np.zeros((l), dtype=np.int8)
        else:
            l2 = len(tensor.tobytes())
            if l2 != l:
                print('ERROR: mismatching size in set_input_tensor %d vs %d' % (l, l2))
        
        data_ptr = tensor.ctypes.data_as(ctypes.c_void_p)
        self._check_status(
            lib.get_output_tensor(self.obj, tensor_index, data_ptr, l)
        )

        return tensor

    def get_input_tensor_size(self, tensor_index):
        return lib.get_input_tensor_size(self.obj, tensor_index)

    def get_output_tensor_size(self, tensor_index):
        return lib.get_output_tensor_size(self.obj, tensor_index)

