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

class TfLiteType(Enum):
    # originally defined in tensorflow/tensorflow/lite/c/common.h
    kTfLiteNoType = 0
    kTfLiteFloat32 = 1
    kTfLiteInt32 = 2
    kTfLiteUInt8 = 3
    kTfLiteInt64 = 4
    kTfLiteString = 5
    kTfLiteBool = 6
    kTfLiteInt16 = 7
    kTfLiteComplex64 = 8
    kTfLiteInt8 = 9
    kTfLiteFloat16 = 10
    kTfLiteFloat64 = 11


__TfLiteType_to_numpy_dtype = {
    # TfLiteType.kTfLiteString: None,  # intentionally not supported
    # TfLiteType.kTfLiteNoType: None,  # intentionally not supported
    TfLiteType.kTfLiteFloat64: np.dtype(np.float64),
    TfLiteType.kTfLiteFloat32: np.dtype(np.float32),
    TfLiteType.kTfLiteFloat16: np.dtype(np.float16),
    TfLiteType.kTfLiteComplex64: np.dtype(np.complex64),
    TfLiteType.kTfLiteInt64: np.dtype(np.int64),
    TfLiteType.kTfLiteInt32: np.dtype(np.int32),
    TfLiteType.kTfLiteInt16: np.dtype(np.int16),
    TfLiteType.kTfLiteInt8: np.dtype(np.int8),
    TfLiteType.kTfLiteUInt8: np.dtype(np.uint8),
    TfLiteType.kTfLiteBool: np.dtype(np.bool_),
}
TfLiteType.to_numpy_dtype = lambda self: __TfLiteType_to_numpy_dtype[self]

class XTFLMInterpreterStatus(Enum):
    OK = 0
    ERROR = 1

class XTFLMInterpreter:
    def __init__(
        self,
        model_path=None,
        model_content=None,
        max_tensor_arena_size=MAX_TENSOR_ARENA_SIZE,
        params_path=None,
        params_content=None,
        max_model_size=50000000,
        print_memory_plan=False
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

        if model_path:
            with open(model_path, "rb") as fd:
                self._model_content = fd.read()
        else:
            self._model_content = model_content

        if params_path:
            with open(params_path, "rb") as fd:
                self._params_content = fd.read()
        elif params_content is None:
            self._params_content = bytes([])
        else:
            self._params_content = params_content

        self._max_tensor_arena_size = max_tensor_arena_size
        self._op_states = []

        self.obj = lib.new_interpreter(max_model_size)
        status = lib.initialize(
            self.obj,
            self._model_content,
            len(self._model_content),
            self._max_tensor_arena_size,
            self._params_content,
        )
        if XTFLMInterpreterStatus(status) is XTFLMInterpreterStatus.ERROR:
            raise RuntimeError("Unable to initialize interpreter")
        if print_memory_plan:
            lib.print_memory_plan(self.obj)

    def __enter__(self) -> "XTFLMInterpreter":
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self.close()

    def _check_status(self, status) -> None:
        if XTFLMInterpreterStatus(status) is XTFLMInterpreterStatus.ERROR:
            lib.get_error(self.obj, self._error_msg)
            raise RuntimeError(self._error_msg.value.decode("utf-8"))

    @property
    def tensor_arena_size(self):
        return lib.arena_used_bytes(self.obj)

    def close(self) -> None:
        if False and self.obj:
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

    def get_output_tensor(self, output_index, tensor = None):
        tensor_index = lib.output_tensor_index(self.obj, output_index)
        l = self.get_output_tensor_size(output_index)
        if tensor is None:
            tensor_details = self._get_tensor_details(tensor_index)
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

    def set_tensor(self, tensor_index, data):
        self.set_input_tensor(0, data)

    def get_tensor(self, tensor_index):
        tensor_details = self._get_tensor_details(tensor_index)
        tensor = np.zeros(tensor_details["shape"], dtype=tensor_details["dtype"])
        data_ptr = tensor.ctypes.data_as(ctypes.c_void_p)
        l = len(tensor.tobytes())
        self._check_status(
            lib.get_output_tensor(self.obj, 0, data_ptr, l)  # TODO: this 0 assumes single output
        )
        return tensor

    def get_input_tensor_size(self, tensor_index):
        return lib.get_input_tensor_size(self.obj, tensor_index)

    def get_output_tensor_size(self, tensor_index):
        return lib.get_output_tensor_size(self.obj, tensor_index)

    def allocate_tensors(self):
        pass

    def _get_tensor_details(self, tensor_index: int):
        # first get the dimensions of the tensor
        shape_size = ctypes.c_size_t()
        scale_size = ctypes.c_size_t()
        zero_point_size = ctypes.c_size_t()

        self._check_status(
            lib.get_tensor_details_buffer_sizes(
                self.obj,
                tensor_index,
                ctypes.byref(shape_size),
                ctypes.byref(scale_size),
                ctypes.byref(zero_point_size),
            )
        )

        # allocate buffer for shape
        tensor_shape = (ctypes.c_int * shape_size.value)()
        tensor_name_max_len = 1024
        tensor_name = ctypes.create_string_buffer(tensor_name_max_len)
        tensor_type = ctypes.c_int()
        tensor_scale = (ctypes.c_float * scale_size.value)()
        tensor_zero_point = (ctypes.c_int32 * zero_point_size.value)()

        self._check_status(
            lib.get_tensor_details(
                self.obj,
                tensor_index,
                tensor_name,
                tensor_name_max_len,
                tensor_shape,
                ctypes.byref(tensor_type),
                tensor_scale,
                tensor_zero_point,
            )
        )
        scales = np.array(tensor_scale, dtype=np.float32)
        if len(tensor_scale) == 1:
            scales = scales[0]

        zero_points = np.array(tensor_zero_point, dtype=np.int32)
        if len(tensor_scale) == 1:
            zero_points = zero_points[0]

        return {
            "index": tensor_index,
            "name": tensor_name.value.decode("utf-8"),
            "shape": np.array(tensor_shape, dtype=np.int32),
            "dtype": TfLiteType(tensor_type.value).to_numpy_dtype(),
            "quantization": (scales, zero_points),
        }

    def get_input_details(self):
        inputs_size = lib.inputs_size(self.obj)
        input_indices = [
            lib.input_tensor_index(self.obj, input_index)
            for input_index in range(inputs_size)
        ]

        return [self._get_tensor_details(idx) for idx in input_indices]

    def get_output_details(self):
        outputs_size = lib.outputs_size(self.obj)
        output_indices = [
            lib.output_tensor_index(self.obj, output_index)
            for output_index in range(outputs_size)
        ]

        return [self._get_tensor_details(idx) for idx in output_indices]
