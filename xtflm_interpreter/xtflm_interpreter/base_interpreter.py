from abc import ABC, abstractmethod

import sys
import struct
import array
import numpy as np

from tflite.Model import  Model
from tflite.TensorType import TensorType

dtypes = {
   0 : 'float32',
   1 : 'float16',
   2 : 'int32',
   3 : 'uint8',
   4 : 'int64',
   5 : 'string',
   6 : 'bool',
   7 : 'int16',
   8 : 'complex64',
   9 : 'int8'
   }

class base_interpreter(ABC):
    def __init__(self):
        self.models = []
        return

    @abstractmethod
    def set_input_tensor(self, input_index, data, engine_num=0):
        return

    @abstractmethod
    def get_output_tensor(self, output_index=0, tensor=None, engine_num=0):
        return

    @abstractmethod
    def get_input_tensor(self, input_index=0, engine_num=0):
        return

    def get_input_tensor_size(self, input_index=0, engine_num=0):
        modelBuf = None
        for model in self.models:
            if model.tile == engine_num:
                modelBuf = Model.GetRootAs(model.content)

        tensorSize = 1
        for i in range(0, modelBuf.Subgraphs(0).Tensors(0).ShapeLength()):
            tensorSize = tensorSize * modelBuf.Subgraphs(0).Tensors(0).Shape(i)
        return tensorSize
        
    def get_output_tensor_size(self, output_index=0, engine_num=0):
        modelBuf = None
        for model in self.models:
            if model.tile == engine_num:
                modelBuf = Model.GetRootAs(model.content)
        tensorsLength = modelBuf.Subgraphs(0).TensorsLength()

        tensorSize = 1
        for i in range(0, modelBuf.Subgraphs(0).Tensors(tensorsLength-1).ShapeLength()):
            tensorSize = tensorSize * modelBuf.Subgraphs(0).Tensors(tensorsLength-1).Shape(i)
        return tensorSize
        return

    @abstractmethod
    def invoke(self):
        return

    def get_input_details(self, engine_num=0):
        modelBuf = None
        for model in self.models:
            if model.tile == engine_num:
                modelBuf = Model.GetRootAs(model.content)

        details = {

          "index": 0,
          "name": str(modelBuf.Subgraphs(0).Tensors(0).Name())[1:].strip("'"),
          "shape": modelBuf.Subgraphs(0).Tensors(0).ShapeAsNumpy(),
          "dtype": dtypes[modelBuf.Subgraphs(0).Tensors(0).Type()],
          "quantization": (modelBuf.Subgraphs(0).Tensors(0).Quantization().Scale(0), modelBuf.Subgraphs(0).Tensors(0).Quantization().ZeroPoint(0))
        }

        return details

    def get_output_details(self, engine_num=0):
        modelBuf = None
        for model in self.models:
            if model.tile == engine_num:
                modelBuf = Model.GetRootAs(model.content)
        tensorIndex = modelBuf.Subgraphs(0).TensorsLength()-1

        details = {
          "index": tensorIndex,
          "name": str(modelBuf.Subgraphs(0).Tensors(tensorIndex).Name())[1:].strip("'"),
          "shape": modelBuf.Subgraphs(0).Tensors(tensorIndex).ShapeAsNumpy(),
          "dtype": dtypes[modelBuf.Subgraphs(0).Tensors(tensorIndex).Type()],
          "quantization": (modelBuf.Subgraphs(0).Tensors(tensorIndex).Quantization().Scale(0), modelBuf.Subgraphs(0).Tensors(tensorIndex).Quantization().ZeroPoint(0))
        }

        return details 

    @abstractmethod
    def close(self, engine_num=0):
        return  

    @abstractmethod
    def tensor_arena_size(self):
        return

    @abstractmethod
    def _check_status(self):
        return

    @abstractmethod
    def print_memory_plan(self):
        return

    def set_model(self, path=None, content=None, params_path=None, params_content=None, engine_num=0):
        if type(path) == str or content is not None:
            tile_found = False
            for model in self.models:
                if model.tile == engine_num:
                    model = self.modelData(path, content, params_path, params_content, engine_num)
                    tile_found = True
                    break
            if not tile_found:
                self.models.append(self.modelData(path, content, params_path, params_content, engine_num))

    class modelData():
        def __init__(self, path, content, params_path, params_content, engine_num):
            self.path = path 
            self.content = content
            self.params_path = params_path
            self.params_content = params_content
            self.tile = engine_num
            self.opList = []
            self.modelToOpList()
            self.pathToContent()

        def modelToOpList(self):

            # Update the path to your model
            if self.path is not None:
                with open(self.path, "rb") as model_file:
                    buffer = model_file.read()
            elif self.content is not None:
                buffer = self.content

            # Get Model
            model = Model.GetRootAs(buffer)
            self.opList = []
            for y in range(0, model.Subgraphs(0).OperatorsLength()):
                opcode = model.OperatorCodes(model.Subgraphs(0).Operators(y).OpcodeIndex())
                if opcode.BuiltinCode() == 32:
                    self.opList.append(str(opcode.CustomCode()).strip("b'"))
                else:
                    self.opList.append(opcode.BuiltinCode())

            f = open('./xtflm_interpreter/schema.fbs', "r")
            lines = f.readlines()[108:238]
            for line in lines:
              if '/' in line:
                lines.remove(line)
            for line in lines:
              if '/' in line:
                lines.remove(line)
            for j in range(len(self.opList)):
                for line in lines:
                    split = line.split(' = ')
                    if str(self.opList[j]) == split[1].strip(',').strip('\n').strip(','):
                        self.opList[j] = split[0].strip()

        def pathToContent(self):
            if self.content == None and self.path != None:
                with open(self.path, "rb") as input_fd:
                    self.content = input_fd.read()

            if self.params_content == None and self.params_path != None:
                with open(self.params_path, "rb") as input_fd2:
                    self.params_content = input_fd2.read()
            else:
                self.params_content = bytes([])
