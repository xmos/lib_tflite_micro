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
    """! The interpreter base class.
    Defines a common interface that is used by the aisrv and the xtflm interpreter.
    """
    def __init__(self) -> None:
        """! Base interpreter initializer.
        Initialises the list of models attached to the interpreter.
        """
        self.models = []
        return

    @abstractmethod
    def set_input_tensor(self, input_index, data, engine_num=0) -> None:
        """! Abstract for writing the input tensor of a model.
        @param input_index  The index of input tensor to target.
        @param data  The blob of data to set the tensor to.
        @param engine_num The engine to target, for interpreters that support multiple models
        running concurrently. Defaults to 0 for use with a single model.
        """
        return

    @abstractmethod
    def get_output_tensor(self, output_index=0, tensor=None, engine_num=0) -> "Output tensor data":
        """! Abstract for reading the data in the output tensor of a model.
        @param output_index  The index of output tensor to target.
        @param tensor Tensor of correct shape to write into (optional)
        @param engine_num The engine to target, for interpreters that support multiple models
        running concurrently. Defaults to 0 for use with a single model.
        @return The data that was stored in the output tensor.
        """
        return

    @abstractmethod
    def get_input_tensor(self, input_index=0, engine_num=0) -> "Input tensor data":
        """! Abstract for reading the data in the input tensor of a model.
        @param output_index  The index of output tensor to target.
        @param tensor Tensor of correct shape to write into (optional)
        @param engine_num The engine to target, for interpreters that support multiple models
        running concurrently. Defaults to 0 for use with a single model.
        @return The data that was stored in the output tensor.
        """
        return

    @abstractmethod
    def invoke(self) -> None:
        """! Abstract for invoking the model and starting inference of the current
        state of the tensors
        """
        return

    @abstractmethod
    def close(self, engine_num=0) -> None:
        """! Abstract deleting the interpreter
        @params engine_num Defines which interpreter to target in systems with multiple
        """
        return  

    @abstractmethod
    def tensor_arena_size(self) -> "Size of tensor arena":
        """! Abstract to read the size of the tensor arena required
        @return size of the tensor arena as an integer
        """
        return

    @abstractmethod
    def _check_status(self, status):
        """! Abstract to read a status code and raise an exception
        @param status Status code
        """
        return

    @abstractmethod
    def print_memory_plan(self) -> None:
        """! Abstract to print a plan of memory allocation
        """
        return

    def get_input_tensor_size(self, input_index=0, engine_num=0) -> "Size of input tensor":
        """! Read the size of the input tensor from the model
        @param input_index  The index of input tensor to target.
        @param engine_num The engine to target, for interpreters that support multiple models
        running concurrently. Defaults to 0 for use with a single model.
        @return The size of the input tensor as an integer.
        """

        #Select correct model from model list
        modelBuf = None
        for model in self.models:
            if model.tile == engine_num:
                modelBuf = Model.GetRootAs(model.content)

        #Calculate tensor size by multiplying shape elements
        tensorSize = 1
        for i in range(0, modelBuf.Subgraphs(0).Tensors(0).ShapeLength()):
            tensorSize = tensorSize * modelBuf.Subgraphs(0).Tensors(0).Shape(i)
        return tensorSize
        
    def get_output_tensor_size(self, output_index=0, engine_num=0) -> "Size of output tensor":
        """! Read the size of the output tensor from the model
        @param output_index  The index of output tensor to target.
        @param engine_num The engine to target, for interpreters that support multiple models
        running concurrently. Defaults to 0 for use with a single model.
        @return The size of the output tensor as an integer.
        """

        #Select correct model from model list
        modelBuf = None
        for model in self.models:
            if model.tile == engine_num:
                modelBuf = Model.GetRootAs(model.content)

        #Output tensor index is last index
        tensorsLength = modelBuf.Subgraphs(0).TensorsLength()

        #Calculate tensor size by multiplying shape elements
        tensorSize = 1
        for i in range(0, modelBuf.Subgraphs(0).Tensors(tensorsLength-1).ShapeLength()):
            tensorSize = tensorSize * modelBuf.Subgraphs(0).Tensors(tensorsLength-1).Shape(i)
        return tensorSize

    def get_input_details(self, engine_num=0) -> "Details of input tensor":
        """! Reads the input tensor details from the model
        @param engine_num The engine to target, for interpreters that support multiple models
        running concurrently. Defaults to 0 for use with a single model.
        @return Tensor details, including the index, name, shape, data type, and quantization
        parameters.
        """

        #Select correct model from model list
        modelBuf = None
        for model in self.models:
            if model.tile == engine_num:
                modelBuf = Model.GetRootAs(model.content)

        #Generate dictioary of tensor details
        details = {

          "index": 0,
          "name": str(modelBuf.Subgraphs(0).Tensors(0).Name())[1:].strip("'"),
          "shape": modelBuf.Subgraphs(0).Tensors(0).ShapeAsNumpy(),
          "dtype": dtypes[modelBuf.Subgraphs(0).Tensors(0).Type()],
          "quantization": (modelBuf.Subgraphs(0).Tensors(0).Quantization().Scale(0), modelBuf.Subgraphs(0).Tensors(0).Quantization().ZeroPoint(0))
        }

        return details

    def get_output_details(self, engine_num=0) -> "Details of output tensor":
        """! Reads the output tensor details from the model
        @param engine_num The engine to target, for interpreters that support multiple models
        running concurrently. Defaults to 0 for use with a single model.
        @return Tensor details, including the index, name, shape, data type, and quantization
        parameters.
        """

        #Select correct model from models list
        modelBuf = None
        for model in self.models:
            if model.tile == engine_num:
                modelBuf = Model.GetRootAs(model.content)

        #Output tensor is last tensor
        tensorIndex = modelBuf.Subgraphs(0).TensorsLength()-1

        #Generate dictioary of tensor details
        details = {
          "index": tensorIndex,
          "name": str(modelBuf.Subgraphs(0).Tensors(tensorIndex).Name())[1:].strip("'"),
          "shape": modelBuf.Subgraphs(0).Tensors(tensorIndex).ShapeAsNumpy(),
          "dtype": dtypes[modelBuf.Subgraphs(0).Tensors(tensorIndex).Type()],
          "quantization": (modelBuf.Subgraphs(0).Tensors(tensorIndex).Quantization().Scale(0), modelBuf.Subgraphs(0).Tensors(tensorIndex).Quantization().ZeroPoint(0))
        }

        return details 

    def set_model(self, path=None, content=None, params_path=None, params_content=None, engine_num=0) -> None:
        """! Adds a model to the interpreters list of models.
        @param path The path to the model file (.tflite), alternative to content.
        @param content The byte array representing a model, alternative to path.
        @param params_path The path to the params file for the model,
        alternaitve to params_content (optional).
        @param params_content The byte array representing the model parameters,
        alternative to params_path (optional).
        @param engine_num The engine to target, for interpreters that support multiple models
        running concurrently. Defaults to 0 for use with a single model.
        """

        #Check path or content is valid
        if type(path) == str or content is not None:
            tile_found = False
            #Find correct model and replace
            for model in self.models:
                if model.tile == engine_num:
                    model = self.modelData(path, content, params_path, params_content, engine_num)
                    tile_found = True
                    break
            #If model wasn't previously set, add it to list
            if not tile_found:
                self.models.append(self.modelData(path, content, params_path, params_content, engine_num))

    class modelData():
        """! The model data class
        A class that holds a model and data associated with a single model.
        """
        def __init__(self, path, content, params_path, params_content, engine_num):
            """! Model data initializer.
            Sets up variables, generates a list of operators used in the model,
            and reads model and params paths into byte arrays (content).
            @param path Path to the model file (.tflite).
            @param content Model content (byte array).
            @param params_path Path to model parameters file.
            @param params_content Model parameters content (byte array)
            @param engine_num The engine to target, for interpreters that support multiple models
            running concurrently. Defaults to 0 for use with a single model.
            """
            self.path = path 
            self.content = content
            self.params_path = params_path
            self.params_content = params_content
            self.tile = engine_num
            self.opList = []
            self.pathToContent()
            self.modelToOpList()

        def modelToOpList(self):
            """! Generates operator list from model.
            """

            #Load model
            buffer = self.content
            model = Model.GetRootAs(buffer)
            self.opList = []

            #Iterate through operators in model and add operators to opList
            for y in range(0, model.Subgraphs(0).OperatorsLength()):
                opcode = model.OperatorCodes(model.Subgraphs(0).Operators(y).OpcodeIndex())
                #If custom opcode parse string
                if opcode.BuiltinCode() == 32:
                    self.opList.append(str(opcode.CustomCode()).strip("b'"))
                #If built in opcode, decode
                else:
                    self.opList.append(opcode.BuiltinCode())

            #Using schema.fbs, decode custom opcodes
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
            """! Reads model and params paths to content (byte arrays)
            """

            #Check if path exists but not content
            if self.content == None and self.path != None:
                with open(self.path, "rb") as input_fd:
                    self.content = input_fd.read()

            #Check if params_path exits but not params_content
            if self.params_content == None and self.params_path != None:
                with open(self.params_path, "rb") as input_fd2:
                    self.params_content = input_fd2.read()
            #If no params, set to empty byte array
            else:
                self.params_content = bytes([])
