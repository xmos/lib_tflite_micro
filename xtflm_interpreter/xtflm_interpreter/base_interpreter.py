from abc import ABC, abstractmethod

import sys
import struct
import array
import numpy as np

from tflite.Model import Model

class base_interpreter(ABC):
    def __init__(self):
        self.models = []
        return

    @abstractmethod
    def set_input_tensor(self, input_index=0, data, engine_num=0):
        return

    @abstractmethod
    def get_output_tensor(self, output_index=0, tensor=None, engine_num=0):
        return

    @abstractmethod
    def get_input_tensor(self, input_index=0, engine_num=0):
        return

    @abstractmethod
    def get_input_size(self, input_index=0, engine_num=0):
        return
        
    @abstractmethod
    def get_output_size(self, output_index=0, engine_num=0):
        return  

    @abstractmethod
    def invoke(self):
        return

    @abstractmethod
    def get_input_details(self, engine_num=0):
        return  

    @abstractmethod
    def get_output_details(self, engine_num=0):
        return  

    @abstractmethod
    def close(self, engine_num=0):
        return  

    @abstractmethod
    def tensor_arena_size(self):
        return

    @abstractmethod
    def check_status(self):
        return

    @abstractmethod
    def print_memory_plan(self):
        return

    def set_model(self, path=None, content=None, engine_num=0):
        if type(path) == str or content is not None:
            tile_found = False
            for model in self.models:
                if model.tile == engine_num:
                    model.path = path
                    tile_found = True
                    break
            if not tile_found:
                self.models.append(self.modelData(path, content, engine_num))

    class modelData():
        def __init__(self, path, content, engine_num):
            self.path = path 
            self.content = content
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

            f = open('../schema.fbs', "r")
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
