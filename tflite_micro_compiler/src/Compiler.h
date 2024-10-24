#ifndef TFLMCOMPILER_COMPILER_H
#define TFLMCOMPILER_COMPILER_H

#include <iostream>
#include <sstream>
#include <unordered_map>

#include "MemMap.h"
#include "llvm/ADT/StringMap.h"
#include "python/tflite_micro/python_ops_resolver.h"
#define private public
#include "tensorflow/lite/micro/micro_interpreter.h"
#undef private
#include "tensorflow/lite/schema/schema_generated.h"
#include "xcore_ops.h"
#include "xcore_shared_config.h"

namespace tflmc {

struct Allocation {
  ptrdiff_t offset;
  size_t len;
  int nodeIndex;
};

TfLiteTensor *GetTensor(tflite_micro::MicroInterpreter *interpreter, int i, int sg);
TfLiteEvalTensor *GetEvalTensor(tflite_micro::MicroInterpreter *interpreter, int i,
                                int sg);

bool CompileFile(const std::string &modelFileName,
                 const std::string &outFileName,
                 const std::string &prefix = "model_");

class Compiler {
 public:
  // modelData: Flatbuffer binary data.
  // prefix: This string is prepended to every global name.
  Compiler(const void *modelData, const std::string &prefix = "model_",
           const bool debugPrint = false);

  Compiler(const void *modelData,
           const struct shared_config::xcore_metadata *sharedCfg,
           const std::string &prefix = "model_", const bool debugPrint = false);

  void writeSource(std::ostream &out);
  void writeHeader(std::ostream &out);

  // Returns a name that describes a tensors relation to network layers.
  std::string getTensorName(int tensorIndex, int sg) const;

  // Returns tensor arena size
  size_t getTensorArenaSize() const { return persistentArenaSize_ + nonPersistentArenaSize_; }

  // Returns persistent arena size
  size_t getPersistentArenaSize() const { return persistentArenaSize_; }

  // Returns non-persistent arena size
  size_t getNonPersistentArenaSize() const { return nonPersistentArenaSize_; }

 private:
  bool init(const void *modelData);

  void deDuplicateData();

 private:
  struct TensorInfo {
    TensorInfo(const TfLiteTensor *tensor_ptr) : tensor(tensor_ptr) {}
    const TfLiteTensor *tensor = nullptr;
  };
  struct RegistrationInfo {
    const TFLMRegistration *reg = nullptr;
    tflite_micro::BuiltinOperator code;
    std::string custom_name;
    bool operator==(const RegistrationInfo &other) {
      if (code != other.code) return false;
      if (code == tflite_micro::BuiltinOperator_CUSTOM) {
        return custom_name == other.custom_name;
      } else
        return true;
    }
  };
  struct NodeInfo {
    NodeInfo() {}
    NodeInfo(TfLiteNode tfl_node, ptrdiff_t reg_index)
        : node(tfl_node), regIndex(reg_index) {}
    TfLiteNode node;
    ptrdiff_t regIndex = -1;
  };
  template <class T>
  struct Option {
    bool None = true;
    T Some = T();
    void operator=(T const &val) {
      None = false;
      Some = val;
    }
    void clear() {
      Some = T();
      None = true;
    }
  };

 private:
  std::string prefix_;
  const struct shared_config::xcore_metadata *sharedCfg_ = nullptr;
  int numXCThreads_ = 1;
  const tflite_micro::Model *model_ = nullptr;
  const tflite_micro::SubGraph *mainGraph_ = nullptr;
  tflite::PythonOpsResolver resolver_;
  std::vector<uint8_t> arena_buf_;
  std::unique_ptr<tflite_micro::MicroInterpreter> interpreter_;
  MemMap memMap_;

  size_t persistentArenaSize_ = 0;
  size_t nonPersistentArenaSize_ = 0;
  size_t varTensors_count = 0;
  // Vector of vector is for subgraphs
  std::vector<std::vector<TensorInfo>> tensors_;
  std::vector<std::vector<NodeInfo>> nodes_;
  std::vector<std::vector<int32_t>> inputTensorIndices_;
  std::vector<std::vector<int32_t>> outputTensorIndices_;
  std::vector<RegistrationInfo> registrations_;
  std::vector<int32_t> scratchBufferOffsets_;

  std::vector<llvm::StringMap<int>> opdataHashMap_;
  std::vector<std::unordered_map<int, int>> opdataMap_;

  std::vector<llvm::StringMap<int>> tensorDimHashMap_;
  std::vector<std::unordered_map<int, int>> tensorDimMap_;

  std::vector<llvm::StringMap<int>> quantHashMap_;
  std::vector<std::unordered_map<int, int>> quantMap_;

  bool has_custom_ops = false;
  bool has_xc_conv_ops = false;
  bool has_tflite_custom_ops = false;
  bool has_quantization = false;
};

}  // namespace tflmc

#endif
