#ifndef TFLMCOMPILER_API_H
#define TFLMCOMPILER_API_H

#include <string>

#include "xcore_shared_config.h"

namespace tflmc {

class Compiler;

class TFLMC_Compiler {
 public:
  TFLMC_Compiler(const void *modelData,
                 const struct shared_config::xcore_metadata *sharedCfg,
                 const std::string &prefix = "model_",
                 const bool debugPrint = false);

  ~TFLMC_Compiler();

  void writeSource(std::ostream &out);
  void writeHeader(std::ostream &out);

  // Returns a name that describes a tensors relation to network layers.
  std::string getTensorName(int tensorIndex, int sg) const;

  // Returns tensor arena size
  size_t getTensorArenaSize() const;

  // Returns non-persistent tensor arena size
  size_t TFLMC_Compiler::getNonPersistentTensorArenaSize() const;

  // Returns persistent tensor arena size
  size_t TFLMC_Compiler::getPersistentTensorArenaSize() const;

 private:
  Compiler *compiler_;
};

}  // namespace tflmc

#endif