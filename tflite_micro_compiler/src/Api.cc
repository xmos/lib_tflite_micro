#include "Api.h"

#include "Compiler.h"

namespace tflmc {

TFLMC_Compiler::TFLMC_Compiler(
    const void *modelData,
    const struct shared_config::xcore_metadata *sharedCfg,
    const std::string &prefix, const bool debugPrint) {
  compiler_ = new Compiler(modelData, sharedCfg, prefix, debugPrint);
}

TFLMC_Compiler::~TFLMC_Compiler() { delete compiler_; }

void TFLMC_Compiler::writeSource(std::ostream &out) {
  compiler_->writeSource(out);
}
void TFLMC_Compiler::writeHeader(std::ostream &out) {
  compiler_->writeHeader(out);
}

// Returns a name that describes a tensors relation to network layers.
std::string TFLMC_Compiler::getTensorName(int tensorIndex, int sg) const {
  return compiler_->getTensorName(tensorIndex, sg);
}

// Returns tensor arena size (persistent + non-persistent)
size_t TFLMC_Compiler::getTensorArenaSize() const {
  return compiler_->getTensorArenaSize();
}

// Returns non-persistent arena size
size_t TFLMC_Compiler::getNonPersistentArenaSize() const {
  return compiler_->getNonPersistentArenaSize();
}

// Returns persistent arena size
size_t TFLMC_Compiler::getPersistentArenaSize() const {
  return compiler_->getPersistentArenaSize();
}

}  // namespace tflmc