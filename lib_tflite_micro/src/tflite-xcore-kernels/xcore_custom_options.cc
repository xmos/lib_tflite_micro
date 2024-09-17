#include "xcore_custom_options.h"

namespace tflite_micro {
namespace ops {
namespace micro {
namespace xcore {

CustomOptionParser::CustomOptionParser(const flexbuffers::Map &map)
    : keys_(flexbuffers::TypedVector::EmptyTypedVector()),
      values_(flexbuffers::Vector::EmptyVector()) {
  keys_ = map.Keys();
  values_ = map.Values();
}

CustomOptionParser::CustomOptionParser(const char *buffer, size_t buffer_length)
    : CustomOptionParser::CustomOptionParser(
          flexbuffers::GetRoot(reinterpret_cast<const uint8_t *>(buffer),
                               buffer_length)
              .AsMap()) {
  assert(buffer != nullptr);
  assert(buffer_length > 0);
}

flexbuffers::Reference
CustomOptionParser::parseNamedCustomOption(const char *name) const {
  for (int i = 0; i < keys_.size(); ++i) {
    const auto &key = keys_[i].AsString().c_str();
    if (strcmp(key, name) == 0) {
      return values_[i];
    }
  }
  return flexbuffers::Reference(nullptr, 1, flexbuffers::NullPackedType());
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite_micro
