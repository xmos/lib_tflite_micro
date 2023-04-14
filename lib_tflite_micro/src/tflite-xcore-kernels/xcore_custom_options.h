#ifndef XCORE_CUSTOM_OPTIONS_H_
#define XCORE_CUSTOM_OPTIONS_H_

#include "flatbuffers/flexbuffers.h"
#include "xcore_ops.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

class CustomOptionParser {
private:
  flexbuffers::TypedVector keys_;
  flexbuffers::Vector values_;

public:
  CustomOptionParser(const flexbuffers::Map &map);
  CustomOptionParser(const char *buffer, size_t buffer_length);
  flexbuffers::Reference parseNamedCustomOption(const char *name) const;
};

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite

#endif // XCORE_CUSTOM_OPTIONS_H_
