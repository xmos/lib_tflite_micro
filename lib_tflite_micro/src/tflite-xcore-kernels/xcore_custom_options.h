#ifndef XCORE_CUSTOM_OPTIONS_H_
#define XCORE_CUSTOM_OPTIONS_H_

#include "flatbuffers/flexbuffers.h"
#include "xcore_ops.h"
#include "xcore_planning.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

void parse_custom_options(TfLiteContext *context, const char *buffer,
                          size_t length, ExecutionPlan *plan);

void parse_custom_options(TfLiteContext *context, const char *buffer,
                          size_t length, PoolingParams &pooling_params,
                          ExecutionPlan *plan = nullptr);

void parse_custom_options(TfLiteContext *context, const char *buffer,
                          size_t length, Conv2DParams &conv2d_params,
                          ExecutionPlan *plan = nullptr);

void parse_custom_options(TfLiteContext *context, const char *buffer,
                          size_t length, int32_t *stride_h = nullptr,
                          int32_t *stride_w = nullptr,
                          int32_t *pool_h = nullptr, int32_t *pool_w = nullptr,
                          int32_t *K_w = nullptr, Conv2DPadding *pad = nullptr,
                          ExecutionPlan *plan = nullptr);

class CustomOptionParser {
private:
  flexbuffers::TypedVector keys_;
  flexbuffers::Vector values_;

public:
  CustomOptionParser(const flexbuffers::Map &map);
  CustomOptionParser(const char *buffer, size_t buffer_length);
  flexbuffers::Reference parseNamedCustomOption(const std::string &name) const;
  flexbuffers::Vector parseElementwiseJobSizes() const;
};

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite

#endif // XCORE_CUSTOM_OPTIONS_H_
