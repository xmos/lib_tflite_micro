#include <cassert>
#include <cstdio>
#include <cstdint>
#include "tensorflow/lite/kernels/internal/reference/integer_ops/pooling.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/internal/common.h"

int main() {
    // 1. Define tensor dimensions (NHWC format: Batch, Height, Width, Channels)
    // Input: 1 batch, 4x4 spatial grid, 1 channel -> shape [1, 4, 4, 1]
    const int input_batches = 1;
    const int input_height = 4;
    const int input_width = 4;
    const int input_depth = 1;

    // Output with a 2x2 pooling window and stride 2 -> shape [1, 2, 2, 1]
    const int output_height = 2;
    const int output_width = 2;

    // 2. Prepare raw int8 input data (values ranging from -128 to 127)
    // 4x4 matrix
    int8_t input_data[16] = {
        10, 20, 30, 40,
        50, 60, 70, 80,
        90, 100, 110, 120,
        -10, -20, -30, -40
    };

    // Output buffer
    int8_t output_data[4] = {0};

    // 3. Configure Pooling Parameters (similar to TfLitePoolParams)
    tflite_micro::PoolParams op_params = {};
    op_params.stride_height = 2;
    op_params.stride_width = 2;
    op_params.filter_height = 2;
    op_params.filter_width = 2;
    op_params.padding_type = tflite_micro::PaddingType::kValid;
    
    // Activation clamp limits for int8 (typically -128 to 127)
    op_params.float_activation_min = -128.0f;
    op_params.float_activation_max = 127.0f;
    op_params.quantized_activation_min = -128;
    op_params.quantized_activation_max = 127;

    // 4. Define shape wrappers expected by TFLite internal APIs
    const int32_t input_dims[] = {input_batches, input_height, input_width, input_depth};
    const int32_t output_dims[] = {input_batches, output_height, output_width, input_depth};
    tflite_micro::RuntimeShape input_shape(4, input_dims);
    tflite_micro::RuntimeShape output_shape(4, output_dims);

    // 5. Invoke the Int8 Reference Average Pool Kernel directly
    bool res = tflite_micro::reference_integer_ops::AveragePool(
        op_params,
        input_shape, input_data,
        output_shape, output_data
    );
    assert(res);

    const int8_t expected_data[] = {35, 55, 40, 40};
    printf("Output data:\n");
    for (int i = 0; i < 4; ++i) {
        printf("%d ", output_data[i]);
        assert(output_data[i] == expected_data[i]);
    }
    printf("\n");
    return 0;
}
