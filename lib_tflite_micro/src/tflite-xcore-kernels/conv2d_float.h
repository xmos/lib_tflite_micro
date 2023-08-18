#ifndef _conv2d_float_h_
#define _conv2d_float_h_

#ifdef __cplusplus
extern "C" {
#endif

/** Function that calculates a fully connected.
 *
 * @param  outputs              pointer to the output data, the output data will
 * be stored as an array [out_features]
 * @param  inputs               pointer to the input data, the input data must
 * be stored as an array [input_features]
 * @param  kernels              pointer to the kernels, the kernels
 *                              must be stored as an array
 *                              [out_features][input_features]
 * @param  out_features         dimension 1 of the output array
 * @param  input_features       dimension 1 of the input array
 * @returns                     number of MACCs
 */
extern int xc_fc_float_ref(float *outputs, float *inputs, float *kernels,
                           int out_features, int input_features);

/** Optimized function that calculates a fully connected.
 *
 * @param  outputs              pointer to the output data, the output data will
 * be stored as an array [out_features]
 * @param  inputs               pointer to the input data, the input data must
 * be stored as an array [input_features]
 * @param  kernels              pointer to the kernels, the kernels
 *                              must be stored as an array
 *                              [out_features][input_features]
 * @param  out_features         dimension 1 of the output array
 * @param  input_features       dimension 1 of the input array
 * @param  out_f_start          output features to start at
 * @param  out_f_end            output features to end at plus one
 * @returns                     number of MACCs
 */
extern int xc_fc_float_opt(float *outputs, float *inputs, float *kernels,
                           int out_features, int input_features, int out_f_start,
                           int out_f_end);

/** Function that calculates a convolution with a 5x2 filter with stride 2
 * over dimension 2 of a tensor over a tensor.
 *
 * @param  outputs     pointer to the output data, the output data will be
 *                     stored as an array [out_w][out_h][out_depth]
 * @param  inputs      pointer to the input data, the input data must be
 *                     stored as an array [input_w][input_h][input_depth]
 * @param  kernels     pointer to the kernels, the kernels
 *                     must be stored as an array
 *                     [out_depth][5][2][depth]
 * @param  bias        pointer to the biases, the bias must be stored as an
 * array [out_depth]
 * @param  out_w       dimension 2 of the output array
 * @param  out_h       dimension 1 of the output array
 * @param  out_depth   dimension 3 of the output array
 * @param  input_w     dimension 2 of the input array
 * @param  input_h     dimension 1 of the input array
 * @param  input_depth dimension 3 of the input array
 * @returns            number of MACCs
 */
extern int xc_conv2d_float_kw5xh2_stride_w3_ref(float *outputs, float *inputs,
                                                float *kernels, float *biases,
                                                int out_w, int out_h,
                                                int out_depth, int input_w,
                                                int input_h, int input_depth);

/** Optimised function that calculates a convolution with a 5x2 filter with
 * stride 2 over dimension 2 of a tensor. For parallel usage, supply
 * multiple invocations with different values of out_depth_start and
 * out_depth_end so that the whole output depth is covered between all of
 * them.
 *
 * @param  outputs     pointer to the output data, the output data will be
 *                     stored as an array [out_w][out_h][out_depth]
 * @param  inputs      pointer to the input data, the input data must be
 *                     stored as an array [input_w][input_h][input_depth]
 * @param  kernels     pointer to the kernels, the kernels
 *                     must be stored as an array
 *                     [out_depth][5][2][depth]
 * @param  bias        pointer to the biases, the bias must be stored as an
 * array [out_depth]
 * @param  out_w       dimension 2 of the output array
 * @param  out_h       dimension 1 of the output array
 * @param  out_depth   dimension 3 of the output array
 * @param  input_w     dimension 2 of the input array
 * @param  input_h     dimension 1 of the input array
 * @param  input_depth dimension 3 of the input array
 * @param  out_depth_start output depth to start at
 * @param  out_depth_end output depth to end at plus one
 */
extern void xc_conv2d_float_kw5xh2_stride_w3_opt(
    float *outputs, float *inputs, float *kernels, float *biases, int out_w,
    int out_h, int out_depth, int input_w, int input_h, int input_depth,
    int out_depth_start, int out_depth_end);

/** Function that calculates a transposed convolution with a 5x2 filter with
 * stride 3 in dimension 2 over a tensor.
 *
 * @param  outputs     pointer to the output data, the output data will be
 *                     stored as an array [out_w][out_h][out_depth]
 * @param  inputs      pointer to the input data, the input data must be
 *                     stored as an array [input_w][input_h][input_depth]
 * @param  kernels     pointer to the kernels, the kernels
 *                     must be stored as an array
 *                     [out_depth][5][2][depth]
 * @param  bias        pointer to the biases, the bias must be stored as an
 * array [out_depth]
 * @param  out_w       dimension 2 of the output array
 * @param  out_h       dimension 1 of the output array
 * @param  out_depth   dimension 3 of the output array
 * @param  input_w     dimension 2 of the input array
 * @param  input_h     dimension 1 of the input array
 * @param  input_depth dimension 3 of the input array
 * @returns            number of MACCs
 */
extern int xc_transpose_conv2d_float_kw5xh2_stride_h3_ref(
    float *outputs, float *inputs, float *kernels, float *biases, int out_w,
    int out_h, int out_depth, int input_w, int input_h, int input_depth);

/** Optimised function that calculates a transposed convolution with a 5x2
 * filter with stride 3 over dimension 2 of a tensor. For parallel usage,
 * supply multiple invocations with different values of out_depth_start and
 * out_depth_end so that the whole output depth is covered between all of
 * them.
 *
 * @param  outputs     pointer to the output data, the output data will be
 *                     stored as an array [out_w][out_h][out_depth]
 * @param  inputs      pointer to the input data, the input data must be
 *                     stored as an array [input_w][input_h][input_depth]
 * @param  kernels     pointer to the kernels, the kernels
 *                     must be stored as an array
 *                     [out_depth][5][2][depth]
 * @param  bias        pointer to the biases, the bias must be stored as an
 * array [out_depth]
 * @param  out_w       dimension 2 of the output array
 * @param  out_h       dimension 1 of the output array
 * @param  out_depth   dimension 3 of the output array
 * @param  input_w     dimension 2 of the input array
 * @param  input_h     dimension 1 of the input array
 * @param  input_depth dimension 3 of the input array
 * @param  out_depth_start output depth to start at
 * @param  out_depth_end output depth to end at plus one
 */
extern void xc_transpose_conv2d_float_kw5xh2_stride_h3_opt(
    float *outputs, float *inputs, float *kernels, float *biases, int out_w,
    int out_h, int out_depth, int input_w, int input_h, int input_depth,
    int out_depth_start, int out_depth_end);

#ifdef __cplusplus
};
#endif

#endif
