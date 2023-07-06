// Copyright (c) 2020, XMOS Ltd, All rights reserved
#ifndef INFERENCE_ENGINE_H_
#define INFERENCE_ENGINE_H_

#ifndef __XC__
#define UNSAFE /**/
#else
#define UNSAFE unsafe
#endif

#if !defined(XTFLM_DISABLED)

#if defined(__xtflm_conf_h_exists__)
#include "xtflm_conf.h"
#else

#ifndef XTFLM_OPERATORS
#define XTFLM_OPERATORS 10
#endif

#endif

#include "tensorflow/lite/c/c_api_types.h"
#include "xcore_config.h"

#ifdef __cplusplus

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "xcore_device_memory.h"
#include "xcore_error_reporter.h"
#include "xcore_interpreter.h"
#include "xcore_ops.h"
#include "xcore_profiler.h"

/** Structure that contains all the TensorFlowLite for Micro objects that must
 * be allocated to create an interpreter. One of these structures has to be
 * allocated for each inference engine. This structure contains C++ objects, and
 * must therefore be allocated inside a C++ source file.
 */
struct tflite_micro_objects {
  tflite::micro::xcore::XCoreErrorReporter error_reporter;
  tflite::micro::xcore::XCoreProfiler xcore_profiler;
  uint64_t interpreter_buffer
      [(sizeof(tflite::micro::xcore::XCoreInterpreter) + sizeof(uint64_t) - 1) /
       sizeof(uint64_t)]; // This needs to be aligned on a double word boundary
  tflite::MicroMutableOpResolver<XTFLM_OPERATORS> resolver;

  tflite::micro::xcore::XCoreInterpreter *interpreter;
  const tflite::Model *model;
};
#endif

#endif

// Opaque definition for the C++ struct above, used in C.
struct tflite_micro_objects;

/** Structure that contains all the data needed to describe an inference engine
 * This structure contains no C++, just standard C pointers and arrays.
 */
typedef struct inference_engine {
  uint32_t *UNSAFE
      memory_primary; ///< Pointer to space for tensor arena and optional model
  uint32_t *UNSAFE
      memory_secondary; ///< Pointer to secondary space. If null,
                        // use the primary for model and tensor arena
  uint32_t
      outputs; ///< Number of output tensors, initialised on loading a model.
  uint32_t inputs; ///< Number of input tensors, initialised on loading a model.
  uint32_t *UNSAFE
      output_buffers[NUM_OUTPUT_TENSORS]; ///< Pointers to output tensors.
  uint32_t *UNSAFE
      input_buffers[NUM_INPUT_TENSORS];      ///< Pointers to input tensors.
  uint32_t output_sizes[NUM_OUTPUT_TENSORS]; ///< Size of each output tensor in
                                             ///< bytes.
  uint32_t
      input_sizes[NUM_INPUT_TENSORS]; ///< Size of each input tensor in bytes.
  uint32_t output_size; ///< Total size of all outputs - TODO: obsolete?
  uint32_t input_size;  ///< Total size of all inputs - TODO: obsolete?
  uint32_t
      memory_primary_bytes; ///< Number of bytes available in primary memory
  uint32_t
      memory_secondary_bytes; ///< Number of bytes available in secondary memory
  uint32_t
      output_times_size; ///< Number of bytes available to store profiling data
  uint32_t operators_size;       ///< ???
  uint32_t *UNSAFE output_times; ///< pointer to profiling data, one per layer
  struct tflite_micro_objects *UNSAFE
      xtflm;             ///< Pointer to C++ XTFLM object - opaque to C
                         // status for the engine to maintain
  uint32_t haveModel;    ///< if 1: we have a model
  uint32_t chainToNext;  ///< if 1: we are chained (could be implicit in c_push
                         ///< being non-null?)
  uint32_t acquireMode;  ///< if non zero we're acquiring data
  uint32_t outputGpioEn; // TODO: should this be here? Possibly not
  int8_t outputGpioThresh[AISRV_GPIO_LENGTH];
  uint8_t outputGpioMode;
  uint32_t debug_log_buffer[MAX_DEBUG_LOG_LENGTH /
                            sizeof(uint32_t)]; ///< buffer for error messages
  uint32_t arena_needed_bytes;                 ///< Total arena needed in bytes.
  uint32_t num_threads;
  struct xc_context_config_t xc_config;
} inference_engine_t;

#ifdef __cplusplus
#ifndef XTFLM_DISABLED
/** Function that initializes the inference_engine object, given a
 * tflite_micro_objects object. This function has to be called from a C++ source
 * files, and it performs the initialisation of the inference engine. This
 * involves setting up basic pointers inside the IE object, nothing else.
 *
 * The function returns the operator-resolver, which must be be used to add all
 * necessary operators to the inference engine. A typical calling sequence is as
 * follows::
 *
 *    uint32_t data_ext[100000/sizeof(int)];
 *    [...]
 *        static struct tflite_micro_objects s0;
 *        auto *resolver = inference_engine_initialize(ie,
 *                                                     data_ext, 100000,
 *                                                     nullptr,  0,
 *                                                     &s0);
 *        resolver->AddAdd();
 *        resolver->AddConv2D();
 *        resolver->AddCustom(tflite::ops::micro::xcore::Conv2D_V2_OpCode,
 *                   tflite::ops::micro::xcore::Register_Conv2D_V2());
 *    [...]
 *
 * Note that when tensorflow lite for micro is disabled this function will not
 * exist as it depends on all and sundry in XTFLM.
 *
 * Note that the lifetime of all spaces passed to this function should be longer
 * than the lifetime of the inference engine. Typically all spaces are declared
 * globally.
 *
 * \param ie                  Pointer to an uninitialized inference engine
 * object, allocated by the caller. \param memory_primary      Pointer to the
 * space to be used for the tensor arena, allocated by the caller. If the fourth
 * parameter is null, then this space will be used for both model and tensor
 * arena. \param n_primary           Number of bytes available in primary memory
 * \param memory_secondary    Pointer to the space to be used for the model
 *                            allocated by the caller. If this parameter is
 * null, then the primary memory will be used for both model and tensor arena.
 * \param n_secondary         Number of bytes available in secondary memory
 * \param xtflmo               C++ structure for storing the XTFLM data
 * structures. Must be allocated by the caller.
 *
 */
tflite::MicroMutableOpResolver<XTFLM_OPERATORS> *inference_engine_initialize(
    inference_engine_t *UNSAFE ie, uint32_t memory_primary[],
    uint32_t n_memory_primary, uint32_t memory_secondary[],
    uint32_t n_secondary, struct tflite_micro_objects *UNSAFE xtflmo);
#endif
extern "C" {
#endif
/** Function that unloads a model frmo the inference engine. This function
 * must be called before before overwriting the model. For example, you
 * have ran a model successfully, before you store a new model over the top
 * of the model (from flash or anywhere else), you must first unload the
 * model, then you can overwrite the model, and finally you can call the
 * inference_engine_load_model to set up the new model.
 *
 * It is safe to call unload model if there is no model loaded.
 *
 * \param ie           pointer to inference engine.
 */
void inference_engine_unload_model(inference_engine_t *UNSAFE ie);

/** Function that loads a model into the inference engine. The model must be
 * stored in either of the two spaces passed to the inference_engine_initialize
 * function above: either the dedicated data_model space or the space shared
 * with the tensor_arena. This funciton assumes the model is in place already
 * and will simply parse it, not copy it.
 *
 * \param ie           pointer to inference engine.
 * \param model_bytes  Number of bytes in the model
 * \param model_data   Pointer to the model (one of data_model or
 * data_tensor_arena passed above) \param c_flash      Optional channel to flash
 * server, to be used for fetching parameter blocks
 *
 * \returns            non zero indicates an error
 */
#ifdef __XC__
    int inference_engine_load_model(inference_engine_t * UNSAFE ie, uint32_t model_bytes, uint32_t * UNSAFE model_data, chanend ?c_flash);
#else
int inference_engine_load_model(inference_engine_t *UNSAFE ie,
                                uint32_t model_bytes,
                                uint32_t *UNSAFE model_data, void *flash_data);
#endif

    /** Function that invokes the inference engine. This function will create an
     * extra four threads enabling a model to run in 5 threads.
     *
     * \param ie           pointer to inference engine.
     */
    int interp_invoke_par_5(inference_engine_t *ie);
    int interp_invoke_par_4(inference_engine_t *ie);
    int interp_invoke_par_3(inference_engine_t *ie);
    int interp_invoke_par_2(inference_engine_t *ie);
    int interp_invoke(inference_engine_t *ie);
    TfLiteStatus interp_invoke_internal(inference_engine_t *ie);

    /** Function that resets variable tensors.
     * This should be called after invoking a model with stateful ops such as LSTM.
     */
    int interp_reset(inference_engine_t *ie);

    /** Function that prints a summary of the time each operator took. This
     * function uses printf - you may want to avoid calling it.
     *
     * \param ie           pointer to inference engine.
     */
    void print_profiler_summary(inference_engine_t *UNSAFE ie);
#ifdef __cplusplus
    };
#endif

#endif // INFERENCE_ENGINE_H_
