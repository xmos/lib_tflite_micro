// Copyright (c) 2021, XMOS Ltd, All rights reserved
#include "lib_nn/api/version.h"
#include "inference_engine.h"
#include "version.h"
#include "xcore_shared_config.h"
#include "thread_call.h"
#include <cstddef>
#include <cstdint>
#include <cstdio>

#if !defined(XTFLM_DISABLED)

tflite_micro::MicroMutableOpResolver<XTFLM_OPERATORS> *
inference_engine_initialize(inference_engine *ie, uint32_t tensor_arena[],
                            uint32_t tensor_arena_size, uint32_t external_memory[],
                            uint32_t external_memory_size,
                            struct tflite_micro_objects *xtflmo) {
  // First initialise the structure with the three memory objects
  // internal memory, external memory, and XTFLM objects.
  memset(ie, 0, sizeof(*ie));
  xtflmo->interpreter = nullptr;
  ie->xtflm = xtflmo;
  ie->tensor_arena = tensor_arena;
  ie->external_memory = external_memory;
  ie->tensor_arena_size = tensor_arena_size;
  ie->external_memory_size = external_memory_size;
  ie->xtflm->error_reporter.Init((char *)ie->debug_log_buffer,
                                 MAX_DEBUG_LOG_LENGTH);
  // Now add all the operators that we need
  auto *resolver = &ie->xtflm->resolver;
  return resolver;
}

void inference_engine_unload_model(inference_engine *ie) {
  if (ie->xtflm->interpreter) {
    delete ie->xtflm->interpreter;
    ie->xtflm->interpreter = nullptr;
  }
}

int inference_engine_load_model(inference_engine *ie, uint32_t model_bytes,
                                uint32_t *model_data, void *weights_data_ptr) {

  if (ie->xtflm->interpreter) {
    TF_LITE_REPORT_ERROR(&ie->xtflm->error_reporter, "Model not unloaded");
    return 9;
  }

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  ie->xtflm->model = tflite_micro::GetModel((uint8_t *)model_data);
  unsigned model_version = ie->xtflm->model->version();

  // Retrieve shared metadata
  shared_config::xcore_metadata_t * shared_config_ptr;
  for (int i = 0; i < ie->xtflm->model->metadata()->size(); ++i) {
    auto metadata = ie->xtflm->model->metadata()->Get(i);
    if (strncmp(metadata->name()->c_str(), shared_config::xcoreMetadataName,
                strlen(shared_config::xcoreMetadataName)) == 0) {
      auto buf = metadata->buffer();
      auto *buffer = (*ie->xtflm->model->buffers())[buf];
      auto *array = buffer->data();

      shared_config_ptr = (shared_config::xcore_metadata_t *)array->data();
      // Check version with metadata version
      // If major version is zero, then minor versions must match
      // Otherwise, major versions must match and binary minor version
      // must be less or equal to runtime minor version
      // Check if lib_tflite_micro version matches with metadata version
      if ((shared_config_ptr->lib_tflite_micro_major_version == 0 &&
           lib_tflite_micro::major_version == 0 &&
           shared_config_ptr->lib_tflite_micro_minor_version !=
               lib_tflite_micro::minor_version) ||
          (shared_config_ptr->lib_tflite_micro_major_version !=
           lib_tflite_micro::major_version) ||
          (shared_config_ptr->lib_tflite_micro_minor_version >
           lib_tflite_micro::minor_version)) {
        TF_LITE_REPORT_ERROR(&ie->xtflm->error_reporter,
                             "Model provided has lib_tflite_micro version "
                             "%d.%d not supported on "
                             "runtime lib_tflite_micro version %u.%u .",
                             shared_config_ptr->lib_tflite_micro_major_version,
                             shared_config_ptr->lib_tflite_micro_minor_version,
                             lib_tflite_micro::major_version,
                             lib_tflite_micro::minor_version);
        return 1;
      }

      // Check if lib_nn version matches with metadata version
      if ((shared_config_ptr->lib_nn_major_version == 0 && lib_nn::major_version == 0 &&
        shared_config_ptr->lib_nn_minor_version != lib_nn::minor_version) ||
          (shared_config_ptr->lib_nn_major_version != lib_nn::major_version) ||
          (shared_config_ptr->lib_nn_minor_version > lib_nn::minor_version)) {
        TF_LITE_REPORT_ERROR(
            &ie->xtflm->error_reporter,
            "Model provided has lib_nn version %d.%d not supported on "
            "runtime lib_nn version %u.%u .",
            shared_config_ptr->lib_nn_major_version, shared_config_ptr->lib_nn_minor_version,
            lib_nn::major_version, lib_nn::minor_version);
        return 1;
      }

      // NOTE: xformer version is saved for debugging purposes
      // If lib_nn and lib_tflite_micro versions are as expected,
      // then the xformer version doesn't matter as the model should execute

      // Get thread count required from the runtime by the xformer
      ie->num_threads = shared_config_ptr->required_thread_count;

      // Set target arch based on the compiled model
      if(shared_config_ptr->target_arch == nn_target_arch_t::TARGET_ARCH_XS3A){
        SetNNTargetArch(nn_target_arch_t::TARGET_ARCH_XS3A);
      } else if(shared_config_ptr->target_arch == nn_target_arch_t::TARGET_ARCH_VX4A) {
        SetNNTargetArch(nn_target_arch_t::TARGET_ARCH_VX4A);
      } else {
        assert(false && "Arch not defined!");
      }
    }
  }

  if (model_version != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(&ie->xtflm->error_reporter,
                         "Model provided is schema version %u not equal to "
                         "supported version %d.",
                         model_version, TFLITE_SCHEMA_VERSION);
    return 1;
  }

  // Now work out where the tensor arena goes
  uint8_t *kTensorArena = (uint8_t *)ie->tensor_arena;
  int kTensorArenaSize = ie->tensor_arena_size;

  uint32_t model_ints = (model_bytes + 3) & ~0x03; // Align 4
  kTensorArena += model_ints;
  kTensorArenaSize -= model_ints;

  // Need to memset the arena to 0 otherwise assertion in xcore_planning.cc
  memset(kTensorArena, 0, kTensorArenaSize);

  // Build an interpreter to run the model with
  ie->xtflm->interpreter = tflite_micro::micro::xcore::XCoreInterpreter::Create(
      (uint8_t *)ie->xtflm->interpreter_buffer, ie->xtflm->model,
      ie->xtflm->resolver, kTensorArena, kTensorArenaSize, true, &ie->xtflm->xcore_profiler);
  ie->xc_config.model_thread_count = ie->num_threads;
  ie->xc_config.paging_ptr = ie->external_memory;
  ie->xc_config.weights_data_ptr = weights_data_ptr;
  TfLiteStatus set_external_context_status =
      ie->xtflm->interpreter->SetMicroExternalContext((void *)&ie->xc_config);
  if (set_external_context_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(&ie->xtflm->error_reporter,
                         "SetExternalContext() failed");
    return 2;
  }

  // Allocate memory from the kTensorArena for the model's tensors.
  TfLiteStatus allocate_tensors_status =
      ie->xtflm->interpreter->AllocateTensors();
  if (allocate_tensors_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(&ie->xtflm->error_reporter,
                         "AllocateTensors() failed");
    return 2;
  }
  ie->operators_size =
      ie->xtflm->model->subgraphs()->Get(0)->operators()->size();

  // Make maps of input/output indexes to shared config data if present
  std::unordered_map<int, shared_config::tensor_info_t*> orig_index_input_map;
  std::unordered_map<int, shared_config::tensor_info_t*> orig_index_output_map;
  for (int i=0; i <shared_config_ptr->num_external_input_tensors; i++ ){
    orig_index_input_map[shared_config_ptr->external_input_tensors_data[i].index] = &shared_config_ptr->external_input_tensors_data[i];
  }
  for (int i=0; i <shared_config_ptr->num_external_output_tensors; i++ ){
    orig_index_output_map[shared_config_ptr->external_output_tensors_data[i].index] = &shared_config_ptr->external_output_tensors_data[i];
  }

  // Obtain pointers to the model's input and output tensors.
  ie->inputs = ie->xtflm->interpreter->inputs_size();
  ie->input_size = 0;
  if (ie->inputs > NUM_INPUT_TENSORS) {
    TF_LITE_REPORT_ERROR(&ie->xtflm->error_reporter, "Too many input tensors");
    return 3;
  }

  for (int i = 0; i < ie->inputs; i++) {
    if(orig_index_input_map.count(i)){
      if (ie->xtflm->interpreter->input(i)->bytes != orig_index_input_map[i]->size) {
        TF_LITE_REPORT_ERROR(&ie->xtflm->error_reporter, "External input tensor size doesn't match!");
        return 6;
      }
      ie->input_buffers[i] = (uint32_t *)(((int8_t *)ie->external_memory) + orig_index_input_map[i]->external_address);
    } else {
      ie->input_buffers[i] =
      (uint32_t *)(ie->xtflm->interpreter->input(i)->data.raw);
    }
    ie->input_sizes[i] = ie->xtflm->interpreter->input(i)->bytes;
    ie->input_size += ie->input_sizes[i];
  }

  ie->outputs = ie->xtflm->interpreter->outputs_size();
  ie->output_size = 0;
  if (ie->outputs > NUM_OUTPUT_TENSORS) {
    TF_LITE_REPORT_ERROR(&ie->xtflm->error_reporter,
                         "Too many output tensors %d", ie->outputs);
    return 4;
  }
  for (int i = 0; i < ie->outputs; i++) {
    ie->output_buffers[i] =
        (uint32_t *)(ie->xtflm->interpreter->output(i)->data.raw);
    ie->output_sizes[i] = ie->xtflm->interpreter->output(i)->bytes;
    ie->output_size += ie->output_sizes[i];
  }
  for (int i = 0; i < ie->outputs; i++) {
    if(orig_index_output_map.count(i)){
      if (ie->xtflm->interpreter->output(i)->bytes != orig_index_output_map[i]->size) {
        TF_LITE_REPORT_ERROR(&ie->xtflm->error_reporter, "External output tensor size doesn't match!");
        return 6;
      }
      ie->output_buffers[i] = (uint32_t *)(((int8_t *)ie->external_memory) + orig_index_output_map[i]->external_address);
    } else {
      ie->output_buffers[i] =
      (uint32_t *)(ie->xtflm->interpreter->output(i)->data.raw);
    }
    ie->output_sizes[i] = ie->xtflm->interpreter->output(i)->bytes;
    ie->output_size += ie->output_sizes[i];
  }
  ie->output_times = (uint32_t *)ie->xtflm->xcore_profiler.GetEventDurations();
  ie->output_times_size = ie->operators_size;

  // We calculate and save the total bytes needed for the arena
  // If the model is in primary memory, we add model bytes as that would also be
  // in the arena
  ie->arena_needed_bytes =
      ie->xtflm->interpreter->arena_used_bytes();
  ie->arena_needed_bytes +=
      16; // buffers are aligned to 16 bytes so we add this to be safe
  ie->arena_needed_bytes += model_bytes;

  return 0;
}

int interp_reset(inference_engine *ie) {
  TfLiteStatus reset_status = ie->xtflm->interpreter->Reset();
  if (reset_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(&ie->xtflm->error_reporter,
                         "Reset() failed");
    return 2;
  }
  return 0;
}

int interp_invoke_par_5(inference_engine *ie) {
  if (ie->num_threads > 5) {
    printf("Thread count (5) does not match model thread count\n");
    return 5;
  }
  thread_client(&ie->xc_config.thread_info, 0);
  thread_client(&ie->xc_config.thread_info, 1);
  thread_client(&ie->xc_config.thread_info, 2);
  thread_client(&ie->xc_config.thread_info, 3);
  return interp_invoke_internal(ie);
}

TfLiteStatus interp_invoke_internal(inference_engine *ie) {
  return ie->xtflm->interpreter->Invoke();
}

void print_profiler_summary(inference_engine *ie) {
  auto *opcodes = ie->xtflm->model->operator_codes();
  uint64_t total = 0;
  const char *op_name;

  uint32_t count = ie->xtflm->xcore_profiler.GetNumEvents();
  uint32_t const *times = ie->xtflm->xcore_profiler.GetEventDurations();
  auto *subgraphs = ie->xtflm->model->subgraphs();

  int n_operators = 0;
  uint64_t operator_times[XCORE_PROFILER_DEFAULT_MAX_LEVELS];
  const char *operator_names[XCORE_PROFILER_DEFAULT_MAX_LEVELS];
  for (size_t i = 0; i < ie->operators_size && i < count; ++i) {
    const auto *op = (*subgraphs)[0]->operators()->Get(i);
    const size_t index = op->opcode_index();
    if (index >= opcodes->size()) {
      op_name = "Missing registration";
    } else {
      auto *opcode = (*opcodes)[index];
      auto builtin_code = std::max(opcode->builtin_code(),
                                   static_cast<tflite_micro::BuiltinOperator>(
                                       opcode->deprecated_builtin_code()));
      if (builtin_code == tflite_micro::BuiltinOperator_CUSTOM) {
        const char *name = ie->xtflm->interpreter->node_name(0, i);
        if (name != NULL) {
          op_name = name;
        } else {
          op_name = opcode->custom_code()->c_str();
        }
      } else {
        op_name = tflite_micro::EnumNameBuiltinOperator(
            tflite_micro::BuiltinOperator(builtin_code));
      }
    }

    total += times[i];
    printf("Operator %3zu %-20s took %5u ms\n", i, op_name, times[i] / 100000);
    int found = 0;
    for (int j = 0; j < n_operators; j++) {
      if (strcmp(operator_names[j], op_name) == 0) {
        operator_times[j] += times[i];
        found = 1;
        break;
      }
    }
    if (!found && n_operators != XCORE_PROFILER_DEFAULT_MAX_LEVELS) {
      operator_names[n_operators] = op_name;
      operator_times[n_operators] = times[i];
      n_operators++;
    }
  }
  printf("TOTAL %llu ticks\n", total);

  for (size_t index = 0; index < n_operators; index++) {
    printf("Operator-class %-20s took %5llu ms %3d%%\n", operator_names[index],
           operator_times[index] / 100000,
           (int)(100 * operator_times[index] / total));
  }
}

#else

// STUBS for when XTFLM is disabled.

void inference_engine_unload_model(inference_engine *ie) {}

void print_profiler_summary(inference_engine *ie) {}

void inference_engine_initialize(inference_engine *ie,
                                 uint8_t data_tensor_arena[], uint32_t n_int,
                                 uint8_t data_ext[], uint32_t n_ext,
                                 struct tflite_micro_object *xtflmo) {}

int inference_engine_load_model(inference_engine *ie, uint32_t model_bytes,
                                uint32_t *model_data) {
  printf("Inference engine disabled, model not loaded\n");
  return AISRV_STATUS_OKAY;
}

aisrv_status_t interp_invoke(inference_engine *ie) {
  printf("Inference engine disabled, model not executed\n");
  return AISRV_STATUS_OKAY;
}

#endif
