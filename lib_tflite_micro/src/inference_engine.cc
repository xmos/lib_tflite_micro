// Copyright (c) 2021, XMOS Ltd, All rights reserved
#include "inference_engine.h"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include "inference_engine.h"
#include "thread_call.h"

#if !defined(XTFLM_DISABLED)

extern "C" void DebugLog(const char *s) {
  while (*s) {
    putchar(*s);
    s++;
  }
} // Not sure why we need this

tflite::MicroMutableOpResolver<XTFLM_OPERATORS> *
inference_engine_initialize(inference_engine *ie, uint32_t memory_primary[],
                            uint32_t n_primary, uint32_t memory_secondary[],
                            uint32_t n_secondary,
                            struct tflite_micro_objects *xtflmo) {
  // First initialise the structure with the three memory objects
  // internal memory, external memory, and XTFLM objects.
  memset(ie, 0, sizeof(*ie));
  ie->xtflm = xtflmo;
  ie->memory_primary = memory_primary;
  ie->memory_secondary = memory_secondary;
  ie->memory_primary_bytes = n_primary;
  ie->memory_secondary_bytes = n_secondary;
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
                                uint32_t *model_data, void *flash_data) {

  if (ie->xtflm->interpreter) {
    TF_LITE_REPORT_ERROR(&ie->xtflm->error_reporter, "Model not unloaded");
    return 9;
  }

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  ie->xtflm->model = tflite::GetModel((uint8_t *)model_data);
  uint model_version = ie->xtflm->model->version();
  if (model_version != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(&ie->xtflm->error_reporter,
                         "Model provided is schema version %u not equal to "
                         "supported version %d.",
                         model_version, TFLITE_SCHEMA_VERSION);
    return 1;
  }

  // Now work out where the tensor arena goes
  uint8_t *kTensorArena = (uint8_t *)ie->memory_primary;
  int kTensorArenaSize = ie->memory_primary_bytes;

  if (model_data != ie->memory_secondary) {
    uint32_t model_ints = (model_bytes + 3) & ~0x03; // Align 4
    kTensorArena += model_ints;
    kTensorArenaSize -= model_ints;
  }

  // Need to memset the arena to 0 otherwise assertion in xcore_planning.cc
  memset(kTensorArena, 0, kTensorArenaSize);

  // Build an interpreter to run the model with
  ie->xtflm->interpreter = tflite::micro::xcore::XCoreInterpreter::Create(
      ie->xtflm->interpreter_buffer, ie->xtflm->model, ie->xtflm->resolver,
      kTensorArena, kTensorArenaSize, &ie->xtflm->error_reporter, true,
      &ie->xtflm->xcore_profiler, flash_data);

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

  // Obtain pointers to the model's input and output tensors.
  ie->inputs = ie->xtflm->interpreter->inputs_size();
  ie->input_size = 0;
  if (ie->inputs > NUM_INPUT_TENSORS) {
    TF_LITE_REPORT_ERROR(&ie->xtflm->error_reporter, "Too many input tensors");
    return 3;
  }
  for (int i = 0; i < ie->inputs; i++) {
    ie->input_buffers[i] =
        (uint32_t *)(ie->xtflm->interpreter->input(i)->data.raw);
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
  ie->output_times = (uint32_t *)ie->xtflm->xcore_profiler.GetEventDurations();
  ie->output_times_size = ie->operators_size;

  return 0;
}

int interp_invoke_par_4(inference_engine *ie)
{
    return thread_invoke_4((void *)ie, (void *)&ie->xtflm->interpreter->thread_info);
    // TODO: when all debugged we can type it solidly.
}

TfLiteStatus interp_invoke_internal(inference_engine *ie)
{
    return ie->xtflm->interpreter->Invoke();
}

int interp_invoke(inference_engine *ie)
{
    // Run inference, and report any error
    TfLiteStatus invoke_status = interp_invoke_internal(ie);

  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(&ie->xtflm->error_reporter, "Invoke failed\n");
    return 1;
  }

  return 0;
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
                                   static_cast<tflite::BuiltinOperator>(
                                       opcode->deprecated_builtin_code()));
      if (builtin_code == tflite::BuiltinOperator_CUSTOM) {
        const char *name = ie->xtflm->interpreter->node_name(0, i);
        if (name != NULL) {
          op_name = name;
        } else {
          op_name = opcode->custom_code()->c_str();
        }
      } else {
        op_name = tflite::EnumNameBuiltinOperator(
            tflite::BuiltinOperator(builtin_code));
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
extern "C" void DebugLog(const char *s) {}

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
