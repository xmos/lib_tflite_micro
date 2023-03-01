#include "Compiler.h"

#include <fstream>
#include <memory>
#include <regex>
#include <vector>

#include "CodeWriter.h"
#include "TypeToString.h"
#include "lib_nn/api/version.h"
#include "lib_tflite_micro/api/version.h"
#include "xcore_config.h"
#include "xtflm_conf.h"

#ifndef SUFFICIENT_ARENA_SIZE
#define SUFFICIENT_ARENA_SIZE (128 * 1024 * 1024)
#endif

#if TF_LITE_PACKED_QUANTIZED_DATA_VERSION
#if TF_LITE_PACKED_QUANTIZED_DATA_VERSION != 100
#error "ONLY TF_LITE_PACKED_QUANTIZED_DATA_VERSION Version 100 supported!"
#endif
#endif

static xc_context_config_t g_xc_config;
static std::vector<tflmc::Allocation> g_loggedAllocations;
static int g_currentNodeIndex = -1;
static uint8_t *g_arenaPtr = nullptr;
static ptrdiff_t g_arena_size = 0;

static void *LoggingAllocatePersistentBuffer(struct TfLiteContext *ctx,
                                             size_t bytes) {
  void *ptr = tflite::GetMicroContext(ctx)->AllocatePersistentBuffer(bytes);
  assert(ptr != nullptr && "Alloc failure");
  g_loggedAllocations.push_back({-(g_arenaPtr - (uint8_t *)ptr + g_arena_size),
                                 bytes, g_currentNodeIndex});
  return ptr;
}

TfLiteStatus tflmc::AllocateTensors(
    std::unique_ptr<tflite::MicroInterpreter> &interpreter) {
  tflite::SubgraphAllocations *allocations =
      interpreter->allocator_.StartModelAllocation(interpreter->model_);

  // if (allocations == nullptr) {
  //   TF_LITE_REPORT_ERROR(error_reporter_,
  //                        "Failed starting model allocation.\n");
  //   initialization_status_ = kTfLiteError;
  //   return kTfLiteError;
  // }

  interpreter->graph_.SetSubgraphAllocations(allocations);

  TF_LITE_ENSURE_STATUS(
      interpreter->PrepareNodeAndRegistrationDataFromFlatbuffer());

  // Only allow AllocatePersistentBuffer in Init stage.
  interpreter->context_.AllocatePersistentBuffer =
      &LoggingAllocatePersistentBuffer;
  interpreter->context_.RequestScratchBufferInArena = nullptr;
  interpreter->context_.GetScratchBuffer = nullptr;
  interpreter->context_.GetExternalContext = nullptr;
  TF_LITE_ENSURE_STATUS(interpreter->graph_.InitSubgraphs());

  // Both AllocatePersistentBuffer and RequestScratchBufferInArena is
  // available in Prepare stage.
  interpreter->context_.RequestScratchBufferInArena =
      tflite::MicroContextRequestScratchBufferInArena;
  // external_context become available in Prepare stage.
  interpreter->context_.GetExternalContext =
      tflite::MicroContextGetExternalContext;

  TF_LITE_ENSURE_STATUS(interpreter->graph_.PrepareSubgraphs());

  // Prepare is done, we're ready for Invoke. Memory allocation is no longer
  // allowed. Kernels can only fetch scratch buffers via GetScratchBuffer.
  interpreter->context_.AllocatePersistentBuffer = nullptr;
  interpreter->context_.RequestScratchBufferInArena = nullptr;
  interpreter->context_.GetScratchBuffer = tflite::MicroContextGetScratchBuffer;

  TF_LITE_ENSURE_OK(
      &interpreter->context_,
      interpreter->allocator_.FinishModelAllocation(
          interpreter->model_, interpreter->graph_.GetAllocations(),
          &interpreter->scratch_buffer_handles_));

  interpreter->micro_context_.SetScratchBufferHandles(
      interpreter->scratch_buffer_handles_);

  // TODO(b/162311891): Drop these allocations when the interpreter supports
  // handling buffers from TfLiteEvalTensor.
  interpreter->input_tensors_ = reinterpret_cast<TfLiteTensor **>(
      interpreter->allocator_.AllocatePersistentBuffer(
          sizeof(TfLiteTensor *) * interpreter->inputs_size()));
  // if (input_tensors_ == nullptr) {
  //   TF_LITE_REPORT_ERROR(
  //       error_reporter_,
  //       "Failed to allocate memory for context->input_tensors_, "
  //       "%d bytes required",
  //       sizeof(TfLiteTensor*) * inputs_size());
  //   return kTfLiteError;
  // }

  for (size_t i = 0; i < interpreter->inputs_size(); ++i) {
    interpreter->input_tensors_[i] =
        interpreter->allocator_.AllocatePersistentTfLiteTensor(
            interpreter->model_, interpreter->graph_.GetAllocations(),
            interpreter->inputs().Get(i), 0);
    // if (input_tensors_[i] == nullptr) {
    //   TF_LITE_REPORT_ERROR(error_reporter_,
    //                        "Failed to initialize input tensor %d", i);
    //   return kTfLiteError;
    // }
  }

  // TODO(b/162311891): Drop these allocations when the interpreter supports
  // handling buffers from TfLiteEvalTensor.
  interpreter->output_tensors_ = reinterpret_cast<TfLiteTensor **>(
      interpreter->allocator_.AllocatePersistentBuffer(
          sizeof(TfLiteTensor *) * interpreter->outputs_size()));
  // if (output_tensors_ == nullptr) {
  //   TF_LITE_REPORT_ERROR(
  //       error_reporter_,
  //       "Failed to allocate memory for context->output_tensors_, "
  //       "%d bytes required",
  //       sizeof(TfLiteTensor*) * outputs_size());
  //   return kTfLiteError;
  // }

  for (size_t i = 0; i < interpreter->outputs_size(); ++i) {
    interpreter->output_tensors_[i] =
        interpreter->allocator_.AllocatePersistentTfLiteTensor(
            interpreter->model_, interpreter->graph_.GetAllocations(),
            interpreter->outputs().Get(i), 0);
    // if (output_tensors_[i] == nullptr) {
    //   TF_LITE_REPORT_ERROR(error_reporter_,
    //                        "Failed to initialize output tensor %d", i);
    //   return kTfLiteError;
    // }
  }

  TF_LITE_ENSURE_STATUS(interpreter->ResetVariableTensors());

  interpreter->tensors_allocated_ = true;
  return kTfLiteOk;
}

TfLiteEvalTensor *tflmc::GetEvalTensor(tflite::MicroInterpreter *interpreter,
                                       int i) {
  auto ctx = &interpreter->context_;
  return ctx->GetEvalTensor(ctx, i);
}

TfLiteTensor *tflmc::GetTensor(tflite::MicroInterpreter *interpreter, int i) {
  auto ctx = &interpreter->context_;
  return ctx->GetTensor(ctx, i);
}

bool tflmc::CompileFile(const std::string &modelFileName,
                        const std::string &outFileName,
                        const std::string &prefix) {
  // Load model flatbuffer.
  std::ifstream model_file(modelFileName, std::ios::binary | std::ios::ate);
  if (!model_file) {
    std::cerr << "Could not open " << modelFileName << " for read\n";
    return false;
  }
  auto sz = model_file.tellg();
  if (sz == std::ifstream::pos_type(-1)) {
    std::cerr << "Failed to read model file size\n";
    return false;
  }
  std::vector<char> model_data(sz);
  model_file.seekg(0, std::ios::beg);
  if (!model_file.read(model_data.data(), sz)) {
    std::cerr << "Failed to read model file\n";
    return false;
  }

  std::ofstream outFile(outFileName);
  if (!outFile) {
    std::cerr << "Failed to create output file\n";
    return false;
  }

  std::ofstream outHeaderFile(outFileName + ".h");
  if (!outHeaderFile) {
    std::cerr << "Failed to create output header file\n";
    return false;
  }

  try {
    bool debugPrint = false;
    Compiler compiler(model_data.data(), prefix);
    compiler.writeSource(outFile);
    compiler.writeHeader(outHeaderFile);
    return true;
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
  } catch (...) {
    std::cerr << "Unknown exception\n";
  }

  return false;
}

tflmc::Compiler::Compiler(const void *modelData, const std::string &prefix,
                          const bool debugPrint)
    : Compiler(modelData, nullptr, prefix, debugPrint) {}

tflmc::Compiler::Compiler(const void *modelData,
                          const struct shared_config::xcore_metadata *sharedCfg,
                          const std::string &prefix, const bool debugPrint)
    : sharedCfg_(sharedCfg), prefix_(prefix), debugPrint_(debugPrint) {
  if (sharedCfg_) {
    numXCThreads_ = sharedCfg_->required_thread_count;
  }
  if (!init(modelData)) {
    throw std::runtime_error("Could not set up compiler");
  }
}

bool tflmc::Compiler::init(const void *modelData) {
  model_ = tflite::GetModel(modelData);
  if (model_->version() != TFLITE_SCHEMA_VERSION) {
    errReporter().Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model_->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }

  auto subgraphs = model_->subgraphs();
  if (subgraphs->size() != 1) {
    std::cerr << "Model needs to have exactly one subgraph as expected by TF "
                 "Lite for Micro\n";
    return false;
  }
  subgraph_ = (*subgraphs)[0];
  auto tensors = subgraph_->tensors();
  if (subgraph_->inputs()->size() == 0 || subgraph_->outputs()->size() == 0) {
    std::cerr << "No inputs or no outputs found in model\n";
    return false;
  }
  for (auto inIndex : *subgraph_->inputs()) {
    inputTensorIndices_.push_back(inIndex);
  }
  for (auto outIndex : *subgraph_->outputs()) {
    outputTensorIndices_.push_back(outIndex);
  }

  if (XTFLM_OPERATORS != 128) {
    std::cerr << "XTFLM_OPERATORS must match the magic number in the template "
                 "parameter for AllOpsResolver!\n";
    return false;
  }
  tflite::ops::micro::xcore::RegisterXCOps(&resolver_);

  // Build an interpreter to run the model with.
  arena_buf_.resize(SUFFICIENT_ARENA_SIZE);

  g_arena_size = SUFFICIENT_ARENA_SIZE;
  std::vector<uint8_t> arena_buf(g_arena_size);
  g_arenaPtr = arena_buf_.data();

  interpreter_ = std::unique_ptr<tflite::MicroInterpreter>(
      new tflite::MicroInterpreter(model_, resolver_, arena_buf_.data(),
                                   arena_buf_.size(), &microErrReporter_));

  assert(interpreter_->graph_.NumSubgraphs() == 1);

  TfLiteStatus set_external_context_status =
      interpreter_->SetMicroExternalContext((void *)&g_xc_config);
  if (set_external_context_status != kTfLiteOk) {
    errReporter().Report("SetExternalContext() failed");
    return false;
  }

  // Allocate memory from the tensor_arena for the model's tensors.
  // TfLiteStatus allocate_status = interpreter_->AllocateTensors();
  TfLiteStatus allocate_status = AllocateTensors(interpreter_);
  if (allocate_status != kTfLiteOk) {
    errReporter().Report("AllocateTensors() failed");
    return false;
  }

  ptrdiff_t ramTensorBufferSize = 0;
  ptrdiff_t romOffset = 0;
  auto numTensors = tensors->size();
  if (numTensors > 0) {
    auto tensor = GetTensor(interpreter_.get(), 0);
    common_tensor_type = tensor->type;
    common_tensor_is_variable = tensor->is_variable;
  }
  printf("\n");
  for (size_t i = 0; i < numTensors; i++) {
    auto tensor = GetTensor(interpreter_.get(), i);
    tensors_.push_back({tensor});
    if (tensor->allocation_type == kTfLiteMmapRo) {
      printf("-1, ");
      memMap_.recordROM(romOffset, tensor->bytes, getTensorName(i));
      romOffset += tensor->bytes;
    } else {
      ptrdiff_t offset = (uint8_t *)tensor->data.data - arena_buf_.data();
      printf("%d, ", offset);
      ptrdiff_t highSize = offset + tensor->bytes;
      ramTensorBufferSize = std::max(ramTensorBufferSize, highSize);
      memMap_.recordRAM(offset, tensor->bytes, getTensorName(i));
    }
    // determine whether we need to individually set these properties for each
    // tensor
    if ((!has_quantization) &&
        tensor->quantization.type != kTfLiteNoQuantization) {
      has_quantization = true;
    }
    if ((!common_tensor_type.None) && common_tensor_type.Some != tensor->type) {
      common_tensor_type.clear();
    }
    if ((!common_tensor_is_variable.None) &&
        common_tensor_is_variable.Some != tensor->is_variable) {
      common_tensor_is_variable.clear();
    }
  }
  printf("\n");

  for (size_t k = 0; k < interpreter_->allocator_.scratch_buffer_request_count_;
       k++) {
    void *data = interpreter_->micro_context_.GetScratchBuffer(k);
    ptrdiff_t offset = (uint8_t *)data - arena_buf_.data();
    tflite::internal::ScratchBufferRequest *requests =
        interpreter_->allocator_.GetScratchBufferRequests();
    int bytes = requests[k].bytes;
    ptrdiff_t highSize = offset + bytes;
    ramTensorBufferSize = std::max(ramTensorBufferSize, highSize);
    memMap_.recordRAM(offset, bytes,
                      "Scratch_idx" + std::to_string((int)k) + "_op" +
                          std::to_string((int)requests[k].node_idx));
    scratchBufferOffsets.push_back(offset);
  }

  for (size_t i = 0; i < interpreter_->operators_size(); i++) {
    auto nodeAndReg = interpreter_->node_and_registration(i);
    auto node = &nodeAndReg.node;
    auto reg = nodeAndReg.registration;
    auto code = tflite::EnumValuesBuiltinOperator()[reg->builtin_code];

    if (debugPrint_) {
      printf("operation %lu: %s\n", i,
             tflite::EnumNamesBuiltinOperator()[code]);
    }

    RegistrationInfo regInfo;
    regInfo.reg = reg;
    regInfo.code = code;
    if (code == tflite::BuiltinOperator_CUSTOM) {
      regInfo.custom_name = reg->custom_name;
      if (regInfo.custom_name == "TFLite_Detection_PostProcess") {
        has_tflite_custom_ops = true;
      }
      has_custom_ops = true;
    }
    auto itOp =
        std::find(registrations_.begin(), registrations_.end(), regInfo);
    if (itOp == registrations_.end()) {
      itOp = registrations_.insert(registrations_.end(), regInfo);
    }

    // There doesn't seem to be a way to get the node pointer, so copy it.
    nodes_.push_back(NodeInfo{*node, itOp - registrations_.begin()});
  }

  // g_loggedAllocations
  auto runtimeAllocations = g_loggedAllocations;
  ptrdiff_t minRuntimeOffset = 0;  // These are negative so zero start is fine.
  for (const auto &alloc : runtimeAllocations) {
    minRuntimeOffset = std::min(minRuntimeOffset, alloc.offset);
  }
  size_t totalRuntimeAllocSize = 0;
  for (const auto &alloc : runtimeAllocations) {
    // TODO: This drops the alignment between buffers. Is this fine?
    totalRuntimeAllocSize += alloc.len;
    ptrdiff_t offset = alloc.offset - minRuntimeOffset + ramTensorBufferSize;
    memMap_.recordRAM(offset, alloc.len,
                      "PersistentBuf" + std::to_string(alloc.nodeIndex));
  }

  // This includes:
  // - Tensors
  // - Scratch buffers
  // - Persistent buffers
  arenaBufferSize_ = ramTensorBufferSize + totalRuntimeAllocSize;

  if (debugPrint_) {
    memMap_.report();
  }

  return true;
}

void tflmc::Compiler::writeSource(std::ostream &out) {
  CodeWriter wr(out, subgraph_);

  wr << R"(

#include "../../api/xcore_config.h"
#include "lib_nn/api/version.h"
#include "lib_tflite_micro/api/version.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/kernels/reduce.h"
#include "tensorflow/lite/micro/kernels/softmax.h"
#include "tensorflow/lite/micro/micro_context.h"

#if defined __GNUC__
#define ALIGN(X) __attribute__((aligned(X)))
#elif defined _MSC_VER
#define ALIGN(X) __declspec(align(X))
#elif defined __TASKING__
#define ALIGN(X) __align(X)
#endif
)";

  if (sharedCfg_) {
    wr << R"(
// Check lib_nn and lib_tflite_micro versions
// NOTE: xformer version is saved for debugging purposes
// If lib_nn and lib_tflite_micro versions are as expected,
// then the xformer version doesn't matter as the model should execute
// If major version is zero, then minor versions must match
// Otherwise, major versions must match and binary minor version
// must be less or equal to runtime minor version
// Check if runtime lib_tflite_micro version matches with compiled version
static_assert(()"
       << sharedCfg_->lib_tflite_micro_major_version
       << R"( == 0 && lib_tflite_micro::major_version == 0 && )"
       << sharedCfg_->lib_tflite_micro_minor_version
       << R"( == lib_tflite_micro::minor_version) ||
              ()"
       << sharedCfg_->lib_tflite_micro_major_version
       << R"( == lib_tflite_micro::major_version) ||
              ()"
       << sharedCfg_->lib_tflite_micro_minor_version
       << R"(  < lib_tflite_micro::minor_version),
             "Model has been compiled with lib_tflite_micro version incompatible with runtime lib_tflite_micro version!");

// Check if runtime lib_nn version matches with compiled version
static_assert(()"
       << sharedCfg_->lib_nn_major_version
       << R"( == 0 && lib_nn::major_version == 0 && )"
       << sharedCfg_->lib_nn_minor_version << R"( == lib_nn::minor_version) ||
              ()"
       << sharedCfg_->lib_nn_major_version << R"( == lib_nn::major_version) ||
              ()"
       << sharedCfg_->lib_nn_minor_version << R"(  < lib_nn::minor_version),
             "Model has been compiled with lib_nn version incompatible with runtime lib_nn version!");

)";
  }

  // declare custom registrations
  if (has_custom_ops) {
    wr << R"(namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
)";
    for (size_t i = 0; i < registrations_.size(); i++) {
      if (registrations_[i].code == tflite::BuiltinOperator_CUSTOM &&
          registrations_[i].custom_name != "TFLite_Detection_PostProcess") {
        wr << "extern TfLiteRegistration *Register_"
           << registrations_[i].custom_name << "(void);\n";
      }
    }
    wr << R"(} // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite

)";
  }
  if (has_tflite_custom_ops) {
    wr << R"(namespace tflite {
)";
    for (size_t i = 0; i < registrations_.size(); i++) {
      if (registrations_[i].code == tflite::BuiltinOperator_CUSTOM &&
          registrations_[i].custom_name == "TFLite_Detection_PostProcess") {
        wr << "extern TfLiteRegistration "
              "*Register_DETECTION_POSTPROCESS(void);\n";
      }
    }
    wr << R"(} // namespace tflite

)";
  }
  wr << R"(namespace {

constexpr int kTensorArenaSize = )"
     << arenaBufferSize_ << R"(;
uint8_t tensor_arena[kTensorArenaSize] ALIGN(8);
template <int SZ, class T> struct TfArray {
  int sz; T elem[SZ];
};
enum used_operators_e {
  )";
  for (size_t i = 0; i < registrations_.size(); i++) {
    if (registrations_[i].code == tflite::BuiltinOperator_CUSTOM) {
      wr << "OP_" << registrations_[i].custom_name << ", ";
    } else {
      wr << "OP_" << tflite::EnumNameBuiltinOperator(registrations_[i].code)
         << ", ";
    }
  }
  wr << R"( OP_LAST
};

#if defined(TFLMC_XCORE_PROFILE) || defined(TFLMC_PRINT_TENSORS)
const char *op_strs[] = {
)";
  for (size_t i = 0; i < registrations_.size(); i++) {
    if (registrations_[i].code == tflite::BuiltinOperator_CUSTOM) {
      wr << "\"OP_" << registrations_[i].custom_name << "\", ";
    } else {
      wr << "\"OP_" << tflite::EnumNameBuiltinOperator(registrations_[i].code)
         << "\", ";
    }
  }
  wr << R"(};
#endif

#ifdef TFLMC_XCORE_PROFILE
int op_times[OP_LAST];
int op_counts[OP_LAST];
int64_t op_times_summed;
int time_t0, time_t1;
#endif

TfLiteContext ctx{};

TfLiteRegistration registrations[OP_LAST];
)";
  for (size_t i = 0; i < tensors_.size(); i++) {
    auto &t = tensors_[i].tensor;
    if (t->allocation_type == kTfLiteMmapRo) {
      wr.writeTensor(*t, "tensor_data" + std::to_string(i));
    }
    wr.writeIntArray(*t->dims, "tensor_dimension" + std::to_string(i));
    wr.writeQuantization(t->quantization, "quant" + std::to_string(i));
#if TF_LITE_PACKED_QUANTIZED_DATA_VERSION
    wr.writeQuantizationDetails(t->quantization,
                                "quant_details" + std::to_string(i));
#endif
  }
  for (size_t i = 0; i < nodes_.size(); i++) {
    auto &node = nodes_[i].node;
    auto &regInfo = registrations_[nodes_[i].regIndex];
    if (regInfo.code == tflite::BuiltinOperator_CUSTOM) {
      wr << "uint8_t ALIGN(4) opdata" + std::to_string(i) << "["
         << node.custom_initial_data_size << "] = { ";
      for (int j = 0; j < node.custom_initial_data_size; ++j)
        wr << int(((uint8_t const *)node.custom_initial_data)[j]) << ", ";
      wr << " }; /* custom_initial_data */\n";
    } else {
      wr.writeBuiltin(regInfo.code, node.builtin_data,
                      "opdata" + std::to_string(i));
    }
    wr.writeIntArray(*node.inputs, "inputs" + std::to_string(i));
    wr.writeIntArray(*node.outputs, "outputs" + std::to_string(i));
  }

  wr << R"(TfLiteTensor tflTensors[] = {
)";
  for (size_t i = 0; i < tensors_.size(); i++) {
    auto &t = tensors_[i].tensor;
    wr << "  { ";

    if (t->allocation_type == kTfLiteMmapRo) {
      wr << "{(int32_t*)tensor_data" << i << "},";
    } else {
      wr << "{(int32_t*)(tensor_arena + "
         << ((uintptr_t)t->data.data - (uintptr_t)arena_buf_.data()) << ")},";
    }
    wr << "(TfLiteIntArray*)&tensor_dimension" << i << ", ";

    wr << tflmc::to_string(t->type) << ", ";

    if (has_quantization) {
      if (t->quantization.type == kTfLiteAffineQuantization) {
        wr << "{kTfLiteAffineQuantization, "
              "const_cast<void*>(static_cast<const void*>(&quant"
           << i << ")) }, {quant" << i << ".scale->data[0], quant" << i
           << ".zero_point->data[0] ";
      } else {
        wr << "{kTfLiteNoQuantization, nullptr }, {0,0";
      }
      wr << "},";
    }

    wr << t->bytes << ", ";

    wr << tflmc::to_string(t->allocation_type) << ", ";
    wr << t->is_variable << ", ";

    wr << "},\n";
  }
  wr << "};\n";

  wr << R"(TfLiteNode tflNodes[] = {
)";
  for (size_t i = 0; i < nodes_.size(); i++) {
    wr << "  { (TfLiteIntArray*)&inputs" << i << ", ";
    wr << "(TfLiteIntArray*)&outputs" << i << ", ";
    wr << "(TfLiteIntArray*)&inputs" << i << ", ";
    wr << "nullptr, ";
    // TODO: Is this cast safe or does the data need to be non-const?
    // CP: I think so (as it typically just carries the trained operator
    // parameters) CP: Also if it were written to, we would see a segfault
    // (write to text segment)
    if (nodes_[i].node.builtin_data || nodes_[i].node.custom_initial_data) {
      wr << "const_cast<void*>(static_cast<const void*>(&opdata" << i << ")), ";
    } else {
      wr << "nullptr, ";
    }
    wr << "nullptr, ";
    auto regI = nodes_[i].regIndex;
    if (registrations_[regI].code == tflite::BuiltinOperator_CUSTOM) {
      wr << nodes_[i].node.custom_initial_data_size << ", ";
    } else {
      wr << "0, ";
    }
    wr << "},\n";
  }
  wr << "};\n";

  wr << R"(used_operators_e used_ops[] = {
)";
  for (size_t i = 0; i < nodes_.size(); i++) {
    auto regI = nodes_[i].regIndex;
    if (registrations_[regI].code == tflite::BuiltinOperator_CUSTOM) {
      wr << "OP_" << registrations_[regI].custom_name << ", ";
    } else {
      wr << "OP_" << tflite::EnumNameBuiltinOperator(registrations_[regI].code)
         << ", ";
    }
  }
  wr << "};\n";

  // TODO: This code assumes that persistent allocations are made from the end
  // (which is true for the current implementation)
  wr << R"(

// Scratch buffer variables
int scratch_buffer_idx = 0;
const int scratch_buffer_offsets[)"
     << scratchBufferOffsets.size() << R"(] = { )";
  if (scratchBufferOffsets.size() > 0) {
    wr << scratchBufferOffsets[0];
    for (int i = 1; i < scratchBufferOffsets.size(); i++) {
      wr << ", " << scratchBufferOffsets[i];
    }
  }
  wr << R"( };
tflite::MicroContext mc;

// Xcore context and thread variables
xc_context_config_t xc_config;
constexpr int kStackWordsPerThread = 256;
constexpr int threadsStackSizeInUint64 = )"
     << numXCThreads_ << R"( * kStackWordsPerThread/2;
// We use uint64_t for xcThreadsStack so that it is aligned to 8 bytes
uint64_t xcThreadsStack[threadsStackSizeInUint64];

// Functions to be used as function pointers for TfLiteContext and MicroContext 
static void* AllocatePersistentBuffer(struct TfLiteContext* ctx,
                                                 size_t bytes) {
  static uint8_t *AllocPtr = tensor_arena + sizeof(tensor_arena);

  AllocPtr -= bytes;
  return AllocPtr;
}

static TfLiteEvalTensor *GetEvalTensor(const struct TfLiteContext *context,
                                       int tensor_idx) {
  return (TfLiteEvalTensor*)&tflTensors[tensor_idx];
}

static TfLiteStatus RequestScratchBufferInArena(struct TfLiteContext *context, size_t bytes,
                                       int *buffer_idx) {
  *buffer_idx = scratch_buffer_idx++;
  return kTfLiteOk;
};

static void *GetScratchBuffer(struct TfLiteContext *context,
                                       int buffer_idx) {
  return tensor_arena + scratch_buffer_offsets[buffer_idx];
}

static TfLiteTensor* AllocateTempInputTensor(const TfLiteNode* node, int index) {
      return &ctx.tensors[node->inputs->data[index]];
}

static TfLiteTensor* AllocateTempOutputTensor(const TfLiteNode* node, int index) {
      return &ctx.tensors[node->outputs->data[index]];
}

static void DeallocateTempTfLiteTensor(TfLiteTensor* tensor) {
}

static void* external_context() {
  return &xc_config;
}

} // namespace

TfLiteStatus )"
     << prefix_ << R"(init(void *flash_data) {
  // Set flash data in xcore context config
  xc_config.flash_data = flash_data;

  // Setup microcontext functions
  mc.AllocateTempInputTensor = &AllocateTempInputTensor;
  mc.AllocateTempOutputTensor = &AllocateTempOutputTensor;
  mc.DeallocateTempTfLiteTensor = &DeallocateTempTfLiteTensor;
  mc.external_context = &external_context;

  // Setup tflitecontext functions
  ctx.AllocatePersistentBuffer = &AllocatePersistentBuffer;
  ctx.GetEvalTensor = &GetEvalTensor;
  ctx.RequestScratchBufferInArena = &RequestScratchBufferInArena;
  ctx.GetScratchBuffer = &GetScratchBuffer;
  
  // Set microcontext as the context ptr
  ctx.impl_ = (void*)&mc;
  ctx.tensors = tflTensors;
)";
  wr << "  ctx.tensors_size = " << tensors_.size() << ";\n";

  for (size_t i = 0; i < registrations_.size(); i++) {
    std::string opName;
    if (registrations_[i].code == tflite::BuiltinOperator_CUSTOM) {
      opName = registrations_[i].custom_name;
      if (opName == "TFLite_Detection_PostProcess") {
        wr << "  registrations[OP_" << opName
           << "] = *(tflite::Register_DETECTION_POSTPROCESS());\n";
      } else {
        wr << "  registrations[OP_" << opName
           << "] = *(tflite::ops::micro::xcore::Register_" << opName
           << "());\n";
      }
    } else if ((registrations_[i].code == tflite::BuiltinOperator_ADD) ||
               (registrations_[i].code ==
                tflite::BuiltinOperator_AVERAGE_POOL_2D) ||
               (registrations_[i].code == tflite::BuiltinOperator_CONV_2D) ||
               (registrations_[i].code ==
                tflite::BuiltinOperator_DEPTHWISE_CONV_2D) ||
               (registrations_[i].code == tflite::BuiltinOperator_DEQUANTIZE) ||
               (registrations_[i].code ==
                tflite::BuiltinOperator_FULLY_CONNECTED) ||
               (registrations_[i].code == tflite::BuiltinOperator_LOGISTIC) ||
               (registrations_[i].code ==
                tflite::BuiltinOperator_MAX_POOL_2D) ||
               (registrations_[i].code == tflite::BuiltinOperator_MEAN) ||
               (registrations_[i].code == tflite::BuiltinOperator_MUL) ||
               (registrations_[i].code == tflite::BuiltinOperator_PRELU) ||
               (registrations_[i].code == tflite::BuiltinOperator_QUANTIZE) ||
               (registrations_[i].code == tflite::BuiltinOperator_RELU) ||
               (registrations_[i].code == tflite::BuiltinOperator_SHAPE) ||
               (registrations_[i].code == tflite::BuiltinOperator_SOFTMAX) ||
               (registrations_[i].code ==
                tflite::BuiltinOperator_TRANSPOSE_CONV)) {
      opName = tflite::EnumNameBuiltinOperator(registrations_[i].code);
      wr << "  registrations[OP_" << opName << "] = tflite::Register_" << opName
         << "();\n";
    } else {
      opName = tflite::EnumNameBuiltinOperator(registrations_[i].code);
      wr << "  registrations[OP_" << opName
         << "] = tflite::ops::micro::Register_" << opName << "();\n";
    }
  }
  wr << "\n";
  wr << R"(
#ifdef TFLMC_XCORE_PROFILE
  printf("\nProfiling init()...");
  memset(op_times, 0, sizeof(op_times));
#endif

)";
  wr << "  for(size_t i = 0; i < " << nodes_.size() << R"(; ++i) {
    if (registrations[used_ops[i]].init) {

#ifdef TFLMC_XCORE_PROFILE
      asm volatile ("gettime %0" : "=r" (time_t0));
#endif

      tflNodes[i].user_data = registrations[used_ops[i]].init(&ctx, (const char*)tflNodes[i].builtin_data, )";
  wr << "tflNodes[i].custom_initial_data_size";
  wr << R"();

#ifdef TFLMC_XCORE_PROFILE
      asm volatile ("gettime %0" : "=r" (time_t1));
      op_times[used_ops[i]] += time_t1 - time_t0;
      printf("\nnode %-5d %-32s %-12d", i, op_strs[used_ops[i]], time_t1 - time_t0);
#endif

    }
  }

#ifdef TFLMC_XCORE_PROFILE
    printf("\n\nCumulative times for init()...");
    for(int i=0; i<OP_LAST; i++){
      printf("\n%-32s %-12d", op_strs[i], op_times[i]);
    }
  printf("\n");
  printf("\nProfiling prepare()...");
  memset(op_times, 0, sizeof(op_times));
#endif

)";
  wr << "  for(size_t i = 0; i < " << nodes_.size() << R"(; ++i) {
    if (registrations[used_ops[i]].prepare) {

#ifdef TFLMC_XCORE_PROFILE
      asm volatile ("gettime %0" : "=r" (time_t0));
#endif

      TfLiteStatus status = registrations[used_ops[i]].prepare(&ctx, &tflNodes[i]);

#ifdef TFLMC_XCORE_PROFILE
      asm volatile ("gettime %0" : "=r" (time_t1));
      op_times[used_ops[i]] += time_t1 - time_t0;
      printf("\nnode %-5d %-32s %-12d", i, op_strs[used_ops[i]], time_t1 - time_t0);
#endif

      if (status != kTfLiteOk) {
        return status;
      }
    }
  }

#ifdef TFLMC_XCORE_PROFILE
    printf("\n\nCumulative times for prepare()...");
    for(int i=0; i<OP_LAST; i++){
      printf("\n%-32s %-12d", op_strs[i], op_times[i]);
    }
  printf("\n");
#endif

  return kTfLiteOk;
}

static const int inTensorIndices[] = {
  )";
  for (auto inIndex : inputTensorIndices_) {
    out << inIndex << ", ";
  }
  out << R"(
};
TfLiteTensor* )"
      << prefix_ << R"(input(int index) {
  return &ctx.tensors[inTensorIndices[index]];
}

static const int outTensorIndices[] = {
  )";  // TODO: perhaps use a smaller type than int?
  for (auto outIndex : outputTensorIndices_) {
    out << outIndex << ", ";
  }
  out << R"(
};
TfLiteTensor* )"
      << prefix_ << R"(output(int index) {
  return &ctx.tensors[outTensorIndices[index]];
}

#ifdef TFLMC_PRINT_TENSORS
unsigned char checksum(char *data, unsigned int length)
{
  static char sum;
  static char * end;
  sum = 0;
  end = data + length;

  do
  {
      sum -= *data++;
  } while (data != end);
  return sum;
}
#endif

TfLiteStatus )"
      << prefix_ << R"(invoke() {
  xc_config.thread_info.nstackwords = kStackWordsPerThread;
  xc_config.thread_info.stacks = &xcThreadsStack[threadsStackSizeInUint64 - 1];
  thread_init_)"<< numXCThreads_ <<R"((&xc_config.thread_info);

#ifdef TFLMC_XCORE_PROFILE
  printf("\nProfiling invoke()...");
  memset(op_times, 0, sizeof(op_times));
  memset(op_counts, 0, sizeof(op_counts));
  op_times_summed = 0;
#endif

#ifdef TFLMC_PRINT_TENSORS
printf("[\n");
#endif

  for(size_t i = 0; i < )"
      << nodes_.size() << R"(; ++i) {

#ifdef TFLMC_PRINT_INPUT_TENSORS
    // print every input tensor
    printf("\nnode in %d", i);
    for (int j=0; j<tflNodes[i].inputs->size; j++){
      printf("\ntensor %d, input %d, %d bytes, checksum %d\n", tflNodes[i].inputs->data[j], j, tflTensors[tflNodes[i].inputs->data[j]].bytes, checksum(tflTensors[tflNodes[i].inputs->data[j]].data.raw, tflTensors[tflNodes[i].inputs->data[j]].bytes));
      for(int k=0; k<tflTensors[tflNodes[i].inputs->data[j]].bytes; k++){
        printf("%d,", (int8_t)tflTensors[tflNodes[i].inputs->data[j]].data.raw[k]);
      }
    }
    printf("\n");
#endif

#ifdef TFLMC_XCORE_PROFILE
  asm volatile ("gettime %0" : "=r" (time_t0));
#endif

    TfLiteStatus status = registrations[used_ops[i]].invoke(&ctx, &tflNodes[i]);

#ifdef TFLMC_XCORE_PROFILE
  asm volatile ("gettime %0" : "=r" (time_t1));
  op_times[used_ops[i]] += time_t1 - time_t0;
  op_counts[used_ops[i]] += 1;
  printf("\nnode %-5d %-32s %-12d", i, op_strs[used_ops[i]], time_t1 - time_t0);
#endif

#ifdef TFLMC_PRINT_TENSORS
    // print every output tensor
    printf("\n{\"node\" : \"%d\", \"op\" : \"%s\", \"data\" : [", i, op_strs[used_ops[i]]);
    for (int j=0; j<tflNodes[i].outputs->size; j++){
      printf("\n{\"tensor\" : %d, \"output\" : %d, \"bytes\" : %d, \"checksum\" : %d,\n", tflNodes[i].outputs->data[j], j, tflTensors[tflNodes[i].outputs->data[j]].bytes, checksum(tflTensors[tflNodes[i].outputs->data[j]].data.raw, tflTensors[tflNodes[i].outputs->data[j]].bytes));
      printf("\"val\" : [");
      for(int k=0; k<tflTensors[tflNodes[i].outputs->data[j]].bytes; k++){
        printf("%d", (int8_t)tflTensors[tflNodes[i].outputs->data[j]].data.raw[k]);
        if (k < tflTensors[tflNodes[i].outputs->data[j]].bytes-1){
          printf(",");
        }
      }
      if(j<tflNodes[i].outputs->size-1){
        printf("]},\n");
      } else {
        printf("]}]\n");
      }
    }

    if(i<)"
      << nodes_.size() << R"(-1){
      printf("},\n");
    } else {
      printf("}\n");
    }
#endif

    if (status != kTfLiteOk) {
      thread_destroy(&xc_config.thread_info);
      return status;
    }
  }
#ifdef TFLMC_PRINT_TENSORS
printf("\n]");
#endif

  thread_destroy(&xc_config.thread_info);

#ifdef TFLMC_XCORE_PROFILE
  struct convopdata{
    const char * name;
    size_t thread_count;
    int evalStartTime;
    int threadsStartTime;
    int threadsDoneTime;
  };
  int conv_times1 = 0, conv_times2 = 0;
  printf("\n\nConv()...");
  for(size_t i = 0; i < )" << nodes_.size() << R"(; ++i) {
    if(used_ops[i] == OP_XC_conv2d_v2) {
      auto *op_data = reinterpret_cast<convopdata *>(tflNodes[i].user_data);
      conv_times1 += op_data->threadsStartTime - op_data->evalStartTime;
      conv_times2 += op_data->threadsDoneTime - op_data->threadsStartTime;
      printf("\nnode %-5d %-25s %-25s %-6d %-6d %-12d", i, op_strs[used_ops[i]], op_data->name, op_data->thread_count, op_data->threadsStartTime - op_data->evalStartTime, op_data->threadsDoneTime - op_data->threadsStartTime);
    }
  }
  printf("\nSummed - %-10d %-10d", conv_times1, conv_times2);

  printf("\n\nCumulative times for invoke()...");
  for(int i=0; i<OP_LAST; i++){
    op_times_summed += op_times[i];
    printf("\n%-5d %-32s %-12d %dms", op_counts[i], op_strs[i], op_times[i], op_times[i]/100000);
  }
  printf("\n\nTotal time for invoke() - %-10lld %lldms\n\n", op_times_summed, op_times_summed/100000);
#endif

  return kTfLiteOk;
}
)";
}

void tflmc::Compiler::writeHeader(std::ostream &out) {
  tflmc::CodeWriter wr(out, subgraph_);

  std::string code = R"(
#ifndef %PREFIX%GEN_H
#define %PREFIX%GEN_H

#include "tensorflow/lite/c/common.h"

// Sets up the model with init and prepare steps.
TfLiteStatus %PREFIX%init(void *flash_data = nullptr);
// Returns the input tensor with the given index.
TfLiteTensor *%PREFIX%input(int index);
// Returns the output tensor with the given index.
TfLiteTensor *%PREFIX%output(int index);
// Runs inference for the model.
TfLiteStatus %PREFIX%invoke();

// Returns the number of input tensors.
inline size_t %PREFIX%inputs() {
  return )" + std::to_string(inputTensorIndices_.size()) +
                     R"(;
}
// Returns the number of output tensors.
inline size_t %PREFIX%outputs() {
  return )" + std::to_string(outputTensorIndices_.size()) +
                     R"(;
}

inline void *%PREFIX%input_ptr(int index) {
  return %PREFIX%input(index)->data.data;
}
inline size_t %PREFIX%input_size(int index) {
  return %PREFIX%input(index)->bytes;
}
inline int %PREFIX%input_dims_len(int index) {
  return %PREFIX%input(index)->dims->data[0];
}
inline int *%PREFIX%input_dims(int index) {
  return &%PREFIX%input(index)->dims->data[1];
}

inline void *%PREFIX%output_ptr(int index) {
  return %PREFIX%output(index)->data.data;
}
inline size_t %PREFIX%output_size(int index) {
  return %PREFIX%output(index)->bytes;
}
inline int %PREFIX%output_dims_len(int index) {
  return %PREFIX%output(index)->dims->data[0];
}
inline int *%PREFIX%output_dims(int index) {
  return &%PREFIX%output(index)->dims->data[1];
}

#endif
)";

  static std::regex rePrefix("%PREFIX%");
  code = std::regex_replace(code, rePrefix, prefix_);

  wr << code;
}

std::string tflmc::Compiler::getTensorName(int tensorIndex) const {
  auto tensor = GetTensor(interpreter_.get(), tensorIndex);

  std::stringstream ss;
  ss << (tensor->allocation_type == kTfLiteMmapRo ? "ROM" : "RAM") << "Tensor_";

  auto nOps = interpreter_->operators_size();
  for (size_t i = 0; i < nOps; i++) {
    auto nodeAndReg = interpreter_->node_and_registration(i);
    auto node = &nodeAndReg.node;

    auto checkAndAdd = [&](const TfLiteIntArray *indices,
                           const std::string &tag) {
      if (indices) {
        for (int k = 0; k < indices->size; k++) {
          if (indices->data[k] == tensorIndex) {
            ss << "L" << i << tag;
          }
        }
      }
    };

    checkAndAdd(node->inputs, "in");
    checkAndAdd(node->outputs, "out");
    //  checkAndAdd(node->intermediates, "int");
    //  checkAndAdd(node->temporaries, "tmp");
  }

  return ss.str();
}
