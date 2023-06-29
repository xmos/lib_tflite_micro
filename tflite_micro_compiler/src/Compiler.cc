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

bool debugPrint_ = false;

#define DEBUG_LOG(x) if(debugPrint_){printf x;}

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

TfLiteTensor *tflmc::GetTensor(tflite::MicroInterpreter *interpreter, int i, int sg) {
  auto ctx = &interpreter->context_;
  return tflite::MicroContextGetTensor(ctx, i, sg);
}

TfLiteEvalTensor *tflmc::GetEvalTensor(tflite::MicroInterpreter *interpreter, int i, int sg) {
  auto ctx = &interpreter->context_;
  return tflite::MicroContextGetEvalTensor(ctx, i, sg);
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
    : sharedCfg_(sharedCfg), prefix_(prefix) {
  debugPrint_ = debugPrint;
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
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model_->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }

  auto subgraphs = model_->subgraphs();
  tensors_.resize(subgraphs->size());
  nodes_.resize(subgraphs->size());
  inputTensorIndices_.resize(subgraphs->size());
  outputTensorIndices_.resize(subgraphs->size());

  mainGraph_ = (*subgraphs)[0];
  if (mainGraph_->inputs()->size() == 0 || mainGraph_->outputs()->size() == 0) {
    std::cerr << "No inputs or no outputs found in model\n";
    return false;
  }
  for(int g = 0; g < subgraphs->size(); g++) {
    auto sg = (*subgraphs)[g];
    for (auto inIndex : *sg->inputs()) {
      inputTensorIndices_[g].push_back(inIndex);
    }
    for (auto outIndex : *sg->outputs()) {
      outputTensorIndices_[g].push_back(outIndex);
    }
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
                                   arena_buf_.size()));

  TfLiteStatus set_external_context_status =
      interpreter_->SetMicroExternalContext((void *)&g_xc_config);
  if (set_external_context_status != kTfLiteOk) {
    MicroPrintf("SetExternalContext() failed");
    return false;
  }

  interpreter_->context_.AllocatePersistentBuffer =
       &LoggingAllocatePersistentBuffer;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter_->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return false;
  }

  // Get all tensors using dummy interpreter.
  // Calling GetTensor() allocates temp TfLiteTensor and trashes 
  // the arena as it is not meant to be called after AllocateTensors().
  std::vector<uint8_t> dummy_arena;
  dummy_arena.resize(SUFFICIENT_ARENA_SIZE);
  auto dummy_interpreter = std::unique_ptr<tflite::MicroInterpreter>(
      new tflite::MicroInterpreter(model_, resolver_, dummy_arena.data(),
                                   dummy_arena.size()));
  for(int g = 0; g < subgraphs->size(); g++) {
    auto sg = (*subgraphs)[g];
    for(int i = 0; i < sg->tensors()->size(); i++) {
      auto tensor = GetTensor(dummy_interpreter.get(), i, g);
      tensors_[g].push_back(tensor);
    }
  }

  // Iterate through all subgraphs and find allocated offsets for tensors
  ptrdiff_t ramTensorBufferSize = 0;
  ptrdiff_t romOffset = 0;
  DEBUG_LOG(("\n\nTFLMC Allocated offsets:\n"));
  for(int g=0; g<subgraphs->size(); g++) {
    auto sg = (*subgraphs)[g];
  for (size_t i = 0; i < sg->tensors()->size(); i++) {
    auto tensor = tensors_[g][i].tensor;
    if (tensor->allocation_type == kTfLiteMmapRo) {
      memMap_.recordROM(romOffset, tensor->bytes, getTensorName(i, g));
      DEBUG_LOG(("-1,"));
      romOffset += tensor->bytes;
    } else {
      auto t = GetEvalTensor(interpreter_.get(), i, g);
      ptrdiff_t offset = (uint8_t *)t->data.data - arena_buf_.data();
      DEBUG_LOG(("%d,", offset));
      // double word align tensor bytes
      int aligned_bytes = ((tensor->bytes + 7) / 8) * 8;
      ptrdiff_t highSize = offset + aligned_bytes;
      ramTensorBufferSize = std::max(ramTensorBufferSize, highSize);
      memMap_.recordRAM(offset, tensor->bytes, getTensorName(i, g));
    }
    // determine whether we need to individually set these properties for each
    // tensor
    if ((!has_quantization) &&
        tensor->quantization.type != kTfLiteNoQuantization) {
      has_quantization = true;
    }
  }
  DEBUG_LOG(("\n\n"));

  for (size_t i = 0; i < interpreter_->operators_size(g); i++) {
    auto nodeAndReg = interpreter_->node_and_registration(i, g);
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
      } else if (regInfo.custom_name == "XC_conv2d_v2") {
        has_xc_conv_ops = true;
      }
      has_custom_ops = true;
    }
    auto itOp =
        std::find(registrations_.begin(), registrations_.end(), regInfo);
    if (itOp == registrations_.end()) {
      itOp = registrations_.insert(registrations_.end(), regInfo);
    }

    // There doesn't seem to be a way to get the node pointer, so copy it.
    nodes_[g].push_back(NodeInfo{*node, itOp - registrations_.begin()});
  }
}

  // scratch buffers
  for (size_t k = 0; k < interpreter_->allocator_.GetScratchBufferRequestCount();
       k++) {
    void *data = interpreter_->micro_context_.GetScratchBuffer(k);
    ptrdiff_t offset = (uint8_t *)data - arena_buf_.data();
    tflite::internal::ScratchBufferRequest *requests =
        interpreter_->allocator_.GetScratchBufferRequests();
    int bytes = requests[k].bytes;
    // double word align scratch buffer bytes
    bytes = ((bytes + 7) / 8) * 8;
    ptrdiff_t highSize = offset + bytes;
    ramTensorBufferSize = std::max(ramTensorBufferSize, highSize);
    memMap_.recordRAM(offset, bytes,
                      "Scratch_idx" + std::to_string((int)k) + "_op" +
                          std::to_string((int)requests[k].node_idx));
    scratchBufferOffsets.push_back(offset);
  }

  // g_loggedAllocations for persistent buffers
  auto runtimeAllocations = g_loggedAllocations;
  ptrdiff_t minRuntimeOffset = 0;  // These are negative so zero start is fine.
  for (const auto &alloc : runtimeAllocations) {
    minRuntimeOffset = std::min(minRuntimeOffset, alloc.offset);
  }
  size_t totalRuntimeAllocSize = 0;
  for (const auto &alloc : runtimeAllocations) {
    int doubleAlignedSize = ((alloc.len + 7) / 8) * 8;
    totalRuntimeAllocSize += doubleAlignedSize;
    ptrdiff_t offset = alloc.offset - minRuntimeOffset + ramTensorBufferSize;
    memMap_.recordRAM(offset, doubleAlignedSize,
                      "PersistentBuf" + std::to_string(alloc.nodeIndex));
  }

  // This includes:
  // - Tensors
  // - Scratch buffers
  // - Persistent buffers
  arenaBufferSize_ = ramTensorBufferSize + totalRuntimeAllocSize;

  if (debugPrint_) {
    interpreter_->allocator_.memory_planner()->PrintMemoryPlan();
    memMap_.report();
  }

  return true;
}

void tflmc::Compiler::writeSource(std::ostream &out) {
  CodeWriter wr(out, mainGraph_);

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

// #define TFLMC_XCORE_PROFILE
// #define TFLMC_PRINT_TENSORS
// #define TFLMC_PRINT_INPUT_TENSORS
// #define SHARED_MEMORY_ARENA

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
        wr << "extern TfLiteRegistration_V1 *Register_"
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
        wr << "extern TfLiteRegistration_V1 "
              "*Register_DETECTION_POSTPROCESS(void);\n";
      }
    }
    wr << R"(} // namespace tflite

)";
  }
  wr << R"(

constexpr int kTensorArenaSize = )"
     << arenaBufferSize_ << R"(;
#ifndef SHARED_MEMORY_ARENA
namespace {
uint8_t tensor_arena[kTensorArenaSize] ALIGN(8);
}
#else
extern uint8_t tensor_arena[];
#endif

namespace {
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

#if defined(TFLMC_XCORE_PROFILE) || defined(TFLMC_PRINT_TENSORS) || defined(TFLMC_PRINT_INPUT_TENSORS)
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

#ifdef TFLMC_XCORE_PROFILE
int op_times[OP_LAST];
int op_counts[OP_LAST];
int64_t op_times_summed;
int time_t0, time_t1;
#endif

TfLiteContext ctx{};

TfLiteRegistration_V1 registrations[OP_LAST];
)";
for(size_t g = 0; g < tensors_.size(); g++) {
wr << R"(
struct {
)";
  for (size_t i = 0; i < tensors_[g].size(); i++) {
    auto &t = tensors_[g][i].tensor;
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
  for (size_t i = 0; i < nodes_[g].size(); i++) {
    auto &node = nodes_[g][i].node;
    auto &regInfo = registrations_[nodes_[g][i].regIndex];
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
wr << R"(} g)"<< g <<R"(;
)";
}

  wr << R"(
TfLiteTensor tflTensors[] = 
{)";
for(size_t g = 0; g < tensors_.size(); g++) {
  for (size_t i = 0; i < tensors_[g].size(); i++) {
    auto &t = tensors_[g][i].tensor;
    auto tEval = GetEvalTensor(interpreter_.get(), i, g);
    wr << "{ ";

    if (t->allocation_type == kTfLiteMmapRo) {
      wr << "{(int32_t*)g"<<g<<".tensor_data" << i << "},";
    } else {
      wr << "{(int32_t*)(tensor_arena + "
         << ((uintptr_t)tEval->data.data - (uintptr_t)arena_buf_.data()) << ")},";
    }
    wr << "(TfLiteIntArray*)&g"<<g<<".tensor_dimension" << i << ", ";

    wr << tflmc::to_string(t->type) << ", ";

    if (has_quantization) {
      if (t->quantization.type == kTfLiteAffineQuantization) {
        wr << "{kTfLiteAffineQuantization, "
              "const_cast<void*>(static_cast<const void*>(&g"<<g<<".quant"
           << i << ")) }, {g"<<g<<".quant" << i << ".scale->data[0], g"<<g<<".quant" << i
           << ".zero_point->data[0] ";
      } else {
        wr << "{kTfLiteNoQuantization, nullptr }, {0,0";
      }
      wr << "},";
    } else {
      wr << "{kTfLiteNoQuantization, nullptr }, {0,0},";
    }

    wr << t->bytes << ", ";

    wr << tflmc::to_string(t->allocation_type) << ", ";
    wr << t->is_variable << ", ";

    wr << "},\n";
  }
}
  wr << "};\n";

  wr << R"(
TfLiteNode tflNodes[] = 
{)";
for(size_t g = 0; g < tensors_.size(); g++) {
  for (size_t i = 0; i < nodes_[g].size(); i++) {
    wr << "{ (TfLiteIntArray*)&g"<<g<<".inputs" << i << ", ";
    wr << "(TfLiteIntArray*)&g"<<g<<".outputs" << i << ", ";
    wr << "(TfLiteIntArray*)&g"<<g<<".inputs" << i << ", ";
    wr << "nullptr, ";
    // TODO: Is this cast safe or does the data need to be non-const?
    // CP: I think so (as it typically just carries the trained operator
    // parameters) CP: Also if it were written to, we would see a segfault
    // (write to text segment)
    if (nodes_[g][i].node.builtin_data || nodes_[g][i].node.custom_initial_data) {
      wr << "const_cast<void*>(static_cast<const void*>(&g"<<g<<".opdata" << i << ")), ";
    } else {
      wr << "nullptr, ";
    }
    wr << "nullptr, ";
    auto regI = nodes_[g][i].regIndex;
    if (registrations_[regI].code == tflite::BuiltinOperator_CUSTOM) {
      wr << nodes_[g][i].node.custom_initial_data_size << ", ";
    } else {
      wr << "0, ";
    }
    wr << "},\n";
  }
}
wr << "};\n";

wr << R"(
used_operators_e used_ops[] =
{)";
for(size_t g = 0; g < tensors_.size(); g++) {
  for (size_t i = 0; i < nodes_[g].size(); i++) {
    auto regI = nodes_[g][i].regIndex;
    if (registrations_[regI].code == tflite::BuiltinOperator_CUSTOM) {
      wr << "OP_" << registrations_[regI].custom_name << ", ";
    } else {
      wr << "OP_" << tflite::EnumNameBuiltinOperator(registrations_[regI].code)
         << ", ";
    }
  }
}
  wr << "};\n\n";

wr << "\n// Indices into tflTensors and tflNodes for subgraphs";
wr << "\nsize_t tflTensors_subgraph_index[] = {0, ";
int index = 0;
for(size_t g = 0; g < tensors_.size(); g++) {
  index += tensors_[g].size();
  wr << index << ", ";
}
wr << "};";

wr << "\nsize_t tflNodes_subgraph_index[] = {0, ";
index = 0;
for(size_t g = 0; g < tensors_.size(); g++) {
  index += nodes_[g].size();
  wr << index << ", ";
}
wr << "};\n";

wr << R"(static const int inTensorIndices[] = {
  )";
  for(size_t g = 0; g < tensors_.size(); g++) {
    for (auto inIndex : inputTensorIndices_[g]) {
      wr << inIndex << ", ";
    }
  }
  wr << R"(
};

static const int outTensorIndices[] = {
  )";  // TODO: perhaps use a smaller type than int?
  for(size_t g = 0; g < tensors_.size(); g++) {
    for (auto outIndex : outputTensorIndices_[g]) {
      wr << outIndex << ", ";
    }
  }
  wr << R"(
};
)";

wr << "\n// Indices into inTensors and outTensors for subgraphs";
wr << "\nsize_t inTensors_subgraph_index[] = {0, ";
index = 0;
for(size_t g = 0; g < tensors_.size(); g++) {
  index += inputTensorIndices_[g].size();
  wr << index << ", ";
}
wr << "};";

wr << "\nsize_t outTensors_subgraph_index[] = {0, ";
index = 0;
for(size_t g = 0; g < tensors_.size(); g++) {
  index += outputTensorIndices_[g].size();
  wr << index << ", ";
}
wr << "};";

  // TODO: This code assumes that persistent allocations are made from the end
  // (which is true for the current implementation)
  wr << R"(

// Scratch buffer variables
int scratch_buffer_idx;
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
tflite::MicroGraph micro_graph;
size_t currentSubgraphIndex = 0;

// Xcore context and thread variables
xc_context_config_t xc_config;
// When using USE_DDR_FIX for enabling LPDDR support, only one thread can be used
#ifdef USE_DDR_FIX
static_assert(()"
       << numXCThreads_
       << R"( == 1),
             "Only one thread can be used when using USE_DDR_FIX! Please recompile with one thread!");
#endif
constexpr int kStackWordsPerThread = 256;
constexpr int threadsStackSizeInUint64 = )"
     << numXCThreads_ << R"( * kStackWordsPerThread/2;
// We use uint64_t for xcThreadsStack so that it is aligned to 8 bytes
uint64_t xcThreadsStack[threadsStackSizeInUint64];

// Persistent buffer ptr
// Initialized to the tail end of the tensor arena
uint8_t *persistentBufferPtr;
// Functions to be used as function pointers for TfLiteContext and MicroContext 
static void* AllocatePersistentBuffer(struct TfLiteContext* ctx,
                                                 size_t bytes) {
  // Align to double word
  bytes = ((bytes + 7) / 8) * 8;
  persistentBufferPtr -= bytes;
  return persistentBufferPtr;
}

static TfLiteEvalTensor *GetEvalTensor(const struct TfLiteContext *context,
                                       int tensor_idx) {
  return (TfLiteEvalTensor*)&tflTensors[tflTensors_subgraph_index[currentSubgraphIndex] + tensor_idx];
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

static TfLiteTensor* mc_AllocateTempInputTensor(const TfLiteNode* node, int index) {
      return &ctx.tensors[tflTensors_subgraph_index[currentSubgraphIndex] + node->inputs->data[index]];
}

static TfLiteTensor* mc_AllocateTempOutputTensor(const TfLiteNode* node, int index) {
      return &ctx.tensors[tflTensors_subgraph_index[currentSubgraphIndex] + node->outputs->data[index]];
}

static void mc_DeallocateTempTfLiteTensor(TfLiteTensor* tensor) {
}

static void* mc_external_context() {
  return &xc_config;
}

static tflite::MicroGraph& mc_graph() {
  return micro_graph;
}

static int mg_NumSubgraphs(){
  return sizeof(tflTensors_subgraph_index)/sizeof(size_t) - 1;
}

static size_t mg_NumSubgraphInputs(int subgraph_idx){
  return inTensors_subgraph_index[subgraph_idx+1] - inTensors_subgraph_index[subgraph_idx];
}

static size_t mg_NumSubgraphOutputs(int subgraph_idx){
  return outTensors_subgraph_index[subgraph_idx+1] - outTensors_subgraph_index[subgraph_idx];
}

static TfLiteEvalTensor* mg_GetSubgraphInput(int subgraph_idx, int i){
  return (TfLiteEvalTensor*)&tflTensors[tflTensors_subgraph_index[subgraph_idx] + inTensorIndices[inTensors_subgraph_index[subgraph_idx] + i]];
}

static TfLiteEvalTensor* mg_GetSubgraphOutput(int subgraph_idx, int i){
  return (TfLiteEvalTensor*)&tflTensors[tflTensors_subgraph_index[subgraph_idx] + outTensorIndices[outTensors_subgraph_index[subgraph_idx] + i]];
}

static TfLiteStatus mg_InvokeSubgraph(int g){
  int prevSubgraphIndex = currentSubgraphIndex;
  currentSubgraphIndex = g;
#ifdef TFLMC_PRINT_TENSORS
printf("[\n");
#endif

  for(size_t i = tflNodes_subgraph_index[g]; i < tflNodes_subgraph_index[g+1]; ++i) {

#ifdef TFLMC_PRINT_INPUT_TENSORS
    // print every input tensor
    printf("\nnode in %d", i);
    for (int j=0; j<tflNodes[i].inputs->size; j++){
      // -1 such as in case of no bias tensor for conv
      if (tflNodes[i].inputs->data[j] != -1) {
        printf("\ntensor %d, input %d, %d bytes, checksum %d\n", tflNodes[i].inputs->data[j], j, tflTensors[tflNodes[i].inputs->data[j]].bytes, checksum(tflTensors[tflNodes[i].inputs->data[j]].data.raw, tflTensors[tflNodes[i].inputs->data[j]].bytes));
        for(int k=0; k<tflTensors[tflTensors_subgraph_index[g] + tflNodes[i].inputs->data[j]].bytes; k++){
          printf("%d,", (int8_t)tflTensors[tflTensors_subgraph_index[g] + tflNodes[i].inputs->data[j]].data.raw[k]);
        }
      }
    }
    printf("\n");
#endif

#ifdef TFLMC_XCORE_PROFILE
#ifdef __xcore__
  asm volatile ("gettime %0" : "=r" (time_t0));
#endif
#endif

    TfLiteStatus status = registrations[used_ops[i]].invoke(&ctx, &tflNodes[i]);

#ifdef TFLMC_XCORE_PROFILE
#ifdef __xcore__
  asm volatile ("gettime %0" : "=r" (time_t1));
#endif
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
      for(int k=0; k<tflTensors[tflTensors_subgraph_index[g] + tflNodes[i].outputs->data[j]].bytes; k++){
        printf("%d", (int8_t)tflTensors[tflTensors_subgraph_index[g] + tflNodes[i].outputs->data[j]].data.raw[k]);
        if (k < tflTensors[tflTensors_subgraph_index[g] + tflNodes[i].outputs->data[j]].bytes-1){
          printf(",");
        }
      }
      if(j<tflNodes[i].outputs->size-1){
        printf("]},\n");
      } else {
        printf("]}]\n");
      }
    }

    if(i < ((tflNodes_subgraph_index[g+1] - tflNodes_subgraph_index[g]) - 1)){
      printf("},\n");
    } else {
      printf("}\n");
    }
#endif

    if (status != kTfLiteOk) {
      currentSubgraphIndex = prevSubgraphIndex;
      return status;
    }
  }
#ifdef TFLMC_PRINT_TENSORS
printf("\n]");
#endif

  currentSubgraphIndex = prevSubgraphIndex;
  return kTfLiteOk;
}

} // namespace

TfLiteTensor* )"
      << prefix_ << R"(input(int index) {
  return &ctx.tensors[inTensorIndices[index]];
}

TfLiteTensor* )"
      << prefix_ << R"(output(int index) {
  return &ctx.tensors[outTensorIndices[index]];
}

TfLiteStatus )"
     << prefix_ << R"(init(void *flash_data) {
  // Clear and initialize
  scratch_buffer_idx = 0;
  persistentBufferPtr = tensor_arena + kTensorArenaSize;

  // Set flash data in xcore context config
  xc_config.flash_data = flash_data;

  // Setup microcontext functions
  mc.AllocateTempInputTensor = &mc_AllocateTempInputTensor;
  mc.AllocateTempOutputTensor = &mc_AllocateTempOutputTensor;
  mc.DeallocateTempTfLiteTensor = &mc_DeallocateTempTfLiteTensor;
  mc.external_context = &mc_external_context;
  mc.graph = &mc_graph;

  micro_graph.NumSubgraphs = &mg_NumSubgraphs;
  micro_graph.NumSubgraphInputs = &mg_NumSubgraphInputs;
  micro_graph.NumSubgraphOutputs = &mg_NumSubgraphOutputs;
  micro_graph.GetSubgraphInput = &mg_GetSubgraphInput;
  micro_graph.GetSubgraphOutput = &mg_GetSubgraphOutput;
  micro_graph.InvokeSubgraph = &mg_InvokeSubgraph;

  // Setup tflitecontext functions
  ctx.AllocatePersistentBuffer = &AllocatePersistentBuffer;
  ctx.GetEvalTensor = &GetEvalTensor;
  ctx.RequestScratchBufferInArena = &RequestScratchBufferInArena;
  ctx.GetScratchBuffer = &GetScratchBuffer;
  
  // Set microcontext as the context ptr
  ctx.impl_ = (void*)&mc;
  ctx.tensors = tflTensors;
)";
  wr << "  ctx.tensors_size = " << tensors_[0].size() << ";\n";

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
    } else if ((registrations_[i].code == tflite::BuiltinOperator_RESHAPE) ||
               (registrations_[i].code == tflite::BuiltinOperator_ROUND)) {
      opName = tflite::EnumNameBuiltinOperator(registrations_[i].code);
      wr << "  registrations[OP_" << opName
         << "] = tflite::ops::micro::Register_" << opName << "();\n";
    } else {
      opName = tflite::EnumNameBuiltinOperator(registrations_[i].code);
      wr << "  registrations[OP_" << opName << "] = tflite::Register_" << opName
         << "();\n";
    }
  }
  wr << "\n";
  wr << R"(
#ifdef TFLMC_XCORE_PROFILE
  printf("\nProfiling init()...");
  memset(op_times, 0, sizeof(op_times));
  op_times_summed = 0;
#endif

)";
  wr << "  for(size_t g = 0; g < " << nodes_.size() << R"(; ++g) {
    currentSubgraphIndex = g;
    for(size_t i = tflNodes_subgraph_index[g]; i < tflNodes_subgraph_index[g+1]; ++i) {
    if (registrations[used_ops[i]].init) {

#ifdef TFLMC_XCORE_PROFILE
#ifdef __xcore__
      asm volatile ("gettime %0" : "=r" (time_t0));
#endif
#endif

      tflNodes[i].user_data = registrations[used_ops[i]].init(&ctx, (const char*)tflNodes[i].builtin_data, )";
  wr << "tflNodes[i].custom_initial_data_size";
  wr << R"();

#ifdef TFLMC_XCORE_PROFILE
#ifdef __xcore__
      asm volatile ("gettime %0" : "=r" (time_t1));
#endif
      op_times[used_ops[i]] += time_t1 - time_t0;
      printf("\nnode %-5d %-32s %-12d", i, op_strs[used_ops[i]], time_t1 - time_t0);
#endif

    }
  }
  }
  currentSubgraphIndex = 0;

#ifdef TFLMC_XCORE_PROFILE
    printf("\n\nCumulative times for init()...");
    for(int i=0; i<OP_LAST; i++){
      op_times_summed += op_times[i];
      printf("\n%-32s %-12d %.2fms", op_strs[i], op_times[i], op_times[i]/100000.0);
    }
    printf("\n\nTotal time for init() - %-10lld %.2fms\n\n", op_times_summed, op_times_summed/100000.0);
  printf("\n");
  printf("\nProfiling prepare()...");
  memset(op_times, 0, sizeof(op_times));
  op_times_summed = 0;
#endif

)";
  wr << "  for(size_t g = 0; g < " << nodes_.size() << R"(; ++g) {
        currentSubgraphIndex = g;
        for(size_t i = tflNodes_subgraph_index[g]; i < tflNodes_subgraph_index[g+1]; ++i) {
    if (registrations[used_ops[i]].prepare) {

#ifdef TFLMC_XCORE_PROFILE
#ifdef __xcore__
      asm volatile ("gettime %0" : "=r" (time_t0));
#endif
#endif

      TfLiteStatus status = registrations[used_ops[i]].prepare(&ctx, &tflNodes[i]);

#ifdef TFLMC_XCORE_PROFILE
#ifdef __xcore__
      asm volatile ("gettime %0" : "=r" (time_t1));
#endif
      op_times[used_ops[i]] += time_t1 - time_t0;
      printf("\nnode %-5d %-32s %-12d", i, op_strs[used_ops[i]], time_t1 - time_t0);
#endif

      if (status != kTfLiteOk) {
        return status;
      }
    }
  }
  }
  currentSubgraphIndex = 0;

#ifdef TFLMC_XCORE_PROFILE
printf("\n\nCumulative times for prepare()...");
    for(int i=0; i<OP_LAST; i++){
      op_times_summed += op_times[i];
      printf("\n%-32s %-12d %.2fms", op_strs[i], op_times[i], op_times[i]/100000.0);
    }
    printf("\n\nTotal time for prepare() - %-10lld %.2fms\n\n", op_times_summed, op_times_summed/100000.0);
  printf("\n");
#endif

  return kTfLiteOk;
}

TfLiteStatus )"
      << prefix_ << R"(invoke() {
  xc_config.thread_info.nstackwords = kStackWordsPerThread;
  xc_config.thread_info.stacks = &xcThreadsStack[threadsStackSizeInUint64 - 1];
  thread_init_)"
      << numXCThreads_ << R"((&xc_config.thread_info);

#ifdef TFLMC_XCORE_PROFILE
  printf("\nProfiling invoke()...");
  memset(op_times, 0, sizeof(op_times));
  memset(op_counts, 0, sizeof(op_counts));
  op_times_summed = 0;
#endif

  mg_InvokeSubgraph(0);

  thread_destroy(&xc_config.thread_info);

#ifdef TFLMC_XCORE_PROFILE
  struct convopdata{
    const char * name;
    size_t thread_count;
    int evalStartTime;
    int threadsStartTime;
    int threadsDoneTime;
  };)";
  if (has_xc_conv_ops) {
    wr << R"(int conv_times1 = 0, conv_times2 = 0;
    printf("\n\nConv()...");
    for(size_t g = 0; g < )" << nodes_.size() << R"(; ++g) {
      for(size_t i = tflNodes_subgraph_index[g]; i < tflNodes_subgraph_index[g+1]; ++i) {
        if(used_ops[i] == OP_XC_conv2d_v2) {
          auto *op_data = reinterpret_cast<convopdata *>(tflNodes[i].user_data);
          conv_times1 += op_data->threadsStartTime - op_data->evalStartTime;
          conv_times2 += op_data->threadsDoneTime - op_data->threadsStartTime;
          printf("\nnode %-5d %-25s %-25s %-6d %-6d %-12d", i, op_strs[used_ops[i]], op_data->name, op_data->thread_count, op_data->threadsStartTime - op_data->evalStartTime, op_data->threadsDoneTime - op_data->threadsStartTime);
        }
      }
    }
    printf("\nSummed - %-10d %-10d", conv_times1, conv_times2);
    )";
  }
  wr << R"(printf("\n\nCumulative times for invoke()...");
  for(int i=0; i<OP_LAST; i++){
    op_times_summed += op_times[i];
    printf("\n%-5d %-32s %-12d %.2fms", op_counts[i], op_strs[i], op_times[i], op_times[i]/100000.0);
  }
  printf("\n\nTotal time for invoke() - %-10lld %.2fms\n\n", op_times_summed, op_times_summed/100000.0);
#endif

  return kTfLiteOk;
}
)";
}

void tflmc::Compiler::writeHeader(std::ostream &out) {
  tflmc::CodeWriter wr(out, mainGraph_);

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
  return )" + std::to_string(inputTensorIndices_[0].size()) +
                     R"(;
}
// Returns the number of output tensors.
inline size_t %PREFIX%outputs() {
  return )" + std::to_string(outputTensorIndices_[0].size()) +
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

std::string tflmc::Compiler::getTensorName(int tensorIndex, int sg) const {
  auto tensor = tensors_[sg][tensorIndex].tensor;

  std::stringstream ss;
  ss << (tensor->allocation_type == kTfLiteMmapRo ? "ROM" : "RAM") << "Tensor_";

  for(size_t g = 0; g < tensors_.size(); g++){
    auto nOps = interpreter_->operators_size(g);
    for (size_t i = 0; i < nOps; i++) {
      auto nodeAndReg = interpreter_->node_and_registration(i, g);
      auto node = &nodeAndReg.node;

      auto checkAndAdd = [&](const TfLiteIntArray *indices,
                            const std::string &tag) {
        if (indices) {
          for (int k = 0; k < indices->size; k++) {
            if (indices->data[k] == tensorIndex) {
              ss << "L" << g << "_" << i << tag;
            }
          }
        }
      };

      checkAndAdd(node->inputs, "in");
      checkAndAdd(node->outputs, "out");
      //  checkAndAdd(node->intermediates, "int");
      //  checkAndAdd(node->temporaries, "tmp");
    }
  }

  return ss.str();
}
