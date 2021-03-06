#include <sstream>
#define private public
#include "tensorflow/lite/micro/micro_interpreter.h"
#undef private

#include "RecordAllocations.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "xcore_ops.h"
#include "xtflm_conf.h"

static std::vector<tflmc::Allocation> g_loggedAllocations;
static int g_currentNodeIndex = -1;
static uint8_t *g_arenaPtr = nullptr;
static ptrdiff_t g_arena_size = 0;
static size_t g_max_scratch_buffer_size = 0;

static void *LoggingAllocatePersistentBuffer(struct TfLiteContext *ctx,
                                             size_t bytes) {
  tflite::MicroInterpreter *con = ((tflite::MicroInterpreter *)ctx->impl_);
  tflite::MicroAllocator &a = con->allocator_;
  void *ptr = a.AllocatePersistentBuffer(bytes);
  assert(ptr != nullptr && "Alloc failure");
  g_loggedAllocations.push_back({-(g_arenaPtr - (uint8_t *)ptr + g_arena_size),
                                 bytes, g_currentNodeIndex});
  return ptr;
}
static TfLiteStatus LoggingRequestScratchBufferInArena(TfLiteContext *ctx,
                                                       size_t bytes,
                                                       int *buffer_idx) {
  // assert(false && "Not handling scratch buffers currently");
  tflite::MicroInterpreter *con = ((tflite::MicroInterpreter *)ctx->impl_);
  tflite::MicroAllocator &a = con->allocator_;
  g_max_scratch_buffer_size = std::max(g_max_scratch_buffer_size, bytes);
  // return a.RequestScratchBufferInArena(bytes,
  //                                                  buffer_idx);
  return kTfLiteOk;
}

std::vector<tflmc::Allocation> tflmc::RecordAllocations(
    const tflite::Model *model, ptrdiff_t arena_size,
    size_t &max_scratch_buffer_size) {
  g_arena_size = arena_size;
  std::vector<uint8_t> arena_buf(g_arena_size);
  g_arenaPtr = arena_buf.data();

  tflite::MicroErrorReporter error_reporter;
  tflite::AllOpsResolver resolver;
  tflite::ops::micro::xcore::RegisterXCOps(&resolver);
  tflite::MicroInterpreter interpreter(model, resolver, arena_buf.data(),
                                       g_arena_size, &error_reporter);

  auto ctx = &interpreter.context_;
  auto allocator = &interpreter.allocator_;
  auto graph = &interpreter.graph_;

  tflite::SubgraphAllocations *subgraphAllocations;
  tflite::ScratchBufferHandle *scratchhandle = nullptr;

  subgraphAllocations = allocator->StartModelAllocation(model);

  graph->SetSubgraphAllocations(subgraphAllocations);
  interpreter.PrepareNodeAndRegistrationDataFromFlatbuffer();

  // Only allow AllocatePersistentBuffer in Init stage.
  ctx->AllocatePersistentBuffer = &LoggingAllocatePersistentBuffer;
  ctx->RequestScratchBufferInArena = nullptr;
  ctx->GetScratchBuffer = nullptr;
  ctx->GetExternalContext = nullptr;
  graph->InitSubgraphs();

  // Both AllocatePersistentBuffer and RequestScratchBufferInArena is
  // available in Prepare stage.
  ctx->RequestScratchBufferInArena = &LoggingRequestScratchBufferInArena;

  graph->PrepareSubgraphs();

  allocator->FinishModelAllocation(model, graph->GetAllocations(),
                                   &scratchhandle);
  // Save max scratch buffer size
  max_scratch_buffer_size = g_max_scratch_buffer_size;

  return g_loggedAllocations;
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
