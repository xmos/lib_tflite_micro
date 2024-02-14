#ifndef XCORE_UTILS_H_
#define XCORE_UTILS_H_

#include <cassert>
#include <cstdint>
#include <utility>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/memory_helpers.h"

namespace tflite {
namespace ops {
namespace micro {

struct XCoreOpData {
  const char *name;
};

namespace xcore {
/* Unpack an integer data type from a byte array
 *  T  data type to unpack
 *
 * Example usage:
 *      int32_t t0 = unpack<int32_t>(&my_buffer[23]);
 *      int32_t t1 = unpack<int32_t>(&my_buffer[27]);
 */
template <class T> T unpack(const uint8_t *buffer) {
  T retval = 0;
  for (int i = 0; i < sizeof(T); ++i)
    retval |= buffer[i] << (8 * i);
  return retval;
}

template <typename T>
static inline T *construct_persistent_object(TfLiteContext *context) {
  return new (context->AllocatePersistentBuffer(context, sizeof(T))) T;
}

static inline bool is_ram_address(uintptr_t a) {
#ifdef XCORE
  return ((a >= 0x80000) && (a <= 0x100000));
#else
  return true;
#endif
}

static inline TfLiteStatus request_scratch_if_needed(TfLiteContext *context,
                                                     const void *source_address,
                                                     const size_t size,
                                                     int &scratch_idx) {
  if (source_address && !is_ram_address((uintptr_t)source_address)) {
    return context->RequestScratchBufferInArena(context, size, &scratch_idx);
  }
  return kTfLiteOk;
}

static inline TfLiteStatus request_scratch_if_needed(TfLiteContext *context,
                                                     const TfLiteTensor *tensor,
                                                     int &scratch_idx) {
  return request_scratch_if_needed(context, tensor->data.data, tensor->bytes,
                                   scratch_idx);
}

extern "C" {
static inline void memload(void *dest, void *src, size_t size) {
  // printf("memload dest=%d   src=%d   size=%d\n", (long)dest, (long)src,
  // size);
  memcpy(dest, src, size);
}
}

size_t FetchBuffer(int8_t **dest, int8_t const *src, size_t size);

template <typename T>
static inline TfLiteStatus
fetch_scratch_if_needed(TfLiteContext *context, T *&array,
                        const TfLiteEvalTensor *tensor, int scratch_idx) {
  if (scratch_idx >= 0) {
    array =
        static_cast<const T *>(context->GetScratchBuffer(context, scratch_idx));
    const RuntimeShape shape = tflite::micro::GetTensorShape(tensor);

    size_t sizeof_tensor_type;
    TfLiteTypeSizeOf(tensor->type, &sizeof_tensor_type);
    FetchBuffer((int8_t **)&array, tflite::micro::GetTensorData<int8_t>(tensor),
                shape.FlatSize() * sizeof_tensor_type);
  } else {
    array = tflite::micro::GetTensorData<T>(tensor);
  }
  TF_LITE_ENSURE(context, array);
  return kTfLiteOk;
}

template <typename T> class PersistentArray {
private:
  size_t max_size_ = 0;
  size_t size_ = 0;
  T *data_ = nullptr;

public:
  // call this only in the Init phase of operators
  PersistentArray<T> &allocate(TfLiteContext *context,
                               size_t max_size) noexcept {
    assert(data_ == nullptr);
    assert(max_size > 0);

    max_size_ = max_size;
    data_ = reinterpret_cast<T *>(
        context->AllocatePersistentBuffer(context, sizeof(T) * max_size));

    return *this;
  };
  PersistentArray<T> &initialize() noexcept {
    assert(size_ == 0);
    while (size_ < max_size_) {
      this->append(T());
    }

    return *this;
  };
  // TODO: begin and end would be better if returned an iterator object
  inline T *begin() noexcept {
    assert(size_ > 0);
    return &data_[0];
  }
  inline T *end() noexcept {
    assert(size_ > 0);
    return &data_[size_];
  }
  inline T &operator[](int i) noexcept {
    assert(i < size_);
    return data_[i];
  }
  inline void append(const T &element) noexcept {
    assert(size_ < max_size_);
    data_[size_++] = element;
  }
  inline void append(T &&element) noexcept {
    assert(size_ < max_size_);
    data_[size_++] = std::move(element);
  }
  inline size_t size() noexcept { return size_; }
  inline size_t max_size() noexcept { return max_size_; }
};

#ifndef UNSUPPORTED_KERNEL_TYPE
#define UNSUPPORTED_KERNEL_TYPE(T)                                             \
  {                                                                            \
    DebugLog("Unsupported " #T " value");                                      \
    TFLITE_ABORT;                                                              \
  }
#endif /*UNSUPPORTED_KERNEL_TYPE*/

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite

#endif // XCORE_UTILS_H_
