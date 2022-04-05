#ifndef SERVER_MEMORY_H_
#define SERVER_MEMORY_H_

#include "inference_engine.h"

#ifdef __cplusplus
extern "C" {
#endif
void inference_engine_initialize_with_memory(inference_engine_t *UNSAFE ie);
#ifdef __cplusplus
};
#endif

#endif
