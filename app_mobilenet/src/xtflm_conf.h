// #pragma once
// // catch all for missing defines missing when we don't use an interpreter
// #define NO_INTERPRETER (1)
// #define XTFLM_OPERATORS (0)
// #define MAX_DEBUG_LOG_LENGTH (0)
// #define NUM_INPUT_TENSORS (1)
// #define NUM_OUTPUT_TENSORS (1)
// #define AISRV_GPIO_LENGTH (0)


#ifndef _xtflm_conf_h_
#define _xtflm_conf_h_

#define XTFLM_OPERATORS 25

#define NETWORK_NUM_THREADS (1)
#define AISRV_GPIO_LENGTH (4)

#define NUM_OUTPUT_TENSORS (2)
#define NUM_INPUT_TENSORS (1)

#define MAX_DEBUG_LOG_LENGTH 256

#endif