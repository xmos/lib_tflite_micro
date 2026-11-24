#pragma once

// Maximum number of threads for XCore
#define XCORE_MAX_NUM_THREADS 5

// VX4B toolchain detection, to be removed in the future
#if defined(__riscv_xxcore) && !defined(__VX4B__)
#define __VX4B__ 1
#endif

// Function pointer group availability
#ifndef HAS_FPTRGROUP
#if defined(__xcore__) || defined(__riscv_xxcore)
#define HAS_FPTRGROUP 1
#else
#define HAS_FPTRGROUP 0
#endif
#endif

// XC backward compatibility
#ifdef __XC__
#define UNSAFE unsafe
#else
#define UNSAFE /**/
#endif

// C API macro for C++ and XC compatibility
#ifndef C_API
#if defined(__cplusplus) || defined(__XC__)
#define C_API extern "C"
#else
#define C_API
#endif
#endif
