
#include "thread_call.h"

#ifdef NO_INTERPRETER

typedef void (*voidfunc)(void);

#if defined(__xcore__) || defined(__riscv_xxcore)
#include <xcore/parallel.h>
DECLARE_JOB(main_task, (thread_info_t *, voidfunc, synchronizer_t));
DECLARE_JOB(client_task, (thread_info_t *, int));
#endif

void main_task(thread_info_t *t, voidfunc fptr, synchronizer_t sync) {
  #if defined(__xcore__) || defined(__riscv_xxcore)
  __attribute__((fptrgroup("_c_trampoline")))
  #endif
  voidfunc trampoline_fn = fptr;
  thread_store_sync(t, sync);
  trampoline_fn();
}

void client_task(thread_info_t *t, int n) {
  thread_client(t, n);
}

void par_invoke_1(thread_info_t *ti, voidfunc f) {
#if defined(__xcore__) || defined(__riscv_xxcore)
  PAR_JOBS(
     PJOB(main_task, (ti, f, PAR_SYNC)));
#else
  main_task(ti, f, 0);
#endif
}

void par_invoke_2(thread_info_t *ti, voidfunc f) {
#if defined(__xcore__) || defined(__riscv_xxcore)
  PAR_JOBS(
     PJOB(main_task, (ti, f, PAR_SYNC)),
     PJOB(client_task, (ti, 0)));
#else
  client_task(ti, 0);
  main_task(ti, f, 0);
#endif
}

void par_invoke_3(thread_info_t *ti, voidfunc f) {
#if defined(__xcore__) || defined(__riscv_xxcore)
  PAR_JOBS(
     PJOB(main_task, (ti, f, PAR_SYNC)),
     PJOB(client_task, (ti, 0)),
     PJOB(client_task, (ti, 1)));
#else
  client_task(ti, 0);
  client_task(ti, 1);
  main_task(ti, f, 0);
#endif
}

void par_invoke_4(thread_info_t *ti, voidfunc f) {
#if defined(__xcore__) || defined(__riscv_xxcore)
  PAR_JOBS(
     PJOB(main_task, (ti, f, PAR_SYNC)),
     PJOB(client_task, (ti, 0)),
     PJOB(client_task, (ti, 1)),
     PJOB(client_task, (ti, 2)));
#else
  client_task(ti, 0);
  client_task(ti, 1);
  client_task(ti, 2);
  main_task(ti, f, 0);
#endif
}

void par_invoke_5(thread_info_t *ti, voidfunc f) {
#if defined(__xcore__) || defined(__riscv_xxcore)
  PAR_JOBS(
     PJOB(main_task, (ti, f, PAR_SYNC)),
     PJOB(client_task, (ti, 0)),
     PJOB(client_task, (ti, 1)),
     PJOB(client_task, (ti, 2)),
     PJOB(client_task, (ti, 3)));
#else
  client_task(ti, 0);
  client_task(ti, 1);
  client_task(ti, 2);
  client_task(ti, 3);
  main_task(ti, f, 0);
#endif
}

#endif