#define STACK_WORDS_PER_THREAD 1024
#define STACK_BYTES_PER_THREAD 4096
#ifndef NO_INTERPRETER


#if defined(__XS3A__) || defined(__XS2A__)
#include <xs1.h>

#define NSTACKWORDS 0

    .globl  thread_client
    .type   thread_client,@function
    .cc_top thread_client.function,thread_client
    .align 4

    //    void thread_client...
    .issue_mode dual

thread_client:
    { dualentsp 0               ; get     r11, id }
    shl     r11, r11, 8
    add     r11, r11, 4
    stw     r11, r0[r1]
thread_client_loop:
    ldc     r10, 0
    ssync
    bf      r10, do_not_run
    {add r4, r1, 0; add r5, r2, 0}
    bla r11
    {add r1, r4, 0; add r2, r5, 0}
do_not_run:
    ssync
    bu thread_client_loop

    // RETURN_REG_HOLDER
    .cc_bottom thread_client.function
    .set    thread_client.nstackwords,NSTACKWORDS+STACK_WORDS_PER_THREAD
    .globl  thread_client.nstackwords
    .set    thread_client.maxcores,1
    .globl  thread_client.maxcores
    .set    thread_client.maxtimers,0
    .globl  thread_client.maxtimers
    .set    thread_client.maxchanends,0
    .globl  thread_client.maxchanends
.Ltmp_thread_client:
    .size   thread_client, .Ltmp_thread_client-thread_client

#undef NSTACKWORDS


#define NSTACKWORDS 6

    .globl  thread_call
    .type   thread_call,@function
    .cc_top thread_call.function,thread_call

     // void thread_call(void *arg0, void *arg1, void *arg2,
     //                           (void *fp)(), thread_info_t *ptr);
    .issue_mode single
    .align 4

thread_call:
    ENTSP_lu6   NSTACKWORDS
    std     r4, r5, sp[1]           // save r4, r5, r6, r7
    std     r6, r7, sp[2]
    ldw     r4, sp[NSTACKWORDS+1]   // Pick up ptr
    ldd     r7, r6, r4[0]           // Pick up two thread-ids, they may be zero...
    bf      r6, .thread_call_start_threads
    set     t[r6]:r11, r3           // set function pointer of thread 0
    set     t[r6]:r0, r0            // set shared data
    bf      r7, .thread_call_start_threads
    set     t[r7]:r11, r3           // set function pointer of thread 1
    set     t[r7]:r0, r0            // set shared data
    ldd     r7, r6, r4[1]
    bf      r6, .thread_call_start_threads
    set     t[r6]:r11, r3           // set function pointer of thread 2
    set     t[r6]:r0, r0            // set shared data
    bf      r7, .thread_call_start_threads
    set     t[r7]:r11, r3           // set function pointer of thread 3
    set     t[r7]:r0, r0            // set shared data
.thread_call_start_threads:
    ldw     r6, r4[4]               // fetch synchroniser
    msync   res[r6]                 // start all clients
    nop                             // align with bf of other threads.
    bla     r3                      // start myself
    msync   res[r6]                 // Wait for all clients to finish

    ldd     r4, r5, sp[1]           // and restore r4, r5, r6, r7.
    ldd     r6, r7, sp[2]
    retsp   NSTACKWORDS

    // RETURN_REG_HOLDER
    .cc_bottom thread_call.function
    .set    thread_call.nstackwords,NSTACKWORDS+STACK_WORDS_PER_THREAD
    .globl  thread_call.nstackwords
    .set    thread_call.maxcores,1
    .globl  thread_call.maxcores
    .set    thread_call.maxtimers,0
    .globl  thread_call.maxtimers
    .set    thread_call.maxchanends,0
    .globl  thread_call.maxchanends
.Ltmp_thread_call:
    .size   thread_call, .Ltmp_thread_call-thread_call

#undef NSTACKWORDS

#endif




#if defined(__VX4A__)

    .globl  thread_client
    .resource_list_empty thread_client, "callees"
    .resource_list_empty thread_client, "tail_callees"
    .resource_list_empty thread_client, "parallel_callees"
    .resource_const thread_client, "stack_frame_bytes", STACK_BYTES_PER_THREAD
    .p2align 2

thread_client:
    { xm.entsp 0               ; xm.getid     x28 }
    xm.shli    x28, x28, 8
    add        x28, x28, 4
    xm.stw     x28, x11(x10)
thread_client_loop:
    xm.ldcu    x24, 0
    xm.csync
    {xm.brff    x24, do_not_run ; xm.addi  x19, x12, 0 }
    {xm.bla     x28             ; xm.addi  x18, x11, 0 }
    {xm.addi   x11, x18, 0      ; xm.addi  x12, x19, 0 }
do_not_run:
    xm.csync
    xm.bu thread_client_loop

    .globl  thread_call
    .resource_list_empty thread_call, "callees"
    .resource_list_empty thread_call, "tail_callees"
    .resource_list_empty thread_call, "parallel_callees"
    .resource_const thread_call, "stack_frame_bytes", STACK_BYTES_PER_THREAD

    #define NSTACKBYTES 32
    
     // void thread_call(void *arg0, void *arg1, void *arg2,
     //                          (void *fp)(), thread_info_t *ptr);
    .p2align 2

thread_call:
    xm.entsp   NSTACKBYTES
    xm.stdsp   x18, x19, 8           // save x18, x19, x20, x21
    xm.stdsp   x20, x21, 16
    addi       x18, x14, 0           // Pick up ptr
    xm.lddi    x20, x21, 0(x18)           // Pick up two thread-ids, they may be zero...
    xm.brff    x20, .thread_call_start_threads
    xm.tsetr   x20, 28, x13           // xm.tsetr function pointer of thread 0
    xm.tsetr   x20, 10, x10            // xm.tsetr shared data
    xm.brff    x21, .thread_call_start_threads
    xm.tsetr   x21, 28, x13           // xm.tsetr function pointer of thread 1
    xm.tsetr   x21, 10, x10            // xm.tsetr shared data
    xm.lddi    x20, x21, 8(x18)
    xm.brff    x20, .thread_call_start_threads
    xm.tsetr   x20, 28, x13           // xm.tsetr function pointer of thread 2
    xm.tsetr   x20, 10, x10            // xm.tsetr shared data
    xm.brff    x21, .thread_call_start_threads
    xm.tsetr   x21, 28, x13           // xm.tsetr function pointer of thread 3
    xm.tsetr   x21, 10, x10            // xm.tsetr shared data
.thread_call_start_threads:
    lw         x20, 16(x18)               // fetch synchroniser
    xm.psync   x20                 // start all clients
    nop                             // align with bf of other threads.
    xm.bla     x13                      // start myself
    xm.psync   x20                 // Wait for all clients to finish

    xm.lddsp   x18, x19, 8           // and restore x18, x19, x20, x21.
    xm.lddsp   x20, x21, 16
    xm.retsp   NSTACKBYTES

#undef NSTACKWORDS

#endif
#endif
    
