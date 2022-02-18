#if defined(__XS3A__) || defined(__XS2A__)
#include <xs1.h>

#define NSTACKWORDS 4
    
    .globl  thread_invoke_4
    .type   thread_invoke_4,@function
    .cc_top thread_invoke_4.function,thread_invoke_4
    .align 4
    //    void thread_invoke_4(inference_engine_t *ie, thread_info *ptr)

thread_invoke_4:
    ENTSP_lu6 NSTACKWORDS
    std r4, r5, sp[1]
    
    getr r4, XS1_RES_TYPE_SYNC     // start with a synchroniser
    stw  r4, r1[4]                 // save synchroniser in xcore_interpreter
    ldw  r3, r1[5]                 // retrieve wordcount for each stack from xc_int
    ldw  r5, r1[6]                 // retrieve base stack pointer from xc_interpreter
    ldap r11, thread_client

    ldaw r5, r5[-r3]
    getst r2, res[r4]             // Obtain first thread
    stw r2, r1[0]
    init     t[r2]:pc, r11
    init     t[r2]:sp, r5

    ldaw r5, r5[-r3]
    getst r2, res[r4]             // Obtain second thread
    stw r2, r1[1]
    init     t[r2]:pc, r11
    init     t[r2]:sp, r5

    ldaw r5, r5[-r3]
    getst r2, res[r4]             // Obtain third thread
    stw r2, r1[2]
    init     t[r2]:pc, r11
    init     t[r2]:sp, r5

    ldaw r5, r5[-r3]
    getst r2, res[r4]             // Obtain fourth thread
    stw r2, r1[3]
    init     t[r2]:pc, r11
    init     t[r2]:sp, r5

    msync res[r4]                 // let all threads initialise

    bl    interp_invoke_internal  // call invoke

    mjoin res[r4]                 // destroy all threads
    freer res[r4]                 // get rid of the evidence
    ldd r4, r5, sp[1]             // and restore r4.
    retsp NSTACKWORDS
    
    
    // RETURN_REG_HOLDER
    .cc_bottom thread_invoke_4.function
    .set    thread_invoke_4.nstackwords,(NSTACKWORDS + interp_invoke_internal.nstackwords)
    .globl  thread_invoke_4.nstackwords
    .set    thread_invoke_4.maxcores,4
    .globl  thread_invoke_4.maxcores
    .set    thread_invoke_4.maxtimers,0
    .globl  thread_invoke_4.maxtimers
    .set    thread_invoke_4.maxchanends,0
    .globl  thread_invoke_4.maxchanends
.Ltmp_thread_invoke_4:
    .size   thread_invoke_4, .Ltmp_thread_invoke_4-thread_invoke_4

#undef NSTACKWORDS


#define NSTACKWORDS 0
    
    .globl  thread_client
    .type   thread_client,@function
    .cc_top thread_client.function,thread_client
    .align 4

    //    void thread_client...
    .issue_mode single

thread_client:
    ldap    r11, _cp
    set     cp, r11
    ldap    r11, _dp                // It shouldn't need this...
    set     dp, r11
thread_client_loop:
    ssync
    bla r11
    ssync
    bu thread_client_loop
    
    // RETURN_REG_HOLDER
    .cc_bottom thread_client.function
    .set    thread_client.nstackwords,NSTACKWORDS
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

#define NSTACKWORDS 0
    
    .globl  thread_variable_setup
    .type   thread_variable_setup,@function
    .cc_top thread_variable_setup.function,thread_variable_setup

    //    void thread_variable_setup(void *arg0, void *arg1, void *arg2, iuint32_t thread_id)
    .issue_mode single
    .align 4

thread_variable_setup:
    ENTSP_lu6   NSTACKWORDS
    set     t[r3]:r0, r0
    set     t[r3]:r1, r1
    set     t[r3]:r2, r2
    retsp   NSTACKWORDS
    
    // RETURN_REG_HOLDER
    .cc_bottom thread_variable_setup.function
    .set    thread_variable_setup.nstackwords,NSTACKWORDS
    .globl  thread_variable_setup.nstackwords
    .set    thread_variable_setup.maxcores,1
    .globl  thread_variable_setup.maxcores
    .set    thread_variable_setup.maxtimers,0
    .globl  thread_variable_setup.maxtimers
    .set    thread_variable_setup.maxchanends,0
    .globl  thread_variable_setup.maxchanends
.Ltmp_thread_variable_setup:
    .size   thread_variable_setup, .Ltmp_thread_variable_setup-thread_variable_setup

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
    ldd     r6, r7, r4[0]
    set     t[r6]:r11, r3           // set function pointer of thread 0
    bf      r7, .thread_call_start_threads
    set     t[r7]:r11, r3           // set function pointer of thread 1
    ldd     r6, r7, r4[1]
    bf      r6, .thread_call_start_threads
    set     t[r6]:r11, r3           // set function pointer of thread 2
    bf      r7, .thread_call_start_threads
    set     t[r7]:r11, r3           // set function pointer of thread 3
.thread_call_start_threads:   
    ldw     r6, r4[4]               // fetch synchroniser
    ldaw    r7, sp[0]
    ldw     r11, r4[6]
    set     sp, r11
    msync   res[r6]                 // start all clients
    bla     r3                      // start myself
    msync   res[r6]                 // Wait for all clients to finish

    mov     r11, r7
    set     sp, r11
    ldd     r4, r5, sp[1]           // and restore r4, r5, r6, r7.
    ldd     r6, r7, sp[2]
    retsp   NSTACKWORDS
    
    // RETURN_REG_HOLDER
    .cc_bottom thread_call.function
    .set    thread_call.nstackwords,NSTACKWORDS
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
