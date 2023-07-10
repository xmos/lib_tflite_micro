#if !defined(__micro_thread_library_h__)
#define __micro_thread_library_h__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define XCORE_MAX_NUM_THREADS 5

#ifdef __XC__
    #define UNSAFE unsafe
#else
    #define UNSAFE /**/
#endif

typedef struct { // THIS STRUCT MUST BE IN SYNC WITH ASSEMBLY CODE.
  union {
    uint64_t id_aligned[2]; // Guarantee 64-bit alignment.
    uint32_t id[4];         // Actual IDs
  } thread_ids;             // ids of at most 4 threads - live during invoke
  uint32_t synchroniser;    // synchroniser for threads - live during invoke
  uint32_t nstackwords;     // nstackwords per stack   - live after load model
  void *UNSAFE stacks;      // pointer to top of stack - live after load model
} thread_info_t;


#ifndef __XC__

typedef void (*thread_function_pointer_t)(void * arg0, void * arg1, void * arg2);
struct inference_engine;

/** Function that creates threads, then calls a interp_invoke_internal,
 * then destroys threads
 * This function creates four threads for a total of five threads.
 * other versions of the functions create 3, 2, 1, or 0 threads.
 *
 * \param   ie     Pointer to the inference object to be passed to
 *                 interp_invoke_internal
 * \param   ptr    Pointer to a thread_info block. The thread-ids will
 *                 be stored in this block, and a stack pointer is expected
 *                 in this block:
 *                     ptr[0] [out] thread-id-0    (versions with fewer threads
 *                     ptr[1] [out] thread-id-1     will only use the first few
 *                     ptr[2] [out] thread-id-2     slots)
 *                     ptr[3] [out] thread-id-3
 *                     ptr[4] [out] synchroniser-id
 *                     ptr[5] [in]  top of stacks
 *                     ptr[6] [in]  number of words per stack
 */
int thread_invoke_5(struct inference_engine *ie, thread_info_t *ptr);
int thread_invoke_4(struct inference_engine *ie, thread_info_t *ptr);
int thread_invoke_3(struct inference_engine *ie, thread_info_t *ptr);
int thread_invoke_2(struct inference_engine *ie, thread_info_t *ptr);
int thread_invoke_1(struct inference_engine *ie, thread_info_t *ptr);

/** Function that creates threads.
 * This function creates four threads for a total of five threads.
 * other versions of the functions create 3, 2, 1, or 0 threads.
 *
 * \param   ptr    Pointer to a thread_info block. The thread-ids will
 *                 be stored in this block, and a stack pointer is expected
 *                 in this block:
 *                     ptr[0] [out] thread-id-0    (versions with fewer threads
 *                     ptr[1] [out] thread-id-1     will only use the first few
 *                     ptr[2] [out] thread-id-2     slots)
 *                     ptr[3] [out] thread-id-3
 *                     ptr[4] [out] synchroniser-id
 *                     ptr[5] [in]  top of stacks
 *                     ptr[6] [in]  number of words per stack
 */
void thread_init_5(thread_info_t *ptr);
void thread_init_4(thread_info_t *ptr);
void thread_init_3(thread_info_t *ptr);
void thread_init_2(thread_info_t *ptr);
void thread_init_1(thread_info_t *ptr);
/** Function that destroys threads. Must be called from the same function that
 * called an _init_ above.
 *
 * \param   ptr    Pointer to a thread_info block.
 */
void thread_destroy(thread_info_t *ptr);

/** Function that sets up parameters for one of the client threads
 * This particular one passes the second and third arguments to the thread.
 * When the thread function is actually called (through thread_call)
 * the thread function will be called with those two arguments, 
 * and the first shared argument provided by thread_call.
 * Note - we can make versions with more or fewer parameters.
 * Note - we could pass this function the thread-function itself
 *
 * \param arg1      Second argument for the thread function
 * \param arg2      Third argument for the thread function
 * \param thread_id The thread_id to initialise; one of ptr[0]..ptr[3] above
 */
void thread_variable_setup(void * arg1, void * arg2, uint32_t thread_id);

/** Function that starts all thread functions and runs them until completion.
 * It is assumed that the variable parts have been set up per thread.
 * by thread_variable_setup.
 * This thread will also invoke the function with the given variable arguments.
 *
 * \param arg0      First argument shared among all threads (usually the output pointer)
 * \param arg1      Second argument for the master thread function
 * \param arg2      Third argument for the master thread function
 * \param fp        thread function to call on all threads.
 * \param ptr       Pointer to the thread info block held in the xcore
 * interpreter.
 */
void thread_call(void * arg0, void * arg1, void * arg2,
                 thread_function_pointer_t fp, thread_info_t *ptr);
#ifdef __cplusplus
};
#endif

#endif // __XC__

#endif // __micro_thread_library_h__
