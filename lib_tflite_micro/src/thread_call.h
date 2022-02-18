#if !defined(__micro_thread_library_h__)
#define __micro_thread_library_h__

#include <stdint.h>

typedef struct {
    uint32_t data[7];
} thread_info_t;

typedef void (*function_pointer)();

/** Function that creates threads, then calls a interp_invoke_internal,
 * then destroys threads
 * This function creates four threads.
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
void thread_invoke_4(void *ie, thread_info_t *ptr);

/** Function that sets up parameters for one of the client threads
 * This particular one passes three arguments to the thread.
 * When the thread function is actually called (through thread_call)
 * the thread function will be called with these three arguments.
 * Note - we can make versions with more or fewer parameters.
 * Note - we could pass this function the thread-function itself
 *
 * \param arg0      First argument for the thread function
 * \param arg1      Second argument for the thread function
 * \param arg2      Third argument for the thread function
 * \param thread_id The thread_id to initialise; one of ptr[0]..ptr[3] above
 */
void thread_variable_setup(void *arg0, void *arg1, void *arg2, uint32_t thread_id);

/** Function that starts all thread functions and runs them until completion.
 * It is assumed that the variable parts have been set up per thread.
 * by thread_variable_setup.
 * This thread will also invoke the function with the given variable arguments.
 *
 * \param arg0      First argument for the master thread function
 * \param arg1      Second argument for the master thread function
 * \param arg2      Third argument for the master thread function
 * \param fp        thread function to call on all threads.
 * \param ptr       Pointer to the thread info block held in the xcore interpreter.
 */
void thread_call(void *arg0, void *arg1, void *arg2,
                          function_pointer fp, thread_info_t *ptr);

#endif
