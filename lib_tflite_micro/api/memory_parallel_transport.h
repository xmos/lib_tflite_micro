#include <stdint.h>
#include "thread_call.h"

#ifdef __XC__
#include <xs1.h>
#else
#include <xcore/chanend.h>
#endif

#ifdef __XC__

extern void memory_parallel_receive(chanend c, uint32_t data[], uint32_t bytes);
extern void memory_parallel_receive_thread_call(chanend c, uint32_t data[], uint32_t bytes, thread_info_t &ptr);
extern void memory_parallel_send(chanend c, uint32_t data[], uint32_t bytes);

#else

/** Function that receives a block of data.
 * The number of bytes must be a multiple of 4.
 * This function creates three threads and three channel ends in order to
 * make full use of the bandwidth of the switch.
 * 
 * \param c        channel end to the sender
 * \param data     pointer where data must be stored
 * \param bytes    number of bytes that will be received.
 */
extern void memory_parallel_receive(chanend_t c, uint32_t *data, uint32_t bytes);

/** Function that receives a block of data.
 * The number of bytes must be a multiple of 4.
 * This function assumes that at least three threads have been created by the
 * thread_call library and will use those together with three fresh channel
 * ends in order to make full use of the bandwidth of the switch.
 * 
 * \param c        channel end to the sender
 * \param data     pointer where data must be stored
 * \param bytes    number of bytes that will be received.
 */
extern void memory_parallel_receive_thread_call(chanend_t c, uint32_t *data, uint32_t bytes, thread_info_t *ptr);

/** Function that sends a block of data.
 * The number of bytes must be a multiple of 4.
 * This function creates three threads and three channel ends in order to
 * make full use of the bandwidth of the switch.
 * 
 * \param c        channel end to the receiver
 * \param data     pointer where data must be loaded frmo
 * \param bytes    number of bytes that will be sent.
 */
extern void memory_parallel_send(chanend_t c, uint32_t *data, uint32_t bytes);

#endif
