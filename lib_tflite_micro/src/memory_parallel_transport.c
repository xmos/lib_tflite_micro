#include <xcore/chanend.h>
#include <xcore/channel.h>
#include <xcore/parallel.h>
#include "thread_call.h"
#include "memory_parallel_transport.h"

typedef struct {
    int whole;
    uint32_t *data;
} destination_description_t;

DECLARE_JOB(receive_rx, (destination_description_t *, chanend_t, int));
DECLARE_JOB(transmit_tx, (chanend_t, int, int, uint32_t *));

extern void receive_rx(destination_description_t *d, chanend_t c, int offset);
extern void transmit_tx(chanend_t C, int offset, int n, uint32_t *data);

extern void memory_parallel_receive(chanend_t c, uint32_t *data,
                                    uint32_t byte_count) {
    int whole = byte_count / 96;
    int last  = byte_count - whole * 96;
    destination_description_t dest = {whole, data};
    chanend_t other_c[3];
    for(int i = 0; i < 3; i++) {
        other_c[i] = chanend_alloc();
        chan_out_word(c, other_c[i]);
        chanend_t other_side = chan_in_word(c);
        chanend_set_dest(other_c[i], other_side);
    }
    PAR_JOBS(
        PJOB(receive_rx, (&dest, other_c[0], 0)),
        PJOB(receive_rx, (&dest, other_c[1], 1)),
        PJOB(receive_rx, (&dest, other_c[2], 2)),
        PJOB(receive_rx, (&dest,          c, 3))
        );
    for(int i = 0; i < 3; i++) {
        chanend_out_control_token(other_c[i], 1);
        chanend_check_control_token(other_c[i], 1);
    }
    chan_in_buf_word(c, &data[whole*24], last>>2);
    for(int i = 0; i < 3; i++) {
        chanend_free(other_c[i]);
    }
}

extern void memory_parallel_send(chanend_t c, uint32_t *data, uint32_t byte_count) {
    int whole = byte_count / 96;
    int last  = byte_count - whole * 96;
    chanend_t other_c[3];
    for(int i = 0; i < 3; i++) {
        other_c[i] = chanend_alloc();
        chanend_t other_side = chan_in_word(c);
        chan_out_word(c, other_c[i]);
        chanend_set_dest(other_c[i], other_side);
    }
    PAR_JOBS(
        PJOB(transmit_tx, (other_c[0], 0, whole, data)),
        PJOB(transmit_tx, (other_c[1], 1, whole, data)),
        PJOB(transmit_tx, (other_c[2], 2, whole, data)),
        PJOB(transmit_tx, (         c, 3, whole, data))
        );
    for(int i = 0; i < 3; i++) {
        chanend_out_control_token(other_c[i], 1);
        chanend_check_control_token(other_c[i], 1);
    }
    chan_out_buf_word(c, &data[whole*24], last>>2);
    for(int i = 0; i < 3; i++) {
        chanend_free(other_c[i]);
    }
}


extern void memory_parallel_receive_thread_call(chanend_t c, uint32_t *data,
                                                uint32_t byte_count, thread_info_t *thread_inf) {
    int whole = byte_count / 96;
    int last  = byte_count - whole * 96;
    destination_description_t dest = {whole, data};
    chanend_t other_c[3];
    for(int i = 0; i < 3; i++) {
        other_c[i] = chanend_alloc();
        chan_out_word(c, other_c[i]);
        chanend_t other_side = chan_in_word(c);
        chanend_set_dest(other_c[i], other_side);
    }
    thread_variable_setup((void*)other_c[1], (void*)1, thread_inf->thread_ids.id[1]);
    thread_variable_setup((void*)other_c[2], (void*)2, thread_inf->thread_ids.id[2]);
    thread_variable_setup((void*)c, (void*)3, thread_inf->thread_ids.id[3]);
    thread_call(&dest, (void*)other_c[0], (void*)0, receive_rx, thread_inf);
    for(int i = 0; i < 3; i++) {
        chanend_out_control_token(other_c[i], 1);
        chanend_check_control_token(other_c[i], 1);
    }
    chan_in_buf_word(c, &data[whole*24], last>>2);
    for(int i = 0; i < 3; i++) {
        chanend_free(other_c[i]);
    }
}
