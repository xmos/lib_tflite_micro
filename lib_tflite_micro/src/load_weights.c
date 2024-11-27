#include <assert.h>
#include "load_weights.h"
#include "flash_server.h"
#include "memory_parallel_transport.h"

void load_weights_synchronous(chanend_t c_flash_or_tile, int *data_ptrs[], int data_sizes_in_words[],
                              int N, int external_addr, int model_thread_count, thread_info_t *tif) {
    // Parallel mode is for reading weights from another tile
    chan_out_word(c_flash_or_tile, FLASH_READ_PARAMETERS);
    int use_parallel_mode = chan_in_word(c_flash_or_tile);
    if (!use_parallel_mode) {
        chan_out_word(c_flash_or_tile, external_addr);

        int32_t total_bytes = 0;
        for (int i = 0; i < N; ++i) {
            total_bytes += data_sizes_in_words[i] * 4;
        }
        chan_out_word(c_flash_or_tile, total_bytes);

        for (int i = 0; i < N; ++i) {
            int *data_ptr = data_ptrs[i];
            // The sizes are in bytes and we read from flash in words
            int op_data_size_in_words = data_sizes_in_words[i];
#pragma clang loop unroll_count(4)
            for (int j = 0; j < op_data_size_in_words; j++) {
                // We are reading directly from flash chanend here.
                // We use chanend_in_word() instead of chan_in_word() to
                // avoid handshake.
                // Adding something like a printf() within this loop
                // might slow it down enough to corrupt the received data.
                ((uint32_t *)data_ptr)[j] = chanend_in_word(c_flash_or_tile);
            }
        }
        // As there is no handshake, we have to accept the end token
        // to close the chanend
        chanend_check_end_token(c_flash_or_tile);
    } else {
        // The parallel mode uses four threads and can only work if
        // the model has been compiled with at least four threads.
        assert(model_thread_count >= 4 &&
               "At least four threads are required for parallel read from "
               "another tile!");
        chan_out_word(c_flash_or_tile, external_addr);
        chan_out_word(c_flash_or_tile, data_sizes_in_words[0]*4);
        external_addr += data_sizes_in_words[0]*4;
        memory_parallel_receive_thread_call(c_flash_or_tile, (uint32_t *)data_ptrs[0],
                                            4*data_sizes_in_words[0], tif);
        for (int i = 1; i < N; ++i) {
            chan_out_word(c_flash_or_tile, 0);
            chan_in_word(c_flash_or_tile);
            chan_out_word(c_flash_or_tile, external_addr);
            chan_out_word(c_flash_or_tile, data_sizes_in_words[i]*4);
            external_addr += data_sizes_in_words[i]*4;
            memory_parallel_receive_thread_call(c_flash_or_tile, (uint32_t *)data_ptrs[i],
                                                4*data_sizes_in_words[i], tif);
        }
    }
}

void load_weights_asynchronous(chanend_t c_flash_or_tile, int *data_ptrs[], int data_sizes_in_words[],
                              int N, int external_addr) {
    chan_out_word(c_flash_or_tile, FLASH_READ_PARAMETERS_ASYNC);
    chan_out_word(c_flash_or_tile, external_addr);
    chan_out_word(c_flash_or_tile, N);
    
    for (int i = 0; i < N; ++i) {
        chan_out_word(c_flash_or_tile, data_sizes_in_words[i] * 4);
        chan_out_word(c_flash_or_tile, (int) data_ptrs[i]);
    }
}

void load_weights_asynchronous_wait(chanend_t c_flash_or_tile) {
    chanend_check_end_token(c_flash_or_tile);
}

void load_weights_quit(chanend_t c_flash_or_tile) {
    chan_out_word(c_flash_or_tile, FLASH_SERVER_QUIT);
}
