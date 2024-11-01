#include <xcore/channel.h>
#include "thread_call.h"

void load_weights_synchronous(chanend_t c_flash_or_tile, int *data_ptr[], int data_sizes_in_words[],
                              int N, int external_addr, int model_thread_count, thread_info_t *tif);
void load_weights_asynchronous(chanend_t c_flash_or_tile, int *data_ptr[], int data_sizes_in_words[],
                               int N, int external_addr, int model_thread_count);
void load_weights_asynchronous_wait(chanend_t c_flash_or_tile);
void load_weights_quit(chanend_t c_flash_or_tile);
