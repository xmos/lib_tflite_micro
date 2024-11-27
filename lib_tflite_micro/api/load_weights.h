#ifndef _load_weights_h_
#define _load_weights_h_

#include <xcore/channel.h>
#include "thread_call.h"

#define LOAD_WEIGHTS_MAX_BLOCKS   2

/**
 * Function that connects to a flash or tile-ram server and loads a series of weights.
 * This function completes when the data is loaded.
 *
 * @param  c_flash_or_tile     channel-end connecting to the flash server
 *
 * @param  data_ptr            array of pointers where the loaded data should be scattered
 *
 * @param  data_sizes_in_words number of words where for each block
 *
 * @param  N                   number of blocks in data_ptr and data_sizes_in_words
 *
 * @param  external_addr       address in flash or tile ram
 *
 * @param  model_thread_count  number of threads available
 *
 * @param  tif                 thread_info structure for multithreading
 */ 
void load_weights_synchronous(chanend_t c_flash_or_tile, int *data_ptr[], int data_sizes_in_words[],
                              int N, int external_addr, int model_thread_count, thread_info_t *tif);

/**
 * Function that connects to a flash server and loads a series of weights.
 * This function continues loading after the call completes
 *
 * @param  c_flash_or_tile     channel-end connecting to the flash server
 *
 * @param  data_ptr            array of pointers where the loaded data should be scattered
 *
 * @param  data_sizes_in_words number of words where for each block
 *
 * @param  N                   number of blocks in data_ptr and data_sizes_in_words
 *
 * @param  external_addr       address in flash or tile ram
 *
 * @param  model_thread_count  number of threads available
 */ 
void load_weights_asynchronous(chanend_t c_flash_or_tile, int *data_ptr[], int data_sizes_in_words[],
                               int N, int external_addr);

/**
 * Function that connects to a flash server and waits for the last outstanding load to complete
 * Only one asynchronous load should be outstanding at any one time.
 *
 * @param  c_flash_or_tile     channel-end connecting to the flash server
 */
void load_weights_asynchronous_wait(chanend_t c_flash_or_tile);

/**
 * Function that connects to a flash or tile ram server and kills it.
 *
 * @param  c_flash_or_tile     channel-end connecting to the flash server
 */
void load_weights_quit(chanend_t c_flash_or_tile);

#endif
