#include <assert.h>
#include <stdio.h>
#include <print.h>
#include <string.h>
#include <stdint.h>
#include <platform.h>
#include <xcore/channel.h>
#include <xcore/chanend.h>
#include "tile_ram_server.h"
#include "memory_parallel_transport.h"

#define TMP_BUF_SIZE_IN_BYTES  1024

#define VERSION_MAJOR 1
#define VERSION_MINOR 2
#define VERSION_LITTLE_ENDING (VERSION_MAJOR |\
                               (VERSION_MINOR << 8) |\
                               ((VERSION_MAJOR^0xff) << 16) |\
                               ((VERSION_MINOR^0xff) << 24))

void tile_ram_server(chanend_t *c_tile_ram, flash_t *headers, int n_tile_ram,
                     const int8_t *tile_ram) {
    uint32_t tmp = ((uint32_t*)tile_ram)[0];
    if ((tmp ^ VERSION_LITTLE_ENDING) != 0) {
        printstr("version check error");
        asm("clre; waiteu");
    }
    memcpy(headers, tile_ram + 4, (n_tile_ram * sizeof(flash_t)));
    assert(n_tile_ram == 1);
    int tile_ram_server_alive = 1;
    while(tile_ram_server_alive) {
        int byte_address, number_bytes;
        flash_command_t cmd;
        int i = 0;           // TODO: extend SELECT-FOR-LOOP
        cmd = chan_in_word(c_tile_ram[i]);
        //if (cmd == FLASH_READ_PARAMETERS || cmd == FLASH_READ_PARAMETERS_COMPRESSED_FLOAT) {
        if (cmd == FLASH_READ_PARAMETERS) {
            // Set parallel mode
            chan_out_word(c_tile_ram[i], 1);
            byte_address = chan_in_word(c_tile_ram[i]);
            number_bytes   = chan_in_word(c_tile_ram[i]);
            byte_address = headers[i].parameters_start + byte_address;
        } else if (cmd == FLASH_READ_OPERATORS) {
            ;
        } else if (cmd == FLASH_SERVER_INIT) {
            ;  // NO init required
        } else if (cmd == FLASH_SERVER_QUIT) {
            tile_ram_server_alive = 0;
        }
        if (tile_ram_server_alive && cmd != FLASH_SERVER_INIT) {
            memory_parallel_send(c_tile_ram[i], &((uint8_t *)tile_ram)[byte_address], number_bytes);
        }
    }
}
