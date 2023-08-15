#include <stdio.h>
#include <print.h>
#include <stdint.h>
#include <platform.h>
#include "flash_server.h"

#include "fast_flash.h"

// TODO: move this to lib_tflite_micro
// It is linked with the flash generation library

#define TMP_BUF_SIZE_IN_BYTES  1024

#define VERSION_MAJOR 1
#define VERSION_MINOR 2
#define VERSION_LITTLE_ENDING (VERSION_MAJOR |\
                               (VERSION_MINOR << 8) |\
                               ((VERSION_MAJOR^0xff) << 16) |\
                               ((VERSION_MINOR^0xff) << 24))

static int flash_version_check(fl_QSPIPorts &qspi) {
    uint32_t tmp[2];
    fast_flash_read(qspi, /*unsigned addr*/32, /*unsigned word_count*/1, /*unsigned read_data[]*/(tmp, unsigned[]), /*chanend ?c_data_out*/NULL);

    return tmp[0] ^ VERSION_LITTLE_ENDING;
}

void flash_server(chanend c_flash[], flash_t headers[], int n_flash,
                  fl_QSPIPorts &qspi, fl_QuadDeviceSpec flash_spec[],
                  int n_flash_spec) {
    int res;
    if ((res = fl_connectToDevice(qspi, flash_spec, n_flash_spec)) != 0) {
        printstr("fl_connect err");printintln(res);    // TODO; these errors needs to be reported through AI server
        asm("clre; waiteu");
    }

    if ((res = fast_flash_init(qspi)) != 0) {
        printstr("fast flash init err ");printintln(res);
        asm("clre; waiteu");
    }

    if ((res = flash_version_check(qspi)) != 0) {
        printstr("version check error");printintln(res);
        asm("clre; waiteu");
    }    
    fast_flash_read(qspi, /*unsigned addr*/36, /*unsigned word_count*/(n_flash * sizeof(flash_t))/4, /*unsigned read_data[]*/(headers, unsigned[]), /*chanend ?c_data_out*/NULL);

    int flash_server_alive = 1;
    while(flash_server_alive) {
        int address, bytes;
        flash_command_t cmd;
        select {
            case (int i = 0; i < n_flash; i++) c_flash[i] :> cmd:
                if (cmd == FLASH_READ_PARAMETERS) {
                    c_flash[i] :> address;
                    c_flash[i] :> bytes;
                    address = headers[i].parameters_start + address;
                } else if (cmd == FLASH_READ_MODEL) {
                    unsigned bytes_length[1];
                    address = headers[i].model_start;
                    fast_flash_read(qspi, address, 1, (bytes_length, unsigned[]), NULL);
                    address += sizeof(uint32_t);
                    bytes   = (bytes_length, unsigned[])[0];
                    c_flash[i] <: bytes;
                } else if (cmd == FLASH_READ_OPERATORS) {
                    ; // TODO
                } else if (cmd == FLASH_SERVER_INIT) {
                    ; // TODO
                } else if (cmd == FLASH_SERVER_QUIT) {
                    flash_server_alive = 0;
                }
                if (flash_server_alive && cmd != FLASH_SERVER_INIT) {
                    fast_flash_read(qspi, address, bytes/4, /*not using this arg*/(address, unsigned[]), c_flash[i]);
                }
                break;
        }
    }
}