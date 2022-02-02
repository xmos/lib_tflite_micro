#include <stdio.h>
#include <print.h>
#include <stdint.h>
#include <platform.h>
#include "flash.h"

// TODO: move this to lib_tflite_micro
// It is linked with the flash generation library

#define TMP_BUF_SIZE  1024

#define VERSION_MAJOR 1
#define VERSION_MINOR 2
#define VERSION_LITTLE_ENDING (VERSION_MAJOR |\
                               (VERSION_MINOR << 8) |\
                               ((VERSION_MAJOR^0xff) << 16) |\
                               ((VERSION_MINOR^0xff) << 24))

static int flash_version_check() {
    uint32_t tmp[1];
    fl_readData(0, 4, (tmp, unsigned char[]));
    return tmp[0] ^ VERSION_LITTLE_ENDING;
}

void flash_server(chanend c_flash[], flash_t headers[], int n_flash,
                  fl_QSPIPorts &qspi, fl_QuadDeviceSpec flash_spec[],
                  int n_flash_spec) {
    int res;
    if ((res = fl_connectToDevice(qspi, flash_spec, n_flash_spec)) != 0) {
        printstr("fl_connect err");printintln(res);    // TODO; these errors needs to be reported through AI server
        asm("waiteu");
    }
    if ((res = fl_dividerOverride(3)) != 0) {          // 25 MHz - sort of safe.
        printstr("fl_divider err");printintln(res);
        asm("waiteu");
    }
    if ((res = flash_version_check()) != 0) {
        printstr("version check error");printintln(res);
        asm("waiteu");
    }
    fl_readData(4, n_flash * sizeof(flash_t), (headers, unsigned char[]) ); // TODO, check?
    while(1) {
        int address, bytes;
        flash_command_t cmd;
        select {
            case (int i = 0; i < n_flash; i++) c_flash[i] :> cmd:
                master {
                    if (cmd == FLASH_READ_PARAMETERS) {
                        c_flash[i] :> address;
                        c_flash[i] :> bytes;
                        address = headers[i].parameters_start + address;
                    } else if (cmd == FLASH_READ_MODEL) {
                        unsigned char bytes_length[sizeof(uint32_t)];
                        address = headers[i].model_start;
                        fl_readData(address, sizeof(uint32_t), bytes_length);  // read length
                        address += sizeof(uint32_t);
                        bytes   = (bytes_length, unsigned[])[0];
                        c_flash[i] <: bytes;
                    } else if (cmd == FLASH_READ_OPERATORS) {
                        ; // TODO
                    }
                    unsigned char buf[TMP_BUF_SIZE];
                    for(int k = 0; k < bytes; k += TMP_BUF_SIZE) {
                        int buf_bytes = TMP_BUF_SIZE;
                        if (k + buf_bytes > bytes) {
                            buf_bytes = bytes - k;
                        }
                        fl_readData(address+k, buf_bytes, buf); // TODO, check?
                        for(int j = 0; j < buf_bytes; j++) {
                            c_flash[i] <: buf[j];
                        }
                    }
                }
                break;
        }
    }
}

