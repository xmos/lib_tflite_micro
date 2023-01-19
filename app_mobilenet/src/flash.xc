#include <stdio.h>
#include <print.h>
#include <stdint.h>
#include <platform.h>
#include "flash.h"

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
    //fl_readData(32, 4, (tmp, unsigned char[]));

    fast_flash_read(qspi, /*unsigned addr*/32, /*unsigned word_count*/1, /*unsigned read_data[]*/(tmp, unsigned[]), /*chanend ?c_data_out*/NULL);

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

    // fast_flash_init(qspi);
    // fast_flash_read(qspi, /*unsigned addr*/0, /*unsigned word_count*/10, /*unsigned read_data[]*/(headers, unsigned[]), /*chanend ?c_data_out*/NULL);

    // if ((res = fl_dividerOverride(3)) != 0) {          // 25 MHz - sort of safe.
    //     printstr("fl_divider err");printintln(res);
    //     asm("waiteu");
    // }

    if ((res = fast_flash_init(qspi)) != 0) {
        printstr("fast flash init err ");printintln(res);
        asm("waiteu");
    }

    if ((res = flash_version_check(qspi)) != 0) {
        printstr("version check error");printintln(res);
        asm("waiteu");
    }
    //fl_readData(36, n_flash * sizeof(flash_t), (headers, unsigned char[]) ); // TODO, check?
    
    fast_flash_read(qspi, /*unsigned addr*/36, /*unsigned word_count*/(n_flash * sizeof(flash_t))/4, /*unsigned read_data[]*/(headers, unsigned[]), /*chanend ?c_data_out*/NULL);

    printstr("in flash_server\n");
    printf("model start %d\n", headers[0].model_start);
    printf("parameters start %d\n", headers[0].parameters_start);

    int flash_server_alive = 1;
    while(flash_server_alive) {
        int address, bytes;
        flash_command_t cmd;
        select {
            case (int i = 0; i < n_flash; i++) c_flash[i] :> cmd:
                //master {
                    if (cmd == FLASH_READ_PARAMETERS) {
                        c_flash[i] :> address;
                        c_flash[i] :> bytes;
                        // printstr("load from flash\n");
                        // printf("address %d bytes %d\n", address, bytes);
                        address = headers[i].parameters_start + address;
                    } else if (cmd == FLASH_READ_MODEL) {
                        unsigned bytes_length[1];
                        address = headers[i].model_start;
                        //fl_readData(address, sizeof(uint32_t), bytes_length);  // read length
                        //fast_flash_read_no_chanend(qspi, /*unsigned addr*/address, /*unsigned word_count*/1, /*unsigned read_data[]*/(bytes_length, unsigned[]));
                        address += sizeof(uint32_t);
                        bytes   = (bytes_length, unsigned[])[0];
                        c_flash[i] <: bytes;
                    } else if (cmd == FLASH_READ_OPERATORS) {
                        ; // TODO
                    } else if (cmd == FLASH_SERVER_QUIT) {
                        flash_server_alive = 0;
                    }
                    int k = 100;
                    // //c_flash[i] <: k;
                    // outuint(c_flash[i], k);


                    fast_flash_read(qspi, address, bytes/4, (k, unsigned[]), c_flash[i]);

                    // unsigned buf[TMP_BUF_SIZE_IN_BYTES/4];
                    // for(int k = 0; k < bytes; k += TMP_BUF_SIZE_IN_BYTES) {
                    //     int buf_bytes = TMP_BUF_SIZE_IN_BYTES;
                    //     if (k + buf_bytes > bytes) {
                    //         buf_bytes = bytes - k;
                    //     }
                    //     //fl_readData(address+k, buf_bytes, buf); // TODO, check?
                    //     //fast_flash_read_no_chanend(qspi, address+k, buf_bytes/4, (buf, unsigned[]));
                    //     int k;
                    //     fast_flash_read(qspi, address+k, buf_bytes/4, (k, unsigned[]), c_flash[i]);



                    //     // for(int j = 0; j < buf_bytes/4; j++) {
                    //     //     c_flash[i] <: buf[j];
                    //     // }
                    // }
                //}
                break;
        }
    }
}