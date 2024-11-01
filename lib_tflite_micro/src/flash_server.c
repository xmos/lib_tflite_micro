#include <stdio.h>
#include <print.h>
#include <stdint.h>
#include <platform.h>
#include <xcore/channel.h>
#include <xcore/select.h>
#include "flash_server.h"

#include "fast_flash.h"

#define TMP_BUF_SIZE_IN_BYTES  1024

#define VERSION_MAJOR 1
#define VERSION_MINOR 2
#define VERSION_LITTLE_ENDING (VERSION_MAJOR |\
                               (VERSION_MINOR << 8) |\
                               ((VERSION_MAJOR^0xff) << 16) |\
                               ((VERSION_MINOR^0xff) << 24))

static int flash_version_check(fl_QSPIPorts *qspi) {
    uint32_t tmp[2];
    fast_flash_read(qspi, /*unsigned addr*/32, /*unsigned word_count*/1, /*unsigned read_data[]*/(unsigned *)tmp, /*chanend ?c_data_out*/ 0);

    return tmp[0] ^ VERSION_LITTLE_ENDING;
}



static int flash_server_operate(chanend_t c_flash, flash_t *headers, fl_QSPIPorts *qspi) {
    int address, bytes;
    flash_command_t cmd;
    cmd = chan_in_word(c_flash);
    if (cmd == FLASH_READ_PARAMETERS) {
        // Set not parallel mode
        chan_out_word(c_flash, 0);
        address = chan_in_word(c_flash);
        bytes   = chan_in_word(c_flash);
        address = headers->parameters_start + address;
        fast_flash_read(qspi, address, bytes/4, /*not using this arg*/(unsigned*)address, c_flash);
    } else if (cmd == FLASH_READ_PARAMETERS_ASYNC) {
        int target_address;
        address = chan_in_word(c_flash);
        bytes   = chan_in_word(c_flash);
        target_address = chan_in_word(c_flash);
        address = headers->parameters_start + address;
        fast_flash_read(qspi, address, bytes/4, (unsigned *)target_address, 0);
        chanend_out_end_token(c_flash);
    } else if (cmd == FLASH_SERVER_INIT) {
        ; // TODO
    } else if (cmd == FLASH_SERVER_QUIT) {
        return 0;
    }
    return 1;
}


void flash_server(chanend_t c_flash[], flash_t headers[], int n_flash,
                  fl_QSPIPorts *qspi, fl_QuadDeviceSpec flash_spec[],
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
    
    fast_flash_read(qspi, /*unsigned addr*/36, /*unsigned word_count*/(n_flash * sizeof(flash_t))/4, /*unsigned read_data[]*/(unsigned*)headers, /*chanend ?c_data_out*/ 0);
    
    int flash_server_alive = 1;
    if (n_flash == 1) {
        while(flash_server_alive) {
            flash_server_alive = flash_server_operate(c_flash[0], &headers[0], qspi);
        }
    } else if (n_flash == 2) {  // This is a bit unpleasant
        SELECT_RES(
            CASE_THEN(c_flash[0], channel0),
            CASE_THEN(c_flash[1], channel1)
            )
        {
        channel0:
            if (flash_server_operate(c_flash[0], &headers[0], qspi)) {
                SELECT_CONTINUE_NO_RESET;
            } else {
                break;
            }
        channel1:
            if (flash_server_operate(c_flash[1], &headers[1], qspi)) {
                SELECT_CONTINUE_NO_RESET;
            } else {
                break;
            }
        }
    } else {
        printstr("Too many flash channels");
        asm("clre; waiteu");
    }
}

#ifdef TEST_FLASH_SERVER_MAIN
#include <xcore/parallel.h>
#include <QuadSpecMacros.h>
#include "load_weights.h"

DECLARE_JOB(f_server, (chanend_t*, flash_t*, int,
                       fl_QSPIPorts *, fl_QuadDeviceSpec*,
                       int) );

DECLARE_JOB(f_client, (chanend_t, int));

void f_server(chanend_t c_flash[], flash_t headers[], int n_flash,
              fl_QSPIPorts *qspi, fl_QuadDeviceSpec flash_spec[],
              int n_flash_spec) {
    flash_server(c_flash, headers, n_flash, qspi, flash_spec, n_flash_spec);
}

void f_client(chanend_t c_flash, int kill) {
    int b[20];
    int a[20];
    int *data_ptrs1[1] = {a};
    int *data_ptrs2[1] = {b};
    int data_sizes_in_words[1] = {20};
    load_weights_synchronous(c_flash, data_ptrs1, data_sizes_in_words, 1, 68, 4, NULL);
    load_weights_asynchronous(c_flash, data_ptrs2, data_sizes_in_words, 1, 68, 4);
    load_weights_asynchronous_wait(c_flash);
    for(int i = 0; i < 20; i++) {
        printf("%08x %08x\n", a[i], b[i]);
    }
    if (kill) {
        load_weights_quit(c_flash);
    }
}

flash_t headers[2];

#define NFLASH_SPECS 1

fl_QuadDeviceSpec flash_spec[NFLASH_SPECS] = {
    FL_QUADDEVICE_DEFAULT //FL_QUADDEVICE_MACRONIX_MX25R6435FM2IH0
};

fl_QSPIPorts qspi = {
    PORT_SQI_CS,
    PORT_SQI_SCLK,
    PORT_SQI_SIO,
    XS1_CLKBLK_2
};


int main(void) {
    channel_t a = chan_alloc();
    channel_t b = chan_alloc();
    chanend_t fs[2] = {a.end_a, b.end_a};
    
    PAR_JOBS(
        PJOB(f_server, (fs,headers,1,&qspi,flash_spec,NFLASH_SPECS)),
        PJOB(f_client, (a.end_b, 1)));
    
    PAR_JOBS(
        PJOB(f_server, (fs,headers,2,&qspi,flash_spec,NFLASH_SPECS)),
        PJOB(f_client, (a.end_b, 1)),
        PJOB(f_client, (b.end_b, 0)));
}
#endif
