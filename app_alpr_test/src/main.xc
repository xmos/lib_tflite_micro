// Copyright (c) 2020, XMOS Ltd, All rights reserved

#include <platform.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xscope.h>
#include <xclib.h>
#include <stdint.h>
#include "flash.h"
#include "mipi_wrapper.h"
#include "i2c.h"
#include "box_calculation.h"
#include "uart.h"
#include "gpio.h"
#include "wrapper.h"
#include <print.h>

on tile[0]: port p_scl = XS1_PORT_1N;
on tile[0]: port p_sda = XS1_PORT_1O;
on tile[0]: port p_uart = PORT_LEDS;

#define ORIGIN_X_INSIDE_SENSOR ((SENSOR_IMAGE_WIDTH - WIDTH_ON_SENSOR)/2)
#define ORIGIN_Y_INSIDE_SENSOR ((SENSOR_IMAGE_HEIGHT - HEIGHT_ON_SENSOR)/2)

void read_array(chanend c_acquire, int8_t * unsafe data, int bytes) {
    for(int i = 0; i < bytes; i++) {
        unsafe {
            c_acquire :> data[i];
        }
    }
}

void send_array(chanend c_acquire, int8_t * unsafe data, int bytes) {
    for(int i = 0; i < bytes; i++) {
        unsafe {
            c_acquire <: data[i];
        }
    }
}

void ai_runner(chanend c_acquire, client interface uart_tx_if uart_tx, chanend c_flash) {
    uint32_t bbox[4];
    detect_rcgn_init(c_flash);
    unsafe {
    int8_t * unsafe detect_in = detect_get_input();
    int8_t * unsafe detect_out = detect_get_output();
    int8_t * unsafe rcgn_in = rcgn_get_input();
    int8_t * unsafe rcgn_out = rcgn_get_output();

    while(1) {
        int status;
        timer tmr; int t0;
        tmr :> t0;
        tmr when timerafter(t0+200000000) :> void;
        c_acquire <: ORIGIN_X_INSIDE_SENSOR + 0;
        c_acquire <: ORIGIN_X_INSIDE_SENSOR + WIDTH_ON_SENSOR;
        c_acquire <: ORIGIN_Y_INSIDE_SENSOR + 0;
        c_acquire <: ORIGIN_Y_INSIDE_SENSOR + HEIGHT_ON_SENSOR;
        c_acquire <: 128;
        c_acquire <: 128;
        read_array(c_acquire, detect_in, 128*128*3);
        if(0) {printf("P3\n128 128 255\n");
        for(int i = 0; i < 128*128*3; i++) {
            printf("%d ", detect_in[i]+128);
            if ((i&127) == 127) printf("\n");
            timer tmr;
            int t0;
            tmr :> t0; tmr when timerafter(t0+100) :> void;
        }
        printf("\n");}
        wrapper_detect_invoke();
        int val = box_calculation(bbox, detect_out, 
                                  WIDTH_ON_SENSOR, HEIGHT_ON_SENSOR);

        for(int i = 0 ; i < 4; i++) {
            printint(bbox[i]);
            printchar(' ');
        }
        printint(val);
        printchar('\n');
        if (1) {        if (val < -100) {
            printstr("Value too small\n");
            continue;
        }
        if (bbox[1] - bbox[0] < 128) {
            printstr("Width too small\n");
            continue;
        }
        if (bbox[3] - bbox[2] < 32) {
            printstr("Height too small\n");
            continue;
        }
        }
        c_acquire <: ORIGIN_X_INSIDE_SENSOR + bbox[0];
        c_acquire <: ORIGIN_X_INSIDE_SENSOR + bbox[1];
        c_acquire <: ORIGIN_Y_INSIDE_SENSOR + bbox[2];
        c_acquire <: ORIGIN_Y_INSIDE_SENSOR + bbox[3];
        c_acquire <: 128;
        c_acquire <: 32;
        read_array(c_acquire, rcgn_in, 128*32*3);
        if (0) {        printf("P3\n128 32 255\n");
        for(int i = 0; i < 128*32*3; i++) {
            printf("%d ", rcgn_in[i]+128);
            if ((i&127) == 127) printf("\n");
            timer tmr;
            int t0;
            tmr :> t0; tmr when timerafter(t0+100) :> void;
        }
        printf("\n");}
        wrapper_rcgn_invoke();
        printf("P2\n66 16 255\n");
        for(int i = 0; i < 66*16; i++) {
            printf("%d ", rcgn_out[i]+128);
            if ((i%66) == 65) printf("\n");
            timer tmr;
            int t0;
            tmr :> t0; tmr when timerafter(t0+100) :> void;
        }
        printf("\n");

        char ocr_outputs[17];
        int len = ocr_calculation(ocr_outputs, rcgn_out);
        printstr(">>>");
        for(int i = 0; i < 12; i++) {
            if (ocr_outputs[i] == '\0') {
                break;
            }
//            uart_tx.write(ocr_outputs[i]);
            printchar(ocr_outputs[i]);
        }
        printstr("<<<\n");
        printstr("Grabbed\n");
        tmr :> t0;
        tmr when timerafter(t0+200000000) :> void;
    }
    }
}

#define FL_QUADDEVICE_MACRONIX_MX25R6435FM2IH0 \
{ \
    16,                     /* MX25R6435FM2IH0 */ \
    256,                    /* page size */ \
    32768,                  /* num pages */ \
    3,                      /* address size */ \
    3,                      /* log2 clock divider */ \
    0x9F,                   /* QSPI_RDID */ \
    0,                      /* id dummy bytes */ \
    3,                      /* id size in bytes */ \
    0xC22817,               /* device id */ \
    0x20,                   /* QSPI_SE */ \
    4096,                   /* Sector erase is always 4KB */ \
    0x06,                   /* QSPI_WREN */ \
    0x04,                   /* QSPI_WRDI */ \
    PROT_TYPE_NONE,         /* no protection */ \
    {{0,0},{0x00,0x00}},    /* QSPI_SP, QSPI_SU */ \
    0x02,                   /* QSPI_PP */ \
    0xEB,                   /* QSPI_READ_FAST */ \
    1,                      /* 1 read dummy byte */ \
    SECTOR_LAYOUT_REGULAR,  /* mad sectors */ \
    {4096,{0,{0}}},         /* regular sector sizes */ \
    0x05,                   /* QSPI_RDSR */ \
    0x01,                   /* QSPI_WRSR */ \
    0x01,                   /* QSPI_WIP_BIT_MASK */ \
}

#define FL_QUADDEVICE_MACRONIX_MX25R3235FM1IH0 \
{ \
    15,                     /* MX25R3235FM1IH0 */ \
    256,                    /* page size */ \
    32768,                  /* num pages */ \
    3,                      /* address size */ \
    3,                      /* log2 clock divider */ \
    0x9F,                   /* QSPI_RDID */ \
    0,                      /* id dummy bytes */ \
    3,                      /* id size in bytes */ \
    0xC22816,               /* device id */ \
    0x20,                   /* QSPI_SE */ \
    4096,                   /* Sector erase is always 4KB */ \
    0x06,                   /* QSPI_WREN */ \
    0x04,                   /* QSPI_WRDI */ \
    PROT_TYPE_NONE,         /* no protection */ \
    {{0,0},{0x00,0x00}},    /* QSPI_SP, QSPI_SU */ \
    0x02,                   /* QSPI_PP */ \
    0xEB,                   /* QSPI_READ_FAST */ \
    1,                      /* 1 read dummy byte */ \
    SECTOR_LAYOUT_REGULAR,  /* mad sectors */ \
    {4096,{0,{0}}},         /* regular sector sizes */ \
    0x05,                   /* QSPI_RDSR */ \
    0x01,                   /* QSPI_WRSR */ \
    0x01,                   /* QSPI_WIP_BIT_MASK */ \
}

fl_QuadDeviceSpec flash_spec[] = {
    FL_QUADDEVICE_MACRONIX_MX25R6435FM2IH0,
    //FL_QUADDEVICE_MACRONIX_MX25R3235FM1IH0
};

on tile[0]: fl_QSPIPorts qspi = {
    PORT_SQI_CS,
    PORT_SQI_SCLK,
    PORT_SQI_SIO,
    XS1_CLKBLK_2
};

on tile[1]: port p_reset_camera = XS1_PORT_1P;

int main(void) 
{
    chan c_flash[1];
    interface uart_tx_if i_tx;
    output_gpio_if i_gpio_tx[1];
    chan c_acquire;
    i2c_master_if i2c[1];
    par 
    {
        on tile[0]: {
            ai_runner(c_acquire, i_tx, c_flash[0]);
        }
        
        on tile[0]: {
            flash_t headers[1];
            flash_server(c_flash, headers, 1, qspi, flash_spec, 1);
        }
        
        on tile[0]: {
            char pin_map[1] = {2};
            output_gpio(i_gpio_tx, 1, p_uart, pin_map);
        }
        
        on tile[0]: uart_tx(i_tx, null,
                            10, UART_PARITY_NONE, 8, 1,
                            i_gpio_tx[0]);
        
        on tile[0]: i2c_master(i2c, 1, p_scl, p_sda, 400);

        on tile[1]: {
            p_reset_camera @ 0 <: 0;
            p_reset_camera @ 1000 <: ~0;
            p_reset_camera @ 2000 <: 0;
            mipi_main(i2c[0], c_acquire);
        }
    }
    return 0;
}
