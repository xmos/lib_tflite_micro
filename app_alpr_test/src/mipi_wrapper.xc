#include <xs1.h>
#include <print.h>
#include <stdio.h>
#include <stdint.h>
#include <platform.h>
#include <string.h>
#include <math.h>
#include "assert.h"
#include "i2c.h"
#include "mipi_wrapper.h"
#ifdef GC2145
#include "gc2145.h"
#else
#include "gc0310.h"
#endif
#include "mipi.h"
#include "debayer.h"
#include "yuv_to_rgb.h"
#include "subsample.h"

#define MIPI_TILE (1)

typedef enum {
    IMAGER_SAMPLE = 1,
    IMAGER_DONTSAMPLE = 0,
} imager_t;


void send_array(chanend c, int8_t * unsafe array, int bytes);

#define MIPI_BUFFER_SIZE_BYTES (SENSOR_IMAGE_WIDTH * SENSOR_IMAGE_DEPTH + 40)

#define MIPI_LINES 8
uint8_t mipiBuffer[MIPI_LINES][MIPI_BUFFER_SIZE_BYTES];

on tile[MIPI_TILE]:buffered in port:32 p_mipi_clk = XS1_PORT_1O;
on tile[MIPI_TILE]:in port p_mipi_rxa = XS1_PORT_1E;
on tile[MIPI_TILE]:in port p_mipi_rxv = XS1_PORT_1I;
on tile[MIPI_TILE]:buffered in port:32 p_mipi_rxd = XS1_PORT_8A;
on tile[MIPI_TILE]:in port p_mipi_rxs = XS1_PORT_1J;

on tile[MIPI_TILE]:clock clk_mipi = XS1_CLKBLK_1;

void exit(int);

uint32_t statusError = 0, headerError = 0, lineCountError = 0, pixelCountError = 0;
uint64_t pixelCounter = 0;
#pragma unsafe arrays
void MipiDecoupler(chanend c, chanend c_kill, chanend c_line)
{
    unsigned tailSize, ourWordCount, mipiHeader;
    int line = 0;
    
    unsafe 
    {
        while(1)
        {
            // Send out a buffer pointer to receiver thread 
            uint8_t * unsafe pt = mipiBuffer[line];
            outuint(c, (unsigned) pt);
            /* Packet receive notification - header */
            mipiHeader = inuint(c);
            outuchar(c_line, mipiHeader);
            
            line = (line + 1) & (MIPI_LINES-1);
            /* Long packet */
            if(mipiHeader & 0x30) {
                ourWordCount = inuint(c);
                tailSize = inuint(c);
                if (ourWordCount != (SENSOR_IMAGE_WIDTH/2) || tailSize != 24)
                {
                    printintln(ourWordCount);
                    printintln(tailSize);
                    pixelCountError++;
                }
            }
        }
    }
}


uint32_t frame_time = 0, line_time = 0;

struct decoupler_buffer 
{
    uint32_t full_image[SUBSAMPLE_MAX_OUTPUT_HEIGHT*SUBSAMPLE_MAX_OUTPUT_WIDTH*3 / sizeof(uint32_t)];
    int serial;
    int8_t x_coefficients[32*SUBSAMPLE_MAX_OUTPUT_WIDTH*3];
    int8_t y_coefficients[16*SUBSAMPLE_MAX_OUTPUT_HEIGHT*SUBSAMPLE_MAX_WINDOW_SIZE];
    uint32_t x_strides[SUBSAMPLE_MAX_OUTPUT_WIDTH];
    uint32_t y_strides[SUBSAMPLE_MAX_OUTPUT_HEIGHT];
} decoupler[1];

unsafe
{
    struct decoupler_buffer * unsafe decoupler_r = decoupler;
}

#define V_OFFSET      (RAW_IMAGE_HEIGHT/2)

#pragma unsafe arrays

int8_t subsample_x_output_buffer[5][3][160];

extern void xor_top_bits(uint8_t * unsafe pt, uint32_t words, uint32_t start_x);

void MipiImager(chanend c_line, chanend c_decoupler, chanend ?c_decoupler2 /*chanend c_l0*/)
{
    int line = 0;
    int lineCount = 0;
    int last_sof = 0, now = 0, start_of_frame = 0;
    int errors = 0;
    int grabbing = 0;
    uint8_t new_grabbing = 0;
    int start_x = 0;
    int end_x = 0;
    int start_y = 0;
    int end_y = 0;
    int width = RAW_IMAGE_WIDTH;
    int cur_x_line = 0;
    int yindex = 0;
    int output_line_cnt = 0;
    int required_width, required_height;
    int8_t *line4 = (int8_t *) subsample_x_output_buffer[0];
    int8_t *line3 = (int8_t *) subsample_x_output_buffer[1];
    int8_t *line2 = (int8_t *) subsample_x_output_buffer[2];
    int8_t *line1 = (int8_t *) subsample_x_output_buffer[3];
    int8_t *line0 = (int8_t *) subsample_x_output_buffer[4];
    unsafe 
    {
        while(1)
        {
            uint8_t * unsafe pt = mipiBuffer[line];
            uint8_t header_byte;
            select {
                case inuchar_byref(c_line, header_byte):
                    int header = header_byte & 0x3f;
                    line = (line + 1) & 7;
                    if (header == 0)      // Start of frame
                    {
                        asm volatile("gettime %0" : "=r" (start_of_frame));
                        lineCount = 0;
                        grabbing = new_grabbing;
                        new_grabbing = 0;
                    } 
                    else if (header == 1) // End of frame
                    {   
                        asm volatile("gettime %0" : "=r" (now));
                        uint32_t * unsafe ft = &frame_time;
                        *ft = now - last_sof;
                        last_sof = now;
                        uint32_t * unsafe lt = &line_time;
                        *lt = (now - start_of_frame) / (lineCount+1);
                        if (lineCount != SENSOR_IMAGE_HEIGHT)
                        {
                            lineCountError++;
                        }
                        pixelCounter += (SENSOR_IMAGE_WIDTH*SENSOR_IMAGE_HEIGHT);
                    } 
                    else if (header == 0x1E) // YUV422
                    {
                        if (grabbing) {
                            subsample_x(subsample_x_output_buffer[cur_x_line], (uint8_t *)pt,
                                        decoupler_r -> x_coefficients, decoupler_r -> x_strides, required_width);
                            line4 = line3;
                            line3 = line2;
                            line2 = line1;
                            line1 = line0;
                            line0 = (int8_t *)&subsample_x_output_buffer[cur_x_line][0][0];
                            cur_x_line++;
                            if (cur_x_line == 5) {
                                cur_x_line = 0;
                            }
                            while(lineCount == decoupler_r -> y_strides[output_line_cnt]
                                && output_line_cnt != required_height) {
                                subsample_y((decoupler_r->full_image, int8_t[])+output_line_cnt*required_width*3,
                                            line4,
                                            line3,
                                            line2,
                                            line1,
                                            line0,
                                            &decoupler_r -> y_coefficients[yindex], required_width);
                                output_line_cnt++;
                                yindex += 16*5;
                            }
                            if (output_line_cnt == required_height)
                            {
                                output_line_cnt = 0;
                                yindex = 0;
                                outuchar(c_decoupler, 0);
                                grabbing = 0;
                            }
                        }
                        lineCount++;
                        if(pt[SENSOR_IMAGE_HEIGHT*SENSOR_IMAGE_DEPTH] != 0)
                        {
                            statusError++;
                        }
                    } 
                    else if (header == 0x12)  // Embedded data - ignore
                    {
                        if(pt[SENSOR_IMAGE_HEIGHT*SENSOR_IMAGE_DEPTH] != 0)
                        {
                            statusError++;
                        }
                    }
                    else 
                    {
                        printstr("# ********"); printhexln(header);
                        errors++;
                        if (errors > 10) {
                            exit(1);
                        }
                    }
                    int error = header_byte & 0xc0;
                    if (error)
                    {
                        headerError++;
                    }
                    break;
                case inuchar_byref(c_decoupler, new_grabbing):
                    start_x   = inuint(c_decoupler);
                    end_x = inuint(c_decoupler);
                    start_y   = inuint(c_decoupler);
                    end_y     = inuint(c_decoupler);
                    required_width  = inuint(c_decoupler);
                    required_height = inuint(c_decoupler);
                    width = end_x - start_x;
                    new_grabbing = 1;
                    break;
            }
        }
    }
}


#pragma unsafe arrays
void acquire_command_handler_single(chanend c_debayerer, chanend c_acquire, int start_x)
{
    int end_x, start_y, end_y, required_width, required_height;
    c_acquire :> end_x;
    c_acquire :> start_y;
    c_acquire :> end_y;
    c_acquire :> required_width;
    c_acquire :> required_height;
    unsafe 
    {
        build_y_coefficients_strides(decoupler_r -> y_coefficients, decoupler_r -> y_strides, start_y, end_y, required_height);
        build_x_coefficients_strides(decoupler_r -> x_coefficients, decoupler_r -> x_strides, start_x, end_x, required_width);
        outuchar(c_debayerer, IMAGER_SAMPLE);        // Tell collector to grab image
        outuint(c_debayerer, start_x);               // Tell collector size
        outuint(c_debayerer, end_x);                 // Tell collector size
        outuint(c_debayerer, start_y);               // Tell collector size
        outuint(c_debayerer, end_y);                 // Tell collector size
        outuint(c_debayerer, required_width);        // Tell collector size
        outuint(c_debayerer, required_height);       // Tell collector size
        inuchar(c_debayerer);                        // Wait for image collector ready
        
        send_array(c_acquire, (int8_t * unsafe)decoupler_r -> full_image, required_width * required_height * 3);
    }
}

void acquire_command_handler(chanend c_debayerer, chanend c_acquire[], int n_acquire)
{
    int x;
    while(1) {
        select {
            case (int i = 0; i < n_acquire; i++) c_acquire[i] :> x:
                acquire_command_handler_single(c_debayerer, c_acquire[i], x);
                break;
        }
    }
}

#define TEST_DEMUX_DATATYPE (0)
#define TEST_DEMUX_MODE     (0x80)              // bias, make all bytes negative.
#define TEST_DEMUX_EN       (0)
#define DELAY_MIPI_CLK      (1)

void mipi_main(client interface i2c_master_if i2c, chanend c_acquire[], int n_acquire)
{
    chan c;
    chan c_kill, c_img, c_ctrl;

#define MIPI_CLK_DIV 1
#define MIPI_CFG_CLK_DIV 3
    
    configure_in_port_strobed_slave(p_mipi_rxd, p_mipi_rxv, clk_mipi);
    set_clock_src(clk_mipi, p_mipi_clk);

    /* Sample on falling edge - shim outputting on rising */
#ifdef DELAY_MIPI_CLK
    set_clock_rise_delay(clk_mipi, DELAY_MIPI_CLK);
#else
    set_clock_rise_delay(clk_mipi, 0);
#endif
    set_pad_delay(p_mipi_rxa, 1);
    start_clock(clk_mipi);
    write_node_config_reg(tile[MIPI_TILE], XS1_SSWITCH_MIPI_DPHY_CFG3_NUM ,
#ifdef GC2145
                          0x7E42  // two lanes
#else
                          0x3E42  // one lane
#endif
        );
#ifdef GC2145
    if (gc2145_stream_start(i2c) != 0)
#else
    if (gc0310_stream_start(i2c) != 0)
#endif
    {
        printstr("Stream start failed\n");
    }

    timer tmr;  // Sensor may have been streaming in some other mode
    int t;      // Wait for 1 second for old frames to be gone.
    tmr :> t; tmr when timerafter(t + 100000000) :> void;
    par 
    {
        MipiReceive(tile[MIPI_TILE], 1, c, p_mipi_rxd, p_mipi_rxa, c_kill,
                    TEST_DEMUX_EN, TEST_DEMUX_DATATYPE, TEST_DEMUX_MODE,
                    MIPI_CLK_DIV, MIPI_CFG_CLK_DIV);
        MipiDecoupler(c, c_kill, c_img);

        MipiImager(c_img, c_ctrl, null);
        acquire_command_handler(c_ctrl, c_acquire, n_acquire);
    }
    i2c.shutdown();
}
