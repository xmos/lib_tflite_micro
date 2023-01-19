// Test program for experimenting with ports

#include <xs1.h>
#include <platform.h>
#include <timer.h>
#include <stdio.h>
#include <xclib.h>
#include <stdint.h>

#include "fast_flash.h"

static int best_read_adj;
static int data_offset;

// Defines for SETPADCTRL

#define DR_STR_2mA     0
#define DR_STR_4mA     1
#define DR_STR_8mA     2
#define DR_STR_12mA    3

#define PORT_PAD_CTL_SMT    0           // Schmitt off
#define PORT_PAD_CTL_SR     1           // Fast slew
#define PORT_PAD_CTL_DR_STR DR_STR_8mA  // 8mA drive
#define PORT_PAD_CTL_REN    1           // Receiver enabled
#define PORT_PAD_CTL_MODE   0x0006

#define PORT_PAD_CTL ((PORT_PAD_CTL_SMT     << 23) | \
                      (PORT_PAD_CTL_SR      << 22) | \
                      (PORT_PAD_CTL_DR_STR  << 20) | \
                      (PORT_PAD_CTL_REN     << 17) | \
                      (PORT_PAD_CTL_MODE    << 0))
                      
#define HIGH_DRIVE_8MA

// Define the clock source divide - 600MHz/800MHz core clock divided by (2*CLK_DIVIDE)
// CORE CLOCK   600MHz    800MHz
// CLK_DIVIDE      SPI_CLK
// 3            100MHz    133MHz
// 4            75MHz     100MHz
// 5            60MHz     80MHz
// 6            50MHz     66MHz

#define CLK_DIVIDE 3

extern unsigned int fl_getDataPartitionBase(void);

// Address in Flash where the training pattern starts
// How many 32 bit words are in the pattern
#define PATTERN_WORDS 8

// This function has a bit longer latency but meets the timing specs for CS_N active setup time to rising edge of clock for faster clock speeds.
#pragma unsafe arrays
static void fast_read_loop(fl_QSPIPorts &qspi, unsigned addr, unsigned mode, unsigned read_adj, unsigned read_count, unsigned read_data[], chanend ?c_data_out)
{
    
    unsigned addr_mode_wr;
    unsigned int1, int2;
    unsigned read_start_pt;
    
    // Starting data read port timer value
    read_start_pt = 27 + read_adj;

    // We shift the address left by 8 bits to align the MSB of address with MSB of the int. We or in the mode bits.
    // Buffered ports remember always shift right.
    // So LSB first
    // We need to nibble swap the address and mode word as it needs to be output MS nibble first and ports do the opposite
    {int1, int2} = unzip(byterev((addr << 8) | mode), 2);
    addr_mode_wr = zip(int2, int1, 2);
    
    // Need to set the first data output bit (MS bit of flash command) before we start the clock.
    partout(qspi.qspiSIO, 4, 0x1);
    start_clock(qspi.qspiClkblk);
    sync(qspi.qspiSIO);
    stop_clock(qspi.qspiClkblk);
    
    qspi.qspiCS <: 0; // Set CS_N low
    
    // Pre load the transfer register in the port with data. This will not go out yet because clock has not been started.
    // This data needs to be shifted as when starting the clock we will clock the first bit of data to flash before this data goes out.
    // We also need to Or in the first nibble of addr_mode_wr that we want to output.
    // This is the 7 LSB of the 0xEB instruction on dat[0], dat[3:1] = 0
    qspi.qspiSIO <: 0x01101011 | ((addr_mode_wr & 0x0000000F) << 28); 
    
    // Start the clock block running. This starts output of data and resets the port timer to 0 on the clk port.
    start_clock(qspi.qspiClkblk);

    partout(qspi.qspiSIO, 28, (addr_mode_wr >> 4)); // Immediately follow up with the remaining address and mode bits
    
    // Now we want to turn the port around at the right point.
    // At specified value of port timer we read the transfer reg word and discard, data will be junk. Exact timing of port going high-z would need simulation but it will be in the cycle specified.
    qspi.qspiSIO @ 18 :> void;
    // Now we need to read the transfer register at the correct port time so that the initial data from flash will be in there
    unsigned first_word;
    qspi.qspiSIO @ read_start_pt :> first_word;
    // All following reads will happen directly after this read with no gaps so do not need to be timed.
    if (isnull(c_data_out)) {
        read_data[0] = first_word;
        for (int i = 1; i < read_count; i++) {
            qspi.qspiSIO :> read_data[i];
        }
    } else {
        outuint(c_data_out, first_word);
        for (int i = 1; i < read_count; i++) {
            unsigned x;
            qspi.qspiSIO :> x;
            outuint(c_data_out, x);
        }
        outct(c_data_out, 1);
    }
    // Stop the clock
    stop_clock(qspi.qspiClkblk);
    
    // Put chip select back high
    qspi.qspiCS <: 1;
}

static void ports_clocks_setup(fl_QSPIPorts &qspi)
{
  
    // Define a clock source - core clock divided by (2*CLK_DIVIDE)
    configure_clock_xcore(qspi.qspiClkblk, CLK_DIVIDE);
    
    // For experimentation only do not use.
    //set_clock_fall_delay(qspi.qspiClkblk, 1);
    
    // Configure this clock to be output on the qspi.qspiSCLK port
    configure_port_clock_output(qspi.qspiSCLK, qspi.qspiClkblk);
    
    // Define all ports initially as being outputs clocked by the qspi.qspiClkblk clock. Initial state for csn is set to 1 (inactive).
    configure_out_port(qspi.qspiSIO, qspi.qspiClkblk, 0xF);
    
    // Set the drive strength to 8mA on all ports.
    // This is optional. If used it may change timing tuning so must tune with the settings we are using.
    // This will reduce rise time uncertainty and ensure ports will fully switch at 100MHz rate over PVT. (Remember our corner silicon was only skewed for core transistors not IO).
    // Setting all ports to the same drive to ensure we don't introduce any extra skew between outputs.
    // Downside is extra EMI.
#ifdef HIGH_DRIVE_8MA
    asm volatile ("setc res[%0], %1" :: "r" (qspi.qspiSCLK), "r" (PORT_PAD_CTL));
    asm volatile ("setc res[%0], %1" :: "r" (qspi.qspiSIO), "r" (PORT_PAD_CTL));
    asm volatile ("setc res[%0], %1" :: "r" (qspi.qspiCS), "r" (PORT_PAD_CTL));
#endif

}

static unsigned read_data_check[PATTERN_WORDS] = {
    0x0f0f00ff,
    0x0f0f0f0f,
    0x00ff00ff,
    0x00ff00ff,
    0x08cef731,
    0x08cef731,
    0x639c639c,
    0x639c639c
};

int fast_flash_init(fl_QSPIPorts &qspi) {
    data_offset = fl_getDataPartitionBase();
    
    // Setup the clock block and ports
    ports_clocks_setup(qspi);

    unsigned read_data_tmp[PATTERN_WORDS];
    int passing_words;
    
    // Declare the results as an array of 36 (max) chars.
    // Bit 0 sdelay, bits 1-2 read adj, bits 3-5 pad delay, bit 7 is pass/fail.
    // the index into the array becomes the nominal time.
    char results[6*CLK_DIVIDE];
    
    // So, lets run the testing and collect pass/fail results.
    // Keep an index of the time we are looping across.
    unsigned char time_index = 0;
    int pass_count = 0;
    int pass_start = 0;
    
    // This loops over the settings in such a way that the result is sequentially increasing in time in units of core clocks.
    for(int read_adj = 0; read_adj < 3; read_adj++) // Data read port time adjust
    {
        for(int sdelay = 0; sdelay < 2; sdelay++) // Sample delays
        {
            if (sdelay == 1) {
                set_port_sample_delay(qspi.qspiSIO); // Set data port to sample on a falling edge instead of rising
            } else {
                set_port_no_sample_delay(qspi.qspiSIO);
            }
            for(int pad_delay = (CLK_DIVIDE - 1); pad_delay >= 0; pad_delay--) // Pad delays (only loop over useful pad delays)
            {
                // Set input pad delay in units of core clocks
                set_pad_delay(qspi.qspiSIO, pad_delay);
                
                // Read the data with the current settings
                fast_read_loop(qspi, data_offset, 0, read_adj, PATTERN_WORDS, read_data_tmp, null);
                               // TODO: remove the +4.
                
                // Check if the data is correct
                passing_words = 0;
                for (int m = 0; m < PATTERN_WORDS; m++)
                {
                    if (read_data_tmp[m] == read_data_check[m]) {
                        passing_words++;
                    }
                }
                char setting = sdelay | (read_adj << 1) | (pad_delay << 3);
                // Store the settings and pass/fail in the results
                if (passing_words == PATTERN_WORDS) {
                    if (pass_count == 0) // This is first PASS we've seen
                    {
                        pass_start = time_index; // Record the setting index
                    }
                    pass_count++;
                    results[time_index] = setting | 1 << 7;
                } else {
                    results[time_index] = setting;
                }
                time_index++;
            }
        }
    }
    
    char best_setting = pass_start + (pass_count >> 1); // Pick the middle setting
    
    if (pass_count < 5) {
        return -1-pass_count;
    }
    
    best_read_adj  = (results[best_setting] & 0x06) >> 1;
    if (results[best_setting] & 0x01) {   // Sdelay set?
        set_port_sample_delay(qspi.qspiSIO); // Set data port to sample on a falling edge instead of rising
    } else {
        set_port_no_sample_delay(qspi.qspiSIO);
    }
    set_pad_delay(qspi.qspiSIO, (results[best_setting] & 0x38) >> 3);
    return 0;
}


void fast_flash_read(fl_QSPIPorts &qspi, unsigned addr, unsigned read_count, unsigned read_data[], chanend ?c_data_out) {
    fast_read_loop(qspi, data_offset + addr, 0, best_read_adj, read_count, read_data, c_data_out);
}

void fast_flash_read_no_chanend(fl_QSPIPorts &qspi, unsigned addr, unsigned read_count, unsigned read_data[]) {
    fast_read_loop(qspi, data_offset + addr, 0, best_read_adj, read_count, read_data, NULL);
}