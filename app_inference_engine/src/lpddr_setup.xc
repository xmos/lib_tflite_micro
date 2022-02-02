#include <platform.h>


#include <stdio.h>
#include <stdlib.h>

#define EXPLORERCOMPILE 1

#define PRINT_FN(...) printf(__VA_ARGS__); printf("\n");

//#define this_tile(x) tile[0]

#define TerminateFail(v) exit(v)

// N.B. defines some constants for timer waits, assumption being that 1 timer tick = 10ns (i.e. 100MHz reg=f clock)
// let App PLL settle for 10us
#define APP_PLL_STARTUP_TIME 1000
// LPDDR1 devices need 200us to settle after clocks start (make it slightly larger for safety)
#define LPDDR_INIT_WAIT_TIME 22000

#ifdef EXPLORERCOMPILE
  // these are the value to be programmed into the GPIOs used for LPDDR, to give suitable settings for Explorer board
  // [6] = Schmitt enable, [5] = Slew, [4:3] = drive strength, [2:1] = pull option, [0] = read enable
  // we want 8mA drive (2'b10), fast slew for Explorer board
  #define BOARD_DDR_PADCTRL_CLK_VAL   0x30  // 0b0110000 8mA-drive, fast-slew output
  #define BOARD_DDR_PADCTRL_CKE_VAL   0x30  // 0b0110000 8mA-drive, fast-slew output
  #define BOARD_DDR_PADCTRL_CS_N_VAL  0x30  // 0b0110000 8mA-drive, fast-slew output
  #define BOARD_DDR_PADCTRL_WE_N_VAL  0x30  // 0b0110000 8mA-drive, fast-slew output
  #define BOARD_DDR_PADCTRL_CAS_N_VAL 0x30  // 0b0110000 8mA-drive, fast-slew output
  #define BOARD_DDR_PADCTRL_RAS_N_VAL 0x30  // 0b0110000 8mA-drive, fast-slew output
  #define BOARD_DDR_PADCTRL_ADDR_VAL  0x30  // 0b0110000 8mA-drive, fast-slew output
  #define BOARD_DDR_PADCTRL_BA_VAL    0x30  // 0b0110000 8mA-drive, fast-slew output
  #define BOARD_DDR_PADCTRL_DQ_VAL    0x31  // 0b0110000 8mA-drive, fast-slew bidir
  #define BOARD_DDR_PADCTRL_DQS_VAL   0x31  // 0b0110000 8mA-drive, fast-slew bidir
  #define BOARD_DDR_PADCTRL_DM_VAL    0x30  // 0b0110000 8mA-drive, fast-slew output
#else
  // we want 12mA drive (2'b10), fast slew for Bringup board using DDR mezzanine
  #define BOARD_DDR_PADCTRL_CLK_VAL   0x38  // 0b0111000 12mA-drive, fast-slew output
  #define BOARD_DDR_PADCTRL_CKE_VAL   0x38  // 0b0111000 12mA-drive, fast-slew output
  #define BOARD_DDR_PADCTRL_CS_N_VAL  0x38  // 0b0111000 12mA-drive, fast-slew output
  #define BOARD_DDR_PADCTRL_WE_N_VAL  0x38  // 0b0111000 12mA-drive, fast-slew output
  #define BOARD_DDR_PADCTRL_CAS_N_VAL 0x38  // 0b0111000 12mA-drive, fast-slew output
  #define BOARD_DDR_PADCTRL_RAS_N_VAL 0x38  // 0b0111000 12mA-drive, fast-slew output
  #define BOARD_DDR_PADCTRL_ADDR_VAL  0x38  // 0b0111000 12mA-drive, fast-slew output
  #define BOARD_DDR_PADCTRL_BA_VAL    0x38  // 0b0111000 12mA-drive, fast-slew output
  #define BOARD_DDR_PADCTRL_DQ_VAL    0x39  // 0b0111001 12mA-drive, fast-slew bidir
  #define BOARD_DDR_PADCTRL_DQS_VAL   0x39  // 0b0111001 12mA-drive, fast-slew bidir
  #define BOARD_DDR_PADCTRL_DM_VAL    0x38  // 0b0111000 12mA-drive, fast-slew output

//  #define BOARD_DDR_PADCTRL_DQ_VAL    0x9  // 0b01 01 00 1 4mA-drive, fast-slew bidir
#endif



//--------------------------------------------------------------------------------------------------------------------
// Constant definitions for LPDDR Controller CSRs
//--------------------------------------------------------------------------------------------------------------------
#define LPDDR_CONTROLLER_CSR_BASE 0xC000  // address in SSwitch register space  NICETOHAVE - use proper constant

#define SSWITCH_SS_LPDDR_CONTROLLER_CONFIG_NUM 19 // NICETOHAVE remove when defined in toolchain

#define PSC_CTRL0_EXTMEM_ENABLE 0 // NICETOHAVE remove when defined in toolchain

#define TB_LPDDR_SIZE 128*1024*1024 // LPDDR size in bytes

// these are timing targets for Micron DDR device, which need to be achieved for different clock periods
#define DDR_TREFI_IN_NS 7790  // 7.8us max is datasheet number for 256Mbit and above
#define DDR_TRAS_IN_NS    40  // min
#define DDR_TXSR_IN_NS   113  // datasheet says 112.5 min.
#define DDR_TWR_IN_NS     15  // min

#define DDR_TRC_IN_NS     55  // min
#define DDR_TRCD_IN_NS    15  // min  
#define DDR_TRP_IN_NS     15  // min
#define DDR_TRFC_IN_NS    72  // min 
#define DDR_TRRD_IN_NS    10  // min 




// register addresses (offsets from base)
#define LPDDR_CSR_IID_ENABLE                 0  // RW, 16bits, reset 0
#define LPDDR_CSR_IID_0_7                    1  // RW, 32bits, reset 0
#define LPDDR_CSR_IID_8_15                   2  // RW, 32bits, reset 0
#define LPDDR_CSR_QUEUE_CONT                 3  // RW,  1bit,  reset to 0
#define LPDDR_CSR_RO_COMMAND_QUEUE_PRIORITY  8  // RW,  3bits, reset 7
#define LPDDR_CSR_RW_COMMAND_QUEUE_PRIORITY  9  // RW,  6bits, reset 5
#define LPDDR_CSR_ARBITRATION_TIMEOUT       10  // RW,  4bits, reset 4
#define LPDDR_CSR_ARBITRATION_MTG_COMMAND   16  // RW,  6bits, reset 0
#define LPDDR_CSR_DLL_CONTROL               20  // RW, 22bits, reset 0
#define LPDDR_CSR_DLL_MEASUREMENT_STATUS    21  // RO, 28bits, reset 0
#define LPDDR_CSR_DLL_MANUAL_CONTROL        22  // RW, 32bits, reset 0
#define LPDDR_CSR_DLL_PHY_CALIBRATION_DATA  23  // RW, 24bits, reset 0x404040
#define LPDDR_CSR_PHY_CONTROL               29  // RW, 14bits, reset 0x2101
#define LPDDR_CSR_LMR_OPCODE                30  // RW, 14bits, reset 'b011_0100
#define LPDDR_CSR_EMR_OPCODE                31  // RW, 14bits, reset 0
#define LPDDR_CSR_PROTOCOL_ENGINE_CONF_0    32  // RW, 24bits, reset 0x33c30b
#define LPDDR_CSR_PROTOCOL_ENGINE_CONF_1    33  // RW, 18bits, reset 0x0ee4b
#define LPDDR_CSR_PROTOCOL_ENGINE_STATUS    34  // RO, not specified or implemented, reads 0

#define LPDDR_CSR_UNIMPLEMENTED_4            4  // RO, 0
#define LPDDR_CSR_UNIMPLEMENTED_5            5  // RO, 0
#define LPDDR_CSR_UNIMPLEMENTED_6            6  // RO, 0
#define LPDDR_CSR_UNIMPLEMENTED_7            7  // RO, 0

#define LPDDR_CSR_UNIMPLEMENTED_11          11  // RO, 0
#define LPDDR_CSR_UNIMPLEMENTED_12          12  // RO, 0
#define LPDDR_CSR_UNIMPLEMENTED_13          13  // RO, 0
#define LPDDR_CSR_UNIMPLEMENTED_14          14  // RO, 0
#define LPDDR_CSR_UNIMPLEMENTED_15          15  // RO, 0

#define LPDDR_CSR_UNIMPLEMENTED_24          24  // RO, 0
#define LPDDR_CSR_UNIMPLEMENTED_25          25  // RO, 0
#define LPDDR_CSR_UNIMPLEMENTED_26          26  // RO, 0
#define LPDDR_CSR_UNIMPLEMENTED_27          27  // RO, 0
#define LPDDR_CSR_UNIMPLEMENTED_28          28  // RO, 0

#define LPDDR_CSR_UNIMPLEMENTED_35          35  // RO, 0

#define LPDDR_CSR_UNIMPLEMENTED_17          17  // RO, 0
#define LPDDR_CSR_UNIMPLEMENTED_18          18  // RO, 0
#define LPDDR_CSR_UNIMPLEMENTED_19          19  // RO, 0

// field & bit position definitions
#define LPDDR_CSR_DLL_CONTROL_ENABLE_BITPOS              0
#define LPDDR_CSR_DLL_CONTROL_ONE_SHOT_MODE_BITPOS       1
#define LPDDR_CSR_DLL_CONTROL_ONE_SHOT_TRIGGER_BITPOS    2
#define LPDDR_CSR_DLL_CONTROL_ONE_SHOT_COARSE_STEP_LSB   3
#define LPDDR_CSR_DLL_CONTROL_ONE_SHOT_START_VAL_BITPOS 10

#define LPDDR_CSR_DLL_MEASUREMENT_STATUS_ONE_SHOT_RUN_BITPOS    0
#define LPDDR_CSR_DLL_MEASUREMENT_STATUS_ONE_SHOT_FINISH_BITPOS 1

#define LPDDR_CSR_DLL_MEASUREMENT_STATUS_ONE_SHOT_MEASURE_START_LSB    8
#define LPDDR_CSR_DLL_MEASUREMENT_STATUS_ONE_SHOT_MEASURE_START_WIDTH 10
#define LPDDR_CSR_DLL_MEASUREMENT_STATUS_ONE_SHOT_MEASURE_END_LSB     18
#define LPDDR_CSR_DLL_MEASUREMENT_STATUS_ONE_SHOT_MEASURE_END_WIDTH   10

#define LPDDR_CSR_PHASE_DETECT_BITPOS    7

#define LPDDR_CSR_ARBITRATION_MTG_INIT_BITPOS             0
#define LPDDR_CSR_ARBITRATION_MTG_LMR_BITPOS              1
#define LPDDR_CSR_ARBITRATION_MTG_POWER_DOWN_EN_BITPOS    2
#define LPDDR_CSR_ARBITRATION_MTG_SELF_REFRESH_EN_BITPOS  3
#define LPDDR_CSR_ARBITRATION_MTG_DLL_PAUSE_BITPOS        4
#define LPDDR_CSR_ARBITRATION_MTG_FSM_IDLE_BITPOS         5


//--------------------------------------------------------------------------------------------------------------------
// Constant definitions for padctrl block CSRs
// FIXME these constants should be auto-generated from XML probably...
//--------------------------------------------------------------------------------------------------------------------
#define PADCTRL_CSR_BASE   0xD000   // address in SSwitch register space  


// register addresses (offsets from base)
#define PADCTRL_CSR_CLK_ADDR   0x00
#define PADCTRL_CSR_CKE_ADDR   0x01
#define PADCTRL_CSR_CS_N_ADDR  0x02
#define PADCTRL_CSR_WE_N_ADDR  0x03
#define PADCTRL_CSR_CAS_N_ADDR 0x04
#define PADCTRL_CSR_RAS_N_ADDR 0x05
#define PADCTRL_CSR_ADDR_ADDR  0x06
#define PADCTRL_CSR_BA_ADDR    0x07
#define PADCTRL_CSR_DQ_ADDR    0x08
#define PADCTRL_CSR_DQS_ADDR   0x09
#define PADCTRL_CSR_DM_ADDR    0x0a

#if 0
static unsigned getTileId(void)
{
  unsigned bootconfig;
  unsigned res = XS1_PS_BOOT_CONFIG;

  asm volatile("get %0, ps[%1]" : "=r" (bootconfig): "r" (res));

  return (bootconfig >> XS1_BOOT_CONFIG_PROCESSOR_SHIFT) & 0x1;
}
#endif

int Si_SetupLPDDRPadCtrl () {
  unsigned x;
  read_sswitch_reg(get_tile_id(tile[0]), PADCTRL_CSR_BASE + PADCTRL_CSR_CLK_ADDR,   x);
  if (x == BOARD_DDR_PADCTRL_CLK_VAL) {
    return 1;
  }

  write_sswitch_reg(get_tile_id(tile[0]), PADCTRL_CSR_BASE + PADCTRL_CSR_CLK_ADDR,   BOARD_DDR_PADCTRL_CLK_VAL);
  write_sswitch_reg(get_tile_id(tile[0]), PADCTRL_CSR_BASE + PADCTRL_CSR_CKE_ADDR,   BOARD_DDR_PADCTRL_CKE_VAL);
  write_sswitch_reg(get_tile_id(tile[0]), PADCTRL_CSR_BASE + PADCTRL_CSR_CS_N_ADDR,  BOARD_DDR_PADCTRL_CS_N_VAL);
  write_sswitch_reg(get_tile_id(tile[0]), PADCTRL_CSR_BASE + PADCTRL_CSR_WE_N_ADDR,  BOARD_DDR_PADCTRL_WE_N_VAL);
  write_sswitch_reg(get_tile_id(tile[0]), PADCTRL_CSR_BASE + PADCTRL_CSR_CAS_N_ADDR, BOARD_DDR_PADCTRL_CAS_N_VAL);
  write_sswitch_reg(get_tile_id(tile[0]), PADCTRL_CSR_BASE + PADCTRL_CSR_RAS_N_ADDR, BOARD_DDR_PADCTRL_RAS_N_VAL);
  write_sswitch_reg(get_tile_id(tile[0]), PADCTRL_CSR_BASE + PADCTRL_CSR_ADDR_ADDR,  BOARD_DDR_PADCTRL_ADDR_VAL);
  write_sswitch_reg(get_tile_id(tile[0]), PADCTRL_CSR_BASE + PADCTRL_CSR_BA_ADDR,    BOARD_DDR_PADCTRL_BA_VAL);
  write_sswitch_reg(get_tile_id(tile[0]), PADCTRL_CSR_BASE + PADCTRL_CSR_DQ_ADDR,    BOARD_DDR_PADCTRL_DQ_VAL);
  write_sswitch_reg(get_tile_id(tile[0]), PADCTRL_CSR_BASE + PADCTRL_CSR_DQS_ADDR,   BOARD_DDR_PADCTRL_DQS_VAL);
  write_sswitch_reg(get_tile_id(tile[0]), PADCTRL_CSR_BASE + PADCTRL_CSR_DM_ADDR,    BOARD_DDR_PADCTRL_DM_VAL);
  return 0;
}


//--------------------------------------------------------------------------------------------------------------------
// Routine to pause traffic ready for a DLL recalibrate
// Returns 1 for ok
//--------------------------------------------------------------------------------------------------------------------
int LPDDRPauseTraffic () {
  unsigned wdata;
  unsigned rdata;
  int      retval;
  int      i;
//  unsigned node;
  
  //node = this_tile();
  
  wdata = 1<<LPDDR_CSR_ARBITRATION_MTG_DLL_PAUSE_BITPOS;
  if (!write_sswitch_reg(get_tile_id(tile[0]), LPDDR_CONTROLLER_CSR_BASE + LPDDR_CSR_ARBITRATION_MTG_COMMAND, wdata)) TerminateFail(0xdd7);

  // poll until we see pausing has occurred
  for (i=0; i<100; i++) {
    retval = read_sswitch_reg(get_tile_id(tile[0]), LPDDR_CONTROLLER_CSR_BASE + LPDDR_CSR_ARBITRATION_MTG_COMMAND, rdata);
    if (!retval) {
      PRINT_FN("FAIL: Fail status on read_sswitch_reg of ARBITRATION_MTG_COMMAND");
      TerminateFail(1);
    }  
    if (rdata>>LPDDR_CSR_ARBITRATION_MTG_DLL_PAUSE_BITPOS & 1) {
      break;
    }
    if (i==99) {
      PRINT_FN("LPDDRPauseTraffic: Did not see dll_pause from ARBITRATION_MTG_COMMAND polling");
      return 0;
    }  
  }
  return 1;
}

//--------------------------------------------------------------------------------------------------------------------
// Routine to unpause (resume) traffic after a DLL recalibrate
// Returns 1 for ok
//--------------------------------------------------------------------------------------------------------------------
int LPDDRUnpauseTraffic () {
//  unsigned node;  
//  node = this_tile();
  if (!write_sswitch_reg(get_tile_id(tile[0]), LPDDR_CONTROLLER_CSR_BASE + LPDDR_CSR_ARBITRATION_MTG_COMMAND, 0)) TerminateFail(0xdd7);
  return 1;
}



//--------------------------------------------------------------------------------------------------------------------
// Routine to recalibrate the DLLs
// It takes the Master DLL measurement of a clock period, and the Write DLL actual setting
// Returns 1 for ok
//--------------------------------------------------------------------------------------------------------------------
int LPDDRRecal (int period, int write_dll_setting) {
  int read_recal_val;
  int wdata;
  unsigned rdata;
  int retval;
//  int node;
  
//  node = this_tile();

  // pause traffic
  if (!LPDDRPauseTraffic()) return 0;


  // rather then divide the period by exactly 4, use an adjustment derived from DLL margin checking to align
  // the read data & strobe at a point 92% of the default number, which is the midpoint of the range of DLL
  // values where reading actually works
  read_recal_val = (period * 92) / 400;
  
  if (read_recal_val>255) {
    PRINT_FN("ERROR: LPDDRRecal: rec_val %d out of range", read_recal_val);
    return 0;
  }

  if (write_dll_setting>255) {
    PRINT_FN("ERROR: LPDDRRecal: write_dll_setting %d out of range", write_dll_setting);
    return 0;
  }

//  PRINT_FN("DEBUG: LPDDRRecal: supplied write_dll_setting = %d, calculated read setting = %d", write_dll_setting, read_recal_val);

  wdata = write_dll_setting | read_recal_val<<8 | read_recal_val<<16;   // same value to 2 read slave DLLs

  if (!write_sswitch_reg(get_tile_id(tile[0]), LPDDR_CONTROLLER_CSR_BASE + LPDDR_CSR_DLL_PHY_CALIBRATION_DATA, wdata)) TerminateFail(0xdd8);

  // just read back reg to check write worked ok
  retval = read_sswitch_reg(get_tile_id(tile[0]), LPDDR_CONTROLLER_CSR_BASE + LPDDR_CSR_DLL_PHY_CALIBRATION_DATA, rdata);
  if (!retval) {
    PRINT_FN("FAIL: LPDDRRecal(): Fail status on read_sswitch_reg of LPDDR_CSR_DLL_PHY_CALIBRATION_DATA");
    TerminateFail(1);
  }  
  if (rdata != wdata) {
    PRINT_FN("FAIL: LPDDRRecal(): Write failed, read 0x%08x, expected 0x%08x", rdata, wdata);
    TerminateFail(1);
  }
//  else
//    PRINT_FN("DEBUG: Programmed 0x%08X to LPDDR_CSR_DLL_PHY_CALIBRATION_DATA ", rdata);

  // unpause traffica
  if (!LPDDRUnpauseTraffic()) return 0;

  return 1;
}

//--------------------------------------------------------------------------------------------------------------------
// Routine to do a one-show Master DLL measurement of the DDR clock period
// Returns {status, period}, status=1 for ok
//--------------------------------------------------------------------------------------------------------------------
int Si_LPDDROneShotMeasurement () {
  unsigned wdata;
  unsigned rdata;
  int      retval;
  int      i;
//  unsigned node;

  timer t;
  unsigned int start_time;
  unsigned int tmp;
  
  int measure_start;
  int measure_end;

  //node = this_tile();

  //PRINT_FN("Run one-shot measurement procedure");

  // start by setting enable bit
  wdata = 1<< LPDDR_CSR_DLL_CONTROL_ENABLE_BITPOS;
  if (!write_sswitch_reg(get_tile_id(tile[0]), LPDDR_CONTROLLER_CSR_BASE + LPDDR_CSR_DLL_CONTROL, wdata)) TerminateFail(0xdd4);
  
  // now need to wait a little while for master DLL to be ready
  t :> start_time;
  t when timerafter(start_time + 100) :> tmp;

  // now set one_shot_mode bit as well
  wdata = 1<<LPDDR_CSR_DLL_CONTROL_ENABLE_BITPOS | 
          1<<LPDDR_CSR_DLL_CONTROL_ONE_SHOT_MODE_BITPOS |
          9<<LPDDR_CSR_DLL_CONTROL_ONE_SHOT_COARSE_STEP_LSB;
  if (!write_sswitch_reg(get_tile_id(tile[0]), LPDDR_CONTROLLER_CSR_BASE + LPDDR_CSR_DLL_CONTROL, wdata)) TerminateFail(0xdd4);
  
  // read DLL_MEASUREMENT_STATUS reg to ensure that one_shot_run bit is 0 (not already running)
  retval = read_sswitch_reg(get_tile_id(tile[0]), LPDDR_CONTROLLER_CSR_BASE + LPDDR_CSR_DLL_MEASUREMENT_STATUS, rdata);
  if (!retval) {
    PRINT_FN("FAIL: Fail status on read_sswitch_reg of LPDDR_CSR_DLL_MEASUREMENT_STATUS");
    TerminateFail(1);
  } 
  
  if (rdata>>LPDDR_CSR_DLL_MEASUREMENT_STATUS_ONE_SHOT_RUN_BITPOS & 1) {
    PRINT_FN("WARNING: LPDDROneShotMeasurement: Unexpected DLL_MEASUREMENT_STATUS read 0x%08x, one_shot_run bit is set", rdata);
    return -1;
  } 

  // now set one_shot_trigger bit as well
  wdata = 1<<LPDDR_CSR_DLL_CONTROL_ENABLE_BITPOS | 
          1<<LPDDR_CSR_DLL_CONTROL_ONE_SHOT_MODE_BITPOS |
          1<<LPDDR_CSR_DLL_CONTROL_ONE_SHOT_TRIGGER_BITPOS |
          9<<LPDDR_CSR_DLL_CONTROL_ONE_SHOT_COARSE_STEP_LSB;

  if (!write_sswitch_reg(get_tile_id(tile[0]), LPDDR_CONTROLLER_CSR_BASE + LPDDR_CSR_DLL_CONTROL, wdata)) TerminateFail(0xdd5);
  
  // now poll the one_shot_finish bit in DLL_MEASUREMENT_STATUS 
  for (i=0; i<100; i++) {
    retval = read_sswitch_reg(get_tile_id(tile[0]), LPDDR_CONTROLLER_CSR_BASE + LPDDR_CSR_DLL_MEASUREMENT_STATUS, rdata);
    if (!retval) {
      PRINT_FN("FAIL: Fail status on read_sswitch_reg polling of LPDDR_CSR_DLL_MEASUREMENT_STATUS");
      TerminateFail(1);
    } 
    if ((rdata>>LPDDR_CSR_DLL_MEASUREMENT_STATUS_ONE_SHOT_FINISH_BITPOS & 1) == 1) {
      //PRINT_FN("LPDDROneShotMeasurement: One-shot status is finished, read 0x%08x (iteration %d)", rdata, i);
      break;
    }
    if (i==99) {
      PRINT_FN("WARNING: LPDDROneShotMeasurement: Did not see finished status from LPDDR_CSR_DLL_MEASUREMENT_STATUS polling");
      PRINT_FN("LPDDROneShotMeasurement: Last LPDDR_CSR_DLL_MEASUREMENT_STATUS read 0x%08x", rdata);
      return -1;
    }
  } 

  measure_start = rdata>>LPDDR_CSR_DLL_MEASUREMENT_STATUS_ONE_SHOT_MEASURE_START_LSB & 
                  ~(-1<<LPDDR_CSR_DLL_MEASUREMENT_STATUS_ONE_SHOT_MEASURE_START_WIDTH);

  measure_end = rdata>>LPDDR_CSR_DLL_MEASUREMENT_STATUS_ONE_SHOT_MEASURE_END_LSB & 
                  ~(-1<<LPDDR_CSR_DLL_MEASUREMENT_STATUS_ONE_SHOT_MEASURE_END_WIDTH);
  

  return measure_end-measure_start;
}

//--------------------------------------------------------------------------------------------------------------------
// Routine to configure DLLs
//--------------------------------------------------------------------------------------------------------------------
void Si_SetupLPDDR_DLLs_cr (int clock_freq, int chip_revision) {
  int status;
  unsigned master_period;
  unsigned actual_clock_period;
  unsigned write_dll_setting;

  //--------------------------------------------------------------------------------------------------------------------
  // Perform a 1-shot Master DLL measurement & then update the Slave DLL calibration values
  //--------------------------------------------------------------------------------------------------------------------
  master_period = Si_LPDDROneShotMeasurement();
  if (master_period == -1) {
    PRINT_FN("FAIL: Fail status on LPDDROneShotMeasurement()");
    TerminateFail(1);
  }  

  actual_clock_period = 1000000 / clock_freq;   // period in ps

  if (chip_revision > 0) {
    // rather then divide the period by exactly 4, use an adjustment derived from DLL margin checking to align
    // the wrute data & strobe at a point 83% of the default number, which is the midpoint of the range of DLL
    // values where writing actually works
    write_dll_setting = (master_period * 83) / 400; 

//    PRINT_FN("NOTE: Si_SetupLPDDR_DLLs() - chip_revision>0, using write_dll_setting = %d (master_period = %d)", 
//              write_dll_setting, master_period);
  }
  else {
    // because of the "glitch" bug in A0, we really want to strobe write data quite late in data phase at ~800ps
    write_dll_setting = (800 * master_period) / actual_clock_period; 
  
//    PRINT_FN("WARNING: Si_SetupLPDDR_DLLs() - using an adjusted write_dll_setting = %d (master_period = %d, actual_clock_period = %d)", 
//              write_dll_setting, master_period, actual_clock_period);
  }

  status = LPDDRRecal (master_period, write_dll_setting);
  if (!status) {
    PRINT_FN("FAIL: Fail status on LPDDRRecal()");
    TerminateFail(1);
  }  


}

//--------------------------------------------------------------------------------------------------------------------
// Routine to configure DLLs, implicitly for Chip Revision A0
//--------------------------------------------------------------------------------------------------------------------
void Si_SetupLPDDR_DLLs (int clock_freq) {
    Si_SetupLPDDR_DLLs_cr (clock_freq, 0);
}


//--------------------------------------------------------------------------------------------------------------------
// Routine to do full "silicon ready" setup to get LPDDR controller & device initialised and ready to access
// Uses Sys PLL which it assumes was previously setup at 400MHz (or at 700MHz for the 87.5 or 117MHz DDR clock cases)
//--------------------------------------------------------------------------------------------------------------------
void Si_SetupLPDDR_SysPLL (int clock_freq) {
  unsigned int ddrClkDivAddr;
  
  unsigned int ddr_divider_value;

  unsigned int ddr_period_in_ps;
  unsigned int trefi_value;
  unsigned int tras_value;
  unsigned int txsr_value;
  unsigned int twr_value;
  
  int wait_scale_factor;

  timer t;
  unsigned int start_time;
  unsigned int fin_time;
//  unsigned int tmp, node;

//  node = 0;

  if (clock_freq == 200) {
    ddr_divider_value = 0x00000000;  // div 1, from Sys PLL
    wait_scale_factor = 1;
    ddr_period_in_ps = 5000;
  }
#if 0  
  else if (clock_freq == 100) {
    ddr_divider_value = 0x00000001;  // div 2, from Sys PLL (DDR has another fixed div2)
    wait_scale_factor = 2;           // need a certain number of clock cycles
    ddr_period_in_ps = 10000;
  }
#else
  // These scalars generate 100MHz from -target=XCORE-AI-EXPLORER with the SysPll running at 600MHz
  else if (clock_freq == 100) {
    ddr_divider_value = 0x00000002;  // div 3, from Sys PLL (DDR has another fixed div2)
    wait_scale_factor = 3;           // need a certain number of clock cycles
    ddr_period_in_ps = 10000;
  }
#endif  
  else if (clock_freq == 117) {
    ddr_divider_value = 0x00000002;  // div 3, from Sys PLL (DDR has another fixed div2)
    wait_scale_factor = 2;           // need a certain number of clock cycles
    ddr_period_in_ps = 8570;
  }
  else if (clock_freq == 87) {
    ddr_divider_value = 0x00000003;  // div 4, from Sys PLL (DDR has another fixed div2)
    wait_scale_factor = 3;           // need a certain number of clock cycles
    ddr_period_in_ps = 11400;
  }
  else {
    PRINT_FN("ERROR: Si_SetupLPDDR_SysPLL: Unsupported DDR clock frequency %d (must be 87, 100, 117, or 200MHz)", clock_freq);
    TerminateFail(0xddc0);
  }  

//  PRINT_FN("NOTE: Si_SetupLPDDR_SysPLL: _Assumes_ that SysPLL already running at 400MHz, or 700MHz for 87.5 or 117MHz DDR", clock_freq);

#if 0
unsigned appPllCtrlAddr = XS1_SSWITCH_SS_APP_PLL_CTL_NUM;       

unsigned rdata2, retval2;
retval2 = read_sswitch_reg(0, appPllCtrlAddr, rdata2);
PRINT_FN("READ: Si_SetupLPDDR: appPllCtrlAddr: 0x%08x\n", rdata2);

unsigned sysPllCtrlAddr = XS1_SSWITCH_PLL_CTL_NUM;       
retval2 = read_sswitch_reg(0, sysPllCtrlAddr, rdata2);

sysPllValue = rdata2;


PRINT_FN("READ: Si_SetupLPDDR: sysPllCtrlAddr: 0x%08x\n", rdata2);

PRINT_FN("Si_SetupLPDDR: clodck freq %u\n", clock_freq);
#endif

  //--------------------------------------------------------------------------------------------------------------------
  // setup DDR clock divider
  //--------------------------------------------------------------------------------------------------------------------
  ddrClkDivAddr     = XS1_SSWITCH_DDR_CLK_DIVIDER_NUM;      
#if 1
  write_sswitch_reg(get_tile_id(tile[0]), ddrClkDivAddr, ddr_divider_value); // set the ddr clock divide ratio
#else  
PRINT_FN("Si_SetupLPDDR: writing 0 DDR divider\n");
ddr_divider_value = 0;
write_sswitch_reg(get_tile_id(tile[0]), ddrClkDivAddr, ddr_divider_value); // set the ddr clock divide ratio
#endif
  //--------------------------------------------------------------------------------------------------------------------
  // Need to configure the LPDDR controller's pad mux, and ensure Tile0 has access to the LPDDR controller
  //--------------------------------------------------------------------------------------------------------------------
  unsigned int lpddrControllerConfigAddr = SSWITCH_SS_LPDDR_CONTROLLER_CONFIG_NUM;

  write_sswitch_reg(get_tile_id(tile[0]), lpddrControllerConfigAddr, 0);   // "dummy" write, should not change reset values
//  write_sswitch_reg(get_tile_id(tile[0]), lpddrControllerConfigAddr, 1 | (1 << get_tile_id(tile[0])));   // enabled, accessed from either tile
  write_sswitch_reg(get_tile_id(tile[0]), lpddrControllerConfigAddr, 3);   // enabled, accessed from either tile
  
  //--------------------------------------------------------------------------------------------------------------------
  // Now configure DDR controller (and the LPDDR device)
  // Initially need a delay with the DDR clocks running
  // Then write to necssary CSR locations in the controller, to make it do init sequence & mode reg programming
  // Also need to set up the Q mapping
  //--------------------------------------------------------------------------------------------------------------------
  unsigned regaddr; 
  unsigned rdata; 
  unsigned wdata; 
  int      retval;
  unsigned tmp;

  t :> start_time;
  fin_time = start_time + LPDDR_INIT_WAIT_TIME * wait_scale_factor;
  t when timerafter(fin_time) :> tmp;

#ifdef EXPLORERCOMPILE
  // for Explorer board, program half drive-strength in the EMR
  wdata = 0x0020; // want bits[7:5] = 3'b001 for half-strength
  if (!write_sswitch_reg(get_tile_id(tile[0]), LPDDR_CONTROLLER_CSR_BASE + LPDDR_CSR_EMR_OPCODE, wdata)) TerminateFail(0xdd2);
#endif

  // Manual Transaction Generator is used to trigger the standard Initialization sequence
  regaddr = LPDDR_CONTROLLER_CSR_BASE + LPDDR_CSR_ARBITRATION_MTG_COMMAND;
  
  retval = read_sswitch_reg(get_tile_id(tile[0]), regaddr, rdata);   // read it first and expect 1
  if (!retval)    TerminateFail(0xdd1); 
  if (rdata != 1) TerminateFail(0xdd2);

  wdata = 0x0001; // this is value for Initialization sequence NICETOHAVE - use a constant
  if (!write_sswitch_reg(get_tile_id(tile[0]), regaddr, wdata)) TerminateFail(0xdd3);


  // now configure such that we can actually enqueue
  // let:
  // IDD  7:0  -> RW Q
  // IDD 15:8  -> RO Q
  
  regaddr = LPDDR_CONTROLLER_CSR_BASE + LPDDR_CSR_IID_ENABLE;
  wdata = 0xFFFF;         // enable all 16 IIDs
  if (!write_sswitch_reg(get_tile_id(tile[0]), regaddr, wdata)) TerminateFail(0xdd4);
  
  // read back, to check
  retval = read_sswitch_reg(get_tile_id(tile[0]), regaddr, rdata);
  if (!retval || rdata != wdata) TerminateFail(0xdd5);

  regaddr = LPDDR_CONTROLLER_CSR_BASE + LPDDR_CSR_IID_0_7;
  wdata = 0x00000000;   // all to RW Q 0
  if (!write_sswitch_reg(get_tile_id(tile[0]), regaddr, wdata)) TerminateFail(0xdd6);

  // read back, to check
  retval = read_sswitch_reg(get_tile_id(tile[0]), regaddr, rdata);  // read it first and expect 1
  if (!retval || rdata != wdata) TerminateFail(0xdd7);
  
  //--------------------------------------------------------------------------------------------------------------------
  // Now make any necessary changes to Protocol Engine config
  // This covers 4 parameters with abolsute time requirements which therefore need a different number of clock
  // cycles to achieve dependent on DDR clock period
  //--------------------------------------------------------------------------------------------------------------------
  trefi_value = (DDR_TREFI_IN_NS * 1000) / ddr_period_in_ps;
  tras_value  = (DDR_TRAS_IN_NS  * 1000) / ddr_period_in_ps;
  txsr_value  = (DDR_TXSR_IN_NS  * 1000) / ddr_period_in_ps;
  twr_value   = (DDR_TWR_IN_NS   * 1000) / ddr_period_in_ps;
  
//  PRINT_FN("DEBUG: Si_SetupLPDDR_SysPLL: tWR=%d, tXSR=%d, tRAS=%d, tREFI=%d", twr_value, txsr_value, tras_value, trefi_value);
  
  regaddr = LPDDR_CONTROLLER_CSR_BASE + LPDDR_CSR_PROTOCOL_ENGINE_CONF_0;
  wdata = twr_value<<21 | txsr_value<<15 | tras_value<<11 | trefi_value; 
  if (!write_sswitch_reg(get_tile_id(tile[0]), regaddr, wdata)) TerminateFail(0xdd7);

  // read back, to check
  retval = read_sswitch_reg(get_tile_id(tile[0]), regaddr, rdata);
  if (!retval || rdata != wdata) TerminateFail(0xdd8);

  //--------------------------------------------------------------------------------------------------------------------
  // before attempting access, need to enable tile EXTEM memory-space via PS CTRL0
  //--------------------------------------------------------------------------------------------------------------------
  #define XS1_XCORE_CTRL0_EXTMEM_ENABLE_MASK 1

  // Enable external memory access
  unsigned int ctrl0 = getps(XS1_PS_XCORE_CTRL0);
  ctrl0 |= XS1_XCORE_CTRL0_EXTMEM_ENABLE_MASK;
  setps(XS1_PS_XCORE_CTRL0, ctrl0);

  // configure DLLs
  Si_SetupLPDDR_DLLs(clock_freq); 
}

void setup_ddr() {
  if (Si_SetupLPDDRPadCtrl()) {
    printf("Already done\n");
    return; // DDR already set up.
  }
  printf("Setting up\n");
  Si_SetupLPDDR_SysPLL(87);
}
