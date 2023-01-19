#include <stdint.h>
#include <stdio.h>
#include <platform.h>
#include "flash.h"

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



#define TEST_INPUT_SIZE  (256*192*3)
#define TEST_OUTPUT_SIZE0  (80)
#define TEST_OUTPUT_SIZE1  (20)
#define TEST_OUTPUT_SIZE2  (20)
#define TEST_OUTPUT_SIZE3  (1)


int8_t s1[TEST_INPUT_SIZE] = {
#include "s1.csv"
};
int8_t s2[TEST_INPUT_SIZE] = {
#include "s2.csv"
};
int8_t s3[TEST_INPUT_SIZE] = {
#include "s1.csv"
};

int32_t output0[TEST_OUTPUT_SIZE0];
int32_t output1[TEST_OUTPUT_SIZE1];
int8_t output2[TEST_OUTPUT_SIZE2];
int32_t output3[TEST_OUTPUT_SIZE3];

#pragma unsafe arrays
void data_creation(chanend x) {
    printf("\nWriting input\n");

    for(int i = 0 ; i < TEST_INPUT_SIZE; i++) {
        outuchar(x, s1[i]);
    }
    printf("\nDone writing input1\n\n");

    for(int i = 0 ; i < TEST_OUTPUT_SIZE0; i++) {
        output0[i] = inuint(x);
        printf("%d,",(int)output0[i]);
    }
    printf("\n\n");


    for(int i = 0 ; i < TEST_OUTPUT_SIZE1; i++) {
        output1[i] = inuint(x);
        printf("%d,",(int)output1[i]);
    }
    printf("\n\n");


    for(int i = 0 ; i < TEST_OUTPUT_SIZE2; i++) {
        output2[i] = inuchar(x);
        printf("%d,",(int)output2[i]);
    }
    printf("\n\n");


    for(int i = 0 ; i < TEST_OUTPUT_SIZE3; i++) {
        output3[i] = inuint(x);
        printf("%d,",(int)output3[i]);
    }
    printf("\n\n");


    outct(x, 1);
    chkct(x, 1);


    // for(int i = 0 ; i < TEST_INPUT_SIZE; i++) {
    //     outuchar(x, s2[i]);
    // }
    // printf("\nDone writing input2\n\n");

    // for(int i = 0 ; i < TEST_OUTPUT_SIZE0; i++) {
    //     output0[i] = inuint(x);
    //     printf("%d,",(int)output0[i]);
    // }
    // printf("\n\n");


    // for(int i = 0 ; i < TEST_OUTPUT_SIZE1; i++) {
    //     output1[i] = inuint(x);
    //     printf("%d,",(int)output1[i]);
    // }
    // printf("\n\n");


    // for(int i = 0 ; i < TEST_OUTPUT_SIZE2; i++) {
    //     output2[i] = inuchar(x);
    //     printf("%d,",(int)output2[i]);
    // }
    // printf("\n\n");


    // for(int i = 0 ; i < TEST_OUTPUT_SIZE3; i++) {
    //     output3[i] = inuint(x);
    //     printf("%d,",(int)output3[i]);
    // }
    // printf("\n\n");


    // outct(x, 1);
    // chkct(x, 1);


    // for(int i = 0 ; i < TEST_INPUT_SIZE; i++) {
    //     outuchar(x, s3[i]);
    // }
    // printf("\nDone writing input3\n\n");

    // for(int i = 0 ; i < TEST_OUTPUT_SIZE0; i++) {
    //     output0[i] = inuint(x);
    //     printf("%d,",(int)output0[i]);
    // }
    // printf("\n\n");


    // for(int i = 0 ; i < TEST_OUTPUT_SIZE1; i++) {
    //     output1[i] = inuint(x);
    //     printf("%d,",(int)output1[i]);
    // }
    // printf("\n\n");


    // for(int i = 0 ; i < TEST_OUTPUT_SIZE2; i++) {
    //     output2[i] = inuchar(x);
    //     printf("%d,",(int)output2[i]);
    // }
    // printf("\n\n");


    // for(int i = 0 ; i < TEST_OUTPUT_SIZE3; i++) {
    //     output3[i] = inuint(x);
    //     printf("%d,",(int)output3[i]);
    // }
    // printf("\n\n");

    // outct(x, 1);
    // chkct(x, 1);
}

extern void inferencer(chanend x, chanend f);

int main(void) {
    chan x;
    chan c_flash[1];

    par {
        on tile[0]: {
            data_creation(x);
        }

        on tile[0]: {
            flash_t headers[2];
            flash_server(c_flash, headers, 1, qspi, flash_spec, 1);            
        }

        on tile[1]: {
            unsafe {
            printf("\nStart inferencing1");
            inferencer(x, c_flash[0]);
            
            // printf("\nStart inferencing2");
            // inferencer(x, c_flash[0]);
                        
            // printf("\nStart inferencing3");
            // inferencer(x, c_flash[0]);

            printf("\nDone inferencing");

            c_flash[0] <: FLASH_SERVER_QUIT;
            }
        }
    }
    return 0;
}
