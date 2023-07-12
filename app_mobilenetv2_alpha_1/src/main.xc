#include "flash_server.h"
#include "stdio.h"
#include <platform.h>
#include <quadflash.h>
#include <stdint.h>

#define NUMBER_OF_MODELS 1

#if defined __GNUC__
#define ALIGN(X) __attribute__((aligned(X)))
#elif defined _MSC_VER
#define ALIGN(X) __declspec(align(X))
#elif defined __TASKING__
#define ALIGN(X) __align(X)
#endif

#define FL_QUADDEVICE_MACRONIX_MX25R6435FM2IH0                                 \
  {                                                                            \
    16,                         /* MX25R6435FM2IH0 */                          \
        256,                    /* page size */                                \
        32768,                  /* num pages */                                \
        3,                      /* address size */                             \
        3,                      /* log2 clock divider */                       \
        0x9F,                   /* QSPI_RDID */                                \
        0,                      /* id dummy bytes */                           \
        3,                      /* id size in bytes */                         \
        0xC22817,               /* device id */                                \
        0x20,                   /* QSPI_SE */                                  \
        4096,                   /* Sector erase is always 4KB */               \
        0x06,                   /* QSPI_WREN */                                \
        0x04,                   /* QSPI_WRDI */                                \
        PROT_TYPE_NONE,         /* no protection */                            \
        {{0, 0}, {0x00, 0x00}}, /* QSPI_SP, QSPI_SU */                         \
        0x02,                   /* QSPI_PP */                                  \
        0xEB,                   /* QSPI_READ_FAST */                           \
        1,                      /* 1 read dummy byte */                        \
        SECTOR_LAYOUT_REGULAR,  /* mad sectors */                              \
        {4096, {0, {0}}},       /* regular sector sizes */                     \
        0x05,                   /* QSPI_RDSR */                                \
        0x01,                   /* QSPI_WRSR */                                \
        0x01,                   /* QSPI_WIP_BIT_MASK */                        \
  }

fl_QuadDeviceSpec flash_spec[] = {
    FL_QUADDEVICE_MACRONIX_MX25R6435FM2IH0,
};

on tile[0] : fl_QSPIPorts qspi = {PORT_SQI_CS, PORT_SQI_SCLK, PORT_SQI_SIO,
                                  XS1_CLKBLK_2};

extern void model1_init(chanend f);
extern void inference1();

int main(void) {
  chan c_flash[1];

  par {
    on tile[0] : {
      flash_t headers[NUMBER_OF_MODELS];
      flash_server(c_flash, headers, NUMBER_OF_MODELS, qspi, flash_spec, 1);
    }

    on tile[1] : {
      unsafe {
        c_flash[0] <: FLASH_SERVER_INIT;
        model1_init(c_flash[0]);

        printf("\nStart inferencing1");
        inference1();
        printf("\nDone inferencing1");

        c_flash[0] <: FLASH_SERVER_QUIT;
      }
    }
  }
  return 0;
}
