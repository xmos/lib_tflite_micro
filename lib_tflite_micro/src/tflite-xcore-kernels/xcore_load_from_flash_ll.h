#ifndef xcore_load_from_flash_ll_h
#define xcore_load_from_flash_ll_h

#include <stdint.h>

#ifdef __XC__

void load_from_flash_ll(chanend c_flash, int8_t data[], uint32_t address, uint32_t bytes);

#else

void load_from_flash_ll(unsigned c_flash, int8_t data[], uint32_t address, uint32_t bytes);

#endif

#endif
