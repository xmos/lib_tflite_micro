#include <xcore_load_from_flash_ll.h>

// TODO, inline this and replace with lib_xcore

void load_from_flash_ll(chanend c_flash, int8_t data[], uint32_t address, uint32_t bytes) {
    c_flash <: address;
    c_flash <: bytes;
    slave {
        for(int i = 0; i < bytes; i++) {
            c_flash :> data[i];
        }
    }
}
