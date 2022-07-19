#include <stdint.h>
#include "i2c.h"

#define GAIN_MIN_DB       0
#define GAIN_MAX_DB      84
#define GAIN_DEFAULT_DB  50

extern int gc0310_stream_start(client interface i2c_master_if i2c);
extern int gc0310_stream_stop(client interface i2c_master_if i2c);
extern int gc0310_set_gain_dB(client interface i2c_master_if i2c,
                              uint32_t dBGain);

