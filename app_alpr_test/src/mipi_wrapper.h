#ifndef _AISRV_MIPI_H_
#define _AISRV_MIPI_H_

#define GC2145

#define RAW_IMAGE_HEIGHT (320)
#define RAW_IMAGE_WIDTH  (320)
#define RAW_IMAGE_DEPTH  (3)

// These define the Sensor height and width
// And the area in the centre of the sensor that we use, measured in sensor pixels
#ifdef GC2145
#define SENSOR_IMAGE_HEIGHT (1200)
#define SENSOR_IMAGE_WIDTH  (1600)
#define WIDTH_ON_SENSOR     (800)
#define HEIGHT_ON_SENSOR    (800)
#else
#define SENSOR_IMAGE_HEIGHT (480)
#define SENSOR_IMAGE_WIDTH  (640)
#define WIDTH_ON_SENSOR     (320)
#define HEIGHT_ON_SENSOR    (320)
#endif

#define SENSOR_IMAGE_DEPTH  (2)          // YUV: YU  YV  YU  YV  ...

#include "i2c.h"

void mipi_main(client interface i2c_master_if i2c, chanend c_acquire);

#endif
