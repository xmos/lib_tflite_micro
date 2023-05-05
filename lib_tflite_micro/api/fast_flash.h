#ifndef _FAST_FLASH_H_
#define _FAST_FLASH_H_

#include <quadflash.h>

/** Fast flash library.
 * Before calling any of the functions in here, lib_quad_flash must be initialised as normal by using
 * fl_connectToDevice(qspi, flash_spec, n_flash_spec).
 * After that, a call to fast_flash_init shall be made.
 * After that, a sequence of calls to fast_flash_read can be made.
 *
 * The data partition must start with the following 32 bytes: **NOTE: REMOVE THE +4 in fast_flash_init**
 *
 *   0xff, 0x00, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f,
 *   0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00,
 *   0x31, 0xf7, 0xce, 0x08, 0x31, 0xf7, 0xce, 0x08,
 *   0x9c, 0x63, 0x9c, 0x63, 0x9c, 0x63, 0x9c, 0x63
 * 
 * This pattern is designed to create maximum difficulties electrically and is used
 * to calibrate the electrical settings. Note that this pattern must be nibble reversed
 * before being written to flash; just like all other data.
 * The rest of the data partition can be used as normal
 */

/** Function that initialises the fast_flash library
 *
 * \param      qspi        ports that connect to flash
 *
 * \returns    a negative value of -1..-5 if the window is too small (size 0..4)
 *             zero if successful
 */
int fast_flash_init(fl_QSPIPorts &qspi);

/** Function that reads a sequential set of bytes from memory.
 * This function assumes that nibbles have been reversed ((x << 4) & 0xf0 | (x >> 4) & 0x0f)
 * before the data was written to flash.
 * Note that reading 32 bytes from offset 0 shall yield the special pattern above.
 *
 * \param      qspi        ports that connect to flash
 * \param      addr        address in flash data segment
 * \param      word_count  Number of words to read
 * \param      read_data   array to store data in to.
 * \param      c_out_data  optional channel end over which data is out() instead.
 */
void fast_flash_read(fl_QSPIPorts &qspi, unsigned addr, unsigned word_count, unsigned read_data[], chanend ?c_data_out);

#endif
