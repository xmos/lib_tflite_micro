#ifndef _tile_ram_server_h_
#define _tile_ram_server_h_

#include "flash_server.h"

/**
 * Function that runs a flash-server. A flash server is a thread that serves one
 * or more clients. There is one flash server per flash-device, and the server
 * can serve clients on one or more tiles.
 *
 * The flash server takes the following commands:
 *   - Read a whole model from the flash.
 *   - Read some parameters from the flash
 *   - (future extension) Read code for an operator from flash
 *
 * This function does, at present, never return. It could be made to return if
 * all clients close their connection
 *
 * \param c_tile_ram_clients Array of channels; one per client.
 *                          Each client is served in turn
 *                          NOTE MUST BE 1 FOR NOW
 * \param headers           Space to store a header for each client
 *                          The header for the client describes the local
 *                          "filesystem" for that client
 * \param n_tile_ram_clients Number of clients. The
 *                          arrays in the first and second parameters should have this many entries
 * \param data              Tile ram data
 * \param n_tile_ram_flash  Number of bytes in array
 */
#ifdef __XC__
void tile_ram_server(chanend c_tile_ram_clients[], flash_t headers[],
                     int n_tile_ram_clients, const int8_t data[]);
#else
void tile_ram_server(chanend_t *c_tile_ram_clients, flash_t *headers,
                     int n_tile_ram_clients, const int8_t *data);
#endif

#endif
