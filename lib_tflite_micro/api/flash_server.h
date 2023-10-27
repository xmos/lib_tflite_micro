#ifndef _flash_server_h_
#define _flash_server_h_

#include <quadflash.h>

/** Struct holding the "file system" meta information for each client
 * The flash is partitioned and each client has a section in the flash
 * that stores data relevant to that particular client. For example, models
 * parameters, code, etc.
 *
 * This struct caches all information necessary for a client for fast access.
 * The main program must allocate this structure, one per client, prior to
 * calling the flash server.
 *
 * If there is more than one flash device connected to the device, there can be
 * multiple flash servers.
 */
typedef struct flash {
  int model_start;            ///< Start address for model.
  int parameters_start;       ///< Start address of parameters.
  int operators_start;        ///< Start address for operator-binaries.
  int execute_in_place_start; ///< Start address for operator-binaries.
} flash_t;

/** Type representing the commands that the flash server accepts */
typedef enum flash_command {
  FLASH_READ_PARAMETERS =
      0, ///< Read a set of parameters.   // TODO: share with lib_tflite_micro
  FLASH_READ_MODEL = 1, ///< Read a whole model.
  FLASH_READ_OPERATORS =
      2, ///< Read the binary for an operator - future extension
  FLASH_READ_XIP =
      3, ///< Read code to execute-in-place throught L2 cache - future extension
  FLASH_SERVER_QUIT = 4,
  FLASH_SERVER_INIT = 5, // Initialize flash server with fast flash pattern speed match setup
  //FLASH_READ_PARAMETERS_COMPRESSED_FLOAT = 6, // Read a set of compressed parameters
} flash_command_t;

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
 * \param c_flash_clients Array of channels; one per client.
 *                        Each client is served in turn
 * \param headers         Space to store a header for each client
 *                        The header for the client describes the local
 * "filesystem" for that client \param n_flash_clients Number of clients. The
 * arrays in the first and second parameters should have this many entries
 * \param qspi            Structure holding the quad-flash ports. This contains
 * three Ports and a clock-block, the CS_N port, the CLK port, the DATA port and
 * a clock block to be used for the flash. \param flash_spec      Array holding
 * specificiations of flash devices, as per the libquadflash documentation
 * \param n_flash_spec    Number of elements in the spec array.
 */
#ifdef __XC__
void flash_server(chanend c_flash_clients[], flash_t headers[],
                  int n_flash_clients, fl_QSPIPorts &qspi,
                  fl_QuadDeviceSpec flash_spec[], int n_flash_spec);
#else
void flash_server(chanend_t *c_flash_clients, flash_t *headers,
                  int n_flash_clients, fl_QSPIPorts *qspi,
                  fl_QuadDeviceSpec *flash_spec, int n_flash_spec);
#endif

#endif
