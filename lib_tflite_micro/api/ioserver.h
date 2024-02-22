#ifndef _io_server_h_
#define _io_server_h_

#ifdef __XC__

void ioserver(chanend c_model[], unsigned n_model);

#else

#include <xcore/chanend.h>
#include <xcore/channel.h>

#define CMD_LENGTH_BYTES (3) // CMD, Model, Tensor

#define IOSERVER_INVOKE 1
#define IOSERVER_TENSOR_SEND_OUTPUT 2
#define IOSERVER_TENSOR_RECV_INPUT 3
#define IOSERVER_ACK 5
#define IOSERVER_NACK 6
#define IOSERVER_RESET 7
#define IOSERVER_EXIT 8

#define MAX_PACKET_SIZE (512)
#define MAX_PACKET_SIZE_WORDS (MAX_PACKET_SIZE / 4)

#ifdef __cplusplus
extern "C" {
#endif
unsigned int ioserver_command_receive(chanend_t c_server, unsigned *tensor_num);
void ioserver_command_acknowledge(chanend_t c_server, unsigned int ack);
void ioserver_tensor_send_output(chanend_t c_server, unsigned int *data,
                                 unsigned int n);

void ioserver_tensor_recv_input(chanend_t c_server, unsigned int *data,
                                unsigned int n);

void ioserver(chanend_t c_model[], unsigned n_model);
#ifdef __cplusplus
}
#endif

#endif

#endif
