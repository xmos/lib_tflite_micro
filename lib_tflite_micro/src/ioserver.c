#include <xs1.h>
#include <stdio.h>
//#include <print.h>
#include <xcore/parallel.h>
#include "ioserver.h"

#include "xud.h"
#include "xud_device.h"

DECLARE_JOB(ioserver_usb_ep,        (chanend_t, chanend_t,
                                     chanend_t, chanend_t*, unsigned));
DECLARE_JOB(ioserver_usb_ep0,       (chanend_t, chanend_t,
                                     chanend_t));
DECLARE_JOB(XUD_Main2,              (chanend_t *, int,
                                     chanend_t *, int,
                                     unsigned,
                                     unsigned int*, unsigned int*,
                                     XUD_BusSpeed_t,
                                     XUD_PwrConfig));

void XUD_Main2(chanend_t *a, int b,
                                     chanend_t *c, int d,
                                     unsigned e,
                                     unsigned int*f, unsigned int*g,
                                     XUD_BusSpeed_t h,
               XUD_PwrConfig i) {
    XUD_Main(a,b,c,d,e,f,g,h,i);
}


#define EP_COUNT_OUT 2
#define EP_COUNT_IN 2
#define BCD_DEVICE   0x1000
#define VENDOR_ID    0x20B1
#define PRODUCT_ID   0xa15e


/* Device Descriptor */
static unsigned char devDesc[] =
{
    0x12,                  /* 0  bLength */
    USB_DESCTYPE_DEVICE,   /* 1  bdescriptorType */
    0x00,                  /* 2  bcdUSB */
    0x02,                  /* 3  bcdUSB */
    0xff,                  /* 4  bDeviceClass */
    0xff,                  /* 5  bDeviceSubClass */
    0xff,                  /* 6  bDeviceProtocol */
    0x40,                  /* 7  bMaxPacketSize */
    (VENDOR_ID & 0xFF),    /* 8  idVendor */
    (VENDOR_ID >> 8),      /* 9  idVendor */
    (PRODUCT_ID & 0xFF),   /* 10 idProduct */
    (PRODUCT_ID >> 8),     /* 11 idProduct */
    (BCD_DEVICE & 0xFF),   /* 12 bcdDevice */
    (BCD_DEVICE >> 8),     /* 13 bcdDevice */
    0x01,                  /* 14 iManufacturer */
    0x02,                  /* 15 iProduct */
    0x00,                  /* 16 iSerialNumber */
    0x01                   /* 17 bNumConfigurations */
};

static unsigned char cfgDesc[] =
{
    /* Configuration descriptor: */ 
    0x09,                               /* 0  bLength */ 
    0x02,                               /* 1  bDescriptorType */ 
    0x20, 0x00,                         /* 2  wTotalLength */ 
    0x01,                               /* 4  bNumInterface: Number of interfaces*/ 
    0x01,                               /* 5  bConfigurationValue */ 
    0x00,                               /* 6  iConfiguration */ 
    0x80,                               /* 7  bmAttributes */ 
    0xFA,                               /* 8  bMaxPower */  

    /*  Interface Descriptor (Note: Must be first with lowest interface number)r */
    0x09,                               /* 0  bLength: 9 */
    0x04,                               /* 1  bDescriptorType: INTERFACE */
    0x00,                               /* 2  bInterfaceNumber */
    0x00,                               /* 3  bAlternateSetting: Must be 0 */
    0x02,                               /* 4  bNumEndpoints (0 or 1 if optional interupt endpoint is present */
    0xff,                               /* 5  bInterfaceClass: AUDIO */
    0xff,                               /* 6  bInterfaceSubClass: AUDIOCONTROL*/
    0xff,                               /* 7  bInterfaceProtocol: IP_VERSION_02_00 */
    0x03,                               /* 8  iInterface */ 

/* Standard Endpoint Descriptor (INPUT): */
    0x07, 			                    /* 0  bLength: 7 */
    0x05, 			                    /* 1  bDescriptorType: ENDPOINT */
    0x01,                               /* 2  bEndpointAddress (D7: 0:out, 1:in) */
    0x02,
    MAX_PACKET_SIZE & 0xff, (MAX_PACKET_SIZE >> 8) & 0xff, /* 4  wMaxPacketSize */
    0x01,                               /* 6  bInterval */

/* Standard Endpoint Descriptor (OUTPUT): */
    0x07, 			                    /* 0  bLength: 7 */
    0x05, 			                    /* 1  bDescriptorType: ENDPOINT */
    0x81,                               /* 2  bEndpointAddress (D7: 0:out, 1:in) */
    0x02,
    MAX_PACKET_SIZE & 0xff, (MAX_PACKET_SIZE >> 8) & 0xff, /* 4  wMaxPacketSize */
    0x01,                               /* 6  bInterval */
};


/* String table */
static char *stringDescriptors[]=
{      
    "\x09\x04",             // Language ID string (US English)
    "XMOS",                 // iManufacturer
    "xAISRV",               // iProduct
    "Config",               // iConfiguration
};


void ioserver_usb_ep0(chanend_t c_ep0_out, chanend_t c_ep0_in, chanend_t c_data) {
    USB_SetupPacket_t sp;
    XUD_BusSpeed_t usbBusSpeed;
    XUD_ep ep0_out = XUD_InitEp(c_ep0_out);
    XUD_ep ep0_in  = XUD_InitEp(c_ep0_in);

    while(1) {
        /* Returns XUD_RES_OKAY on success */
        XUD_Result_t result = USB_GetSetupPacket(ep0_out, ep0_in, &sp);

        if(result == XUD_RES_OKAY)
        {
            /* Set result to ERR, we expect it to get set to OKAY if a request is handled */
            result = XUD_RES_ERR;
            // TODO 
            //result = AisrvClassRequests(ep0_out, ep0_in, sp);
        }

        /* If we haven't handled the request about then do standard enumeration requests */
        if(result == XUD_RES_ERR )
        {
            /* Returns  XUD_RES_OKAY if handled okay,
             *          XUD_RES_ERR if request was not handled (STALLed),
             *          XUD_RES_RST for USB Reset */
            result = USB_StandardRequests(ep0_out, ep0_in,
                                          devDesc, sizeof(devDesc),
                                          cfgDesc, sizeof(cfgDesc),
                                          0, 0, 0, 0,
                                          stringDescriptors, sizeof(stringDescriptors)/sizeof(stringDescriptors[0]),
                                          &sp, usbBusSpeed);
        }
        
        unsigned bmRequestType = (sp.bmRequestType.Direction<<7) | (sp.bmRequestType.Type<<5) | (sp.bmRequestType.Recipient);

        if((bmRequestType == USB_BMREQ_H2D_STANDARD_EP)
            && (sp.bRequest == USB_CLEAR_FEATURE)
            && (sp.wLength == 0)
            /* The only Endpoint feature selector is HALT (bit 0) see figure 9-6 */
            && (sp.wValue == USB_ENDPOINT_HALT)
            && ((sp.wIndex & 0x7F) == 1)) { // EP 1 IN or OUT
            chan_out_word(c_data, sp.wIndex);
        }

        /* USB bus reset detected, reset EP and get new bus speed */
        if(result == XUD_RES_RST) {
            usbBusSpeed = XUD_ResetEndpoint(ep0_out, &ep0_in);
        }
    }
}

void ioserver(chanend_t c_models[], unsigned n_models) {
    chanend_t c_ep_out[EP_COUNT_OUT], c_ep_in[EP_COUNT_IN];
    chanend_t c_ep_out_ends[EP_COUNT_OUT], c_ep_in_ends[EP_COUNT_IN];
    XUD_EpType epTypeTableOut[EP_COUNT_OUT] = {XUD_EPTYPE_CTL | XUD_STATUS_ENABLE, XUD_EPTYPE_BUL};
    XUD_EpType epTypeTableIn[EP_COUNT_IN] =   {XUD_EPTYPE_CTL | XUD_STATUS_ENABLE, XUD_EPTYPE_BUL};
    for(unsigned i = 0; i < EP_COUNT_OUT; i++) {
        channel_t c = chan_alloc();
        c_ep_out[i] = c.end_a;
        c_ep_out_ends[i] = c.end_b;
    }
    for(unsigned i = 0; i < EP_COUNT_IN; i++) {
        channel_t c = chan_alloc();
        c_ep_in[i] = c.end_a;
        c_ep_in_ends[i] = c.end_b;
    }
    channel_t c_usb_ep0_dat;
    PAR_JOBS(
        PJOB(ioserver_usb_ep, (c_ep_out[1], c_ep_in[1],
                               c_usb_ep0_dat.end_a,
                               c_models, n_models)),
        PJOB(ioserver_usb_ep0, (c_ep_out[0], c_ep_in[0],
                             c_usb_ep0_dat.end_b)),
        PJOB(XUD_Main2,     (c_ep_out_ends, EP_COUNT_OUT,
                             c_ep_in_ends, EP_COUNT_IN,
                             0, epTypeTableOut, epTypeTableIn,
                             XUD_SPEED_HS, XUD_PWR_BUS)));
}

unsigned int ioserver_command_receive(chanend_t c_server, unsigned *tensor_num) {
    unsigned int cmd = chan_in_word(c_server);
    *tensor_num = chan_in_word(c_server);
    return cmd;
}

void ioserver_command_acknowledge(chanend_t c_server, unsigned int ack) {
    chan_out_word(c_server, ack);
}

void ioserver_tensor_send_output(chanend_t c_server,
                                 unsigned int *data,
                                 unsigned int n) {
    chanend_out_word(c_server, n);
    for(int i = 0; i < n; i++) {
        chanend_out_word(c_server, data[i]);
    }
    chanend_out_control_token(c_server, XS1_CT_END);
}

void ioserver_tensor_recv_input(chanend_t c_server,
                                unsigned int *data,
                                unsigned int n) {
    for(int i = 0; i < n; i++) {
        data[i] = chanend_in_word(c_server);
    }
    chanend_check_control_token(c_server, XS1_CT_END);
}

void ioserver_usb_ep(chanend_t c_ep_out, chanend_t c_ep_in,
                              chanend_t c_ep0,
                              chanend_t c_model[], unsigned n_models)
{
    int32_t data[MAX_PACKET_SIZE_WORDS];

    XUD_ep ep_out = XUD_InitEp(c_ep_out);
    XUD_ep ep_in  = XUD_InitEp(c_ep_in);

    uint32_t cmd = 0;
    uint32_t model_num = 0;
    uint32_t tensor_num = 0;
    
    int stalled_in = 0;
    int stalled_out = 0;

    while(1)
    {
        unsigned length = 0;
        unsigned pktLength;
            
        while(stalled_in || stalled_out)
        {
            unsigned x;
            
            /* Wait for clear on both Endpoints */
            x = chan_in_word(c_ep0);
            
            if(x == 0x01)
                stalled_out = 0;
            else if(x == 0x81)
                stalled_in = 0;
        }

        /* Get command */
        XUD_GetBuffer(ep_out, (uint8_t *)data, &length);

        if(length != CMD_LENGTH_BYTES) {
            printstr("Bad cmd length: "); printintln(length);
            continue;
        }
                
        cmd        = ((uint8_t*)data)[0];
        model_num  = ((uint8_t*)data)[1];
        tensor_num = ((uint8_t*)data)[2];

        unsigned n;
        if (model_num >= n_models) {
            printint(model_num);
            printstrln(" model not available, using 0\n");
            model_num = 0;
        }
        chan_out_word(c_model[model_num], cmd);
        chan_out_word(c_model[model_num], tensor_num);
        switch(cmd) {
        case IOSERVER_INVOKE:
            // TODO: remove this and friend in PY side
            XUD_GetBuffer(ep_out, (uint8_t*)data, &pktLength);
            
            (void) chan_in_word(c_model[model_num]); // discard ack
            break;
        
        case IOSERVER_TENSOR_SEND_OUTPUT:
            n = chanend_in_word(c_model[model_num]);
            size_t i = 0;
            for(unsigned k = 0; k < n; k++) {
                int tmp = chanend_in_word(c_model[model_num]);
                data[i++] = tmp; 
                if(i == MAX_PACKET_SIZE_WORDS)
                {
                    XUD_SetBuffer(ep_in, (uint8_t *)data, MAX_PACKET_SIZE);
                    i = 0;
                }
            }
            chanend_check_control_token(c_model[model_num], XS1_CT_END);
            XUD_SetBuffer(ep_in, (uint8_t *)data, i*4);
            break;
        
        case IOSERVER_TENSOR_RECV_INPUT:
            while(1)
            {
                XUD_GetBuffer(ep_out, (uint8_t *)data, &pktLength);
                for(unsigned i = 0; i < (pktLength+3)/4; i++) {
                    chanend_out_word(c_model[model_num], data[i]);
                }
                if(pktLength != MAX_PACKET_SIZE) {
                    break;
                }
            }
            chanend_out_control_token(c_model[model_num], XS1_CT_END);
            break;
            
        default:
            printstr("Unknown command (ioserver): "); printintln(cmd);
            break;
        }
    } // while(1)
}
