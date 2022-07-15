// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__XS3A__)

#define FUNCTION_NAME subsample_x_asm

    .cc_top subsample_x_asm.function,subsample_x_asm
    .text
    .align 4
shift_8:
    .word    0x00060006
    .word    0x00060006
    .word    0x00060006
    .word    0x00060006
    .word    0x00060006
    .word    0x00060006
    .word    0x00060006
    .word    0x00060006

    .issue_mode  dual


#define NSTACKWORDS  6

    // r0: outp
    // r1: line
    // r2: coefficients
    // r3: strides
    // r4: ox
    // r5: indexc
        
    .globl FUNCTION_NAME
    .align 16
    .type FUNCTION_NAME,@function

FUNCTION_NAME:
    {dualentsp NSTACKWORDS; nop}
    std r4, r5, sp[0]
    std r6, r7, sp[1]
    ldw r4, sp[NSTACKWORDS+1]
    ldw r5, sp[NSTACKWORDS+2]

    ldc r11, 0x200
    vsetc r11
    { add r2, r2, r5            ; ldc r5, 32  }
    { add r0, r0, r4            ; ldc r7, 2 }
    
    
rgbloop:    
    { vclrdr                    ; add r6, r4, 0 }
#define ONE \
    { ldw r11, r3[r6]           ; add r6, r6, 1 } ;\
    { vldc r2[0]                ; add r11, r1, r11 } ;\
    { vlmaccr r11[0]            ; add r2, r2, r5 }

    ONE
    ONE
    ONE
    ONE
    
    ONE
    ONE
    ONE
    ONE
    
    ONE
    ONE
    ONE
    ONE
    
    ONE
    ONE
    ONE
    ONE

    { ldap r11, shift_8           ; ldc r6, 10 }
    { vlsat r11[0]                ; shl r6, r6, 4 }
    { vdepth8                     ; mkmsk r11, 16 }
    vstrpv r0[0], r11

    add r0, r0, r6
    { bt r7, rgbloop              ; sub r7, r7, 1 }

    
    ldd r6, r7, sp[1]
    ldd r4, r5, sp[0]
    retsp NSTACKWORDS

    .set FUNCTION_NAME.nstackwords,NSTACKWORDS
    .globl FUNCTION_NAME.nstackwords
    .set FUNCTION_NAME.maxcores,1
    .globl FUNCTION_NAME.maxcores
    .set FUNCTION_NAME.maxtimers,0
    .globl FUNCTION_NAME.maxtimers
    .set FUNCTION_NAME.maxchanends,0
    .globl FUNCTION_NAME.maxchanends
.Ltmp0:
    .size FUNCTION_NAME, .Ltmp0-FUNCTION_NAME

#undef FUNCTION_NAME
#undef NSTACKWORDS
#define FUNCTION_NAME subsample_y_asm




//int8_t *OT_int8::output_transform_fn(int8_t *Y, VPURingBuffer *A, int32_t output_channel_group)

#define NSTACKWORDS  16

    // r0: outp
    // r1: line
    // r2: coefficients
    // r3: strides
    // r4: ox
    // r5: indexc
    
    // void transform_vertical_line_vpu(int8_t outp[3][160], int8_t inp[3][160][5], int8_t coefficients[], uint32_t strides[], int oy) {
    
    .issue_mode  dual
    .text
    .globl FUNCTION_NAME
    .align 16
    .type FUNCTION_NAME,@function
FUNCTION_NAME:
    {dualentsp NSTACKWORDS; nop}

    std r4, r5, sp[0]
    std r6, r7, sp[1]
    std r8, r9, sp[2]
    stw r10, sp[6]
    ldw r4, sp[NSTACKWORDS+1]
    ldw r5, sp[NSTACKWORDS+2]
    // r0: outputs
    // r1..r5: inputs
    
    ldc r11, 0x200
    vsetc r11

#undef ONE
#define ONE(R) \
    { vldc R[0]                 ; add R, R, r8 } ;\
    { vlmacc r6[0]              ; add r6, r6, r8 } ;\

    ldw r7, sp[NSTACKWORDS+4]
ox_loop:    
    ldw r6, sp[NSTACKWORDS+3]
    { vclrdr                    ; ldc r8, 16 }
    ONE(r1)
    ONE(r2)
    ONE(r3)
    ONE(r4)
    ONE(r5)
    ldap r11, shift_8
    vlsat r11[0]
    { vdepth8                   ; ldaw r11, sp[7] }
    { vstr r11[0]               ; ldc r9, 48 }
    ldw r8, sp[NSTACKWORDS+5]
    { add r9, r9, r8            ; ldc r8, 15 }
copy_loop:
    { ld8u r6, r11[r8]            ; sub r9, r9, 3 }
    st8 r6, r0[r9]
    { bt r8, copy_loop            ;sub r8, r8, 1 }
    { ldc r9, 48                  ; sub r7, r7, 1  }
    { bt r7, ox_loop              ; add r0, r0, r9 }

    
    
    ldw r10, sp[6]
    ldd r8, r9, sp[2]
    ldd r6, r7, sp[1]
    ldd r4, r5, sp[0]
    retsp NSTACKWORDS

    .set FUNCTION_NAME.nstackwords,NSTACKWORDS
    .globl FUNCTION_NAME.nstackwords
    .set FUNCTION_NAME.maxcores,1
    .globl FUNCTION_NAME.maxcores
    .set FUNCTION_NAME.maxtimers,0
    .globl FUNCTION_NAME.maxtimers
    .set FUNCTION_NAME.maxchanends,0
    .globl FUNCTION_NAME.maxchanends
.Ltmp1:
    .size FUNCTION_NAME, .Ltmp1-FUNCTION_NAME
    .cc_bottom subsample_x_asm.function

#endif


