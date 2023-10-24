    .issue_mode dual
    .cc_top transmit_tx.function,transmit_tx
    .globl transmit_tx
    .globl transmit_tx.nstackwords
    .linkset transmit_tx.nstackwords, 0
    .align 16
transmit_tx:
    { dualentsp 0                ; ldc r11, 6*4 }
    mul r1, r1, r11
    mul r1, r1, r2
    add r1, r3, r1
tx_loop:   
    { out res[r0], r3            ; ldw r3, r1[0] }
    { out res[r0], r3            ; ldw r3, r1[1] }
    { out res[r0], r3            ; ldw r3, r1[2] }
    { nop                         ; sub r2, r2, 1 }
    { out res[r0], r3            ; ldw r3, r1[3] }
    { out res[r0], r3            ; ldw r3, r1[4] }
    { out res[r0], r3            ; ldw r3, r1[5] }
    { bt  r2, tx_loop             ; add r1, r1, r11  }
    { out res[r0], r3            ; nop }
    { out res[r0], r3            ; nop } // dummy out
    { outct res[r0], 1            ; retsp 0 }
    .cc_bottom transmit_tx.function

    .cc_top receive_rx.function,receive_rx
    .globl receive_rx
    .globl receive_rx.nstackwords
    .linkset receive_rx.nstackwords, 0
    .align 16
    .skip 4
receive_rx:
    { dualentsp 0                ; ldc r11, 6*4 }
    mul r2, r2, r11
    ldw r3, r0[0]
    mul r2, r2, r3
    ldw r0, r0[1]
    { add r2, r0, r2            ; in r0, res[r1] }  // Dummy in
    in r0, res[r1]
rx_loop:   
    { in r0, res[r1]             ; stw r0, r2[0] }
    { in r0, res[r1]             ; stw r0, r2[1] }
    { in r0, res[r1]             ; stw r0, r2[2] }
    { nop                         ; sub r3, r3, 1 }
    { in r0, res[r1]             ; stw r0, r2[3] }
    { in r0, res[r1]             ; stw r0, r2[4] }
    { in r0, res[r1]             ; stw r0, r2[5] }
    { bt  r3, rx_loop             ; add r2, r2, r11 }
    { chkct res[r1], 1            ; retsp 0 }
    .cc_bottom receive_rx.function
    
