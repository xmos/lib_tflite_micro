#ifdef __VX4A__	

	.text

    .globl fast_read_loop
    .globl fast_read_loop.nstackwords

	.p2align	4
	.type	fast_read_loop,@function

fast_read_loop:
	xm.entsp  64
	xm.stdsp  x19,x18,8            // Save x18..x23
	xm.stdsp  x21,x20,16
	xm.stdsp  x23,x22,24
//    tail dddd
    li        x21, 0
	sw        x24, 4(sp)           // Save x24 todo: dual issue
	add       x18, x14, 0          // Get argument 4
	add       x22, x16, 0          // Get argument 6
	li        x20, 27
	add       x23, x13, x20        // r9 <- 27 + read_adj
	slli      x20, x11, 8          // todo: make xm.shl and dual issue with prev
    or        x20, x20, x12        // r6 <- (addr << 8) | mode
	xm.rev8   x28, x20

    xm.unzip  x28, x20, 2
    xm.zip    x20, x28, 2
                            // r6 <- nibble_swapped(byte_revved(r6))
	lw        x13, 8(x10)
	{ li        x11, 1          ;	xm.ldwi   x12, 12(x10) }
	xm.outpwi x13, x11, 4
	xm.setci  x12, 15
	xm.syncr  x13
	{ xm.setci  x12, 7          ;	xm.ldwi   x28, 0(x10) }
	xm.out    x28, x21
	li        x10, 28  // todo: pack with setci
	xm.shl    x21, x20, x10
    lui       x24, 0x01101
    add       x24, x24, 0x011
	or        x21, x21, x24
	xm.out    x13, x21
	xm.setci  x12, 15
	srli      x20, x20, 4       // TODO: make shr and dual issue with setci
	xm.setpsc x13, x10
	xm.out    x13, x20
	li        x20, 18 // TODO: dual issue with out before
	xm.setpt  x13, x20 
	add       x20, x15, 0      // Get arg 5
	xm.in     x24, x13
	xm.setpt  x13, x23
    
    // This is where Timing matters
	xm.in     x10, x13
	beqz      x22, SendToMemory   // todo: dual issue with before

SendToChannel:
	xm.out    x22, x10
	addi      x20, x18, -1    // todo: dual issue
	beqz      x20, EndOfChanLoop
ChanLoop:
	xm.in     x18, x13
	xm.out    x22, x18
	addi      x20, x20, -1    // todo: dual issue
	bnez      x20, ChanLoop   // todo: dual issue
EndOfChanLoop:
	xm.outcti x22, 1
	tail      EndOfFunction  // todo: dual issue
    
SendToMemory:
	sw        x10, 0(x20)
	addi      x18, x18, -1   // todo: dual issue
	beqz      x18, EndOfMemoryLoop
	addi      x20, x20, 4  // todo: delete
MemoryLoop:
	xm.in     x19, x13
	sw        x19, 0(x20) 
	addi      x20, x20, 4 // todo: dual issue with in
	addi      x18, x18, -1 // todo: dual issue with sw
	bnez      x18, MemoryLoop
EndOfMemoryLoop:
EndOfFunction:
	xm.setci  x12, 7
	xm.out    x28, x11
	lw        x24, 4(sp) // todo: dual issue with xm.out
dddd:   
	xm.lddsp  x23,x22,24
	xm.lddsp  x21,x20,16
	xm.lddsp  x19,x18,8
	xm.retsp 64

    // todo: change those
	.set	fast_read_loop.nstackwords,8
	.set	fast_read_loop.maxcores,1
	.set	fast_read_loop.maxtimers,0
	.set	fast_read_loop.maxchanends,0
.Ltmp2:
	.size	fast_read_loop, .Ltmp2-fast_read_loop

#endif