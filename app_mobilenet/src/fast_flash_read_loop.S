	.text

    .globl fast_read_loop
    .globl fast_read_loop.nstackwords
	.section	.cp.rodata.cst4,"aMc",@progbits,4
	.cc_top .Const0x01101011.data,.Const0x01101011
	.align	4
	.type	.Const0x01101011,@object
	.size	.Const0x01101011, 4
.Const0x01101011:
	.long	0x01101011
	.cc_bottom .Const0x01101011.data

	.text
	.align	16
	.type	fast_read_loop,@function
	.cc_top fast_read_loop.function,fast_read_loop

fast_read_loop:
	entsp 8
	std   r4, r5, sp[1]
	std   r6, r7, sp[2]
	std   r8, r9, sp[3]
    ldc r7, 0
	stw   r10, sp[1]
	ldw   r4, sp[9]
	ldw   r8, sp[11]
	ldc   r6, 27
	add   r9, r3, r6        // r9 <- 27 + read_adj
	shl   r6, r1, 8
    or    r6, r6, r2        // r6 <- (addr << 8) | mode
	byterev r11, r6

    unzip r11, r6, 2
	zip   r6, r11, 2
                            // r6 <- nibble_swapped(byte_revved(r6))
	ldw   r3, r0[2]
	ldc   r1, 1
	outpw res[r3], r1, 4
	ldw   r2, r0[3]
	setc  res[r2], 15
	syncr res[r3]
	setc  res[r2], 7
	ldw   r11, r0[0]
	out   res[r11], r7
	ldc   r0, 28
	shl   r7, r6, r0
	ldw   r10, cp[.Const0x01101011]
	or    r7, r7, r10
	out   res[r3], r7
	setc  res[r2], 15
	shr   r6, r6, 4
	setpsc res[r3], r0
	out   res[r3], r6
	ldc   r6, 18
	setpt res[r3], r6
	ldw   r6, sp[10]
	in    r10, res[r3]
	setpt res[r3], r9
    
    // This is where Timing matters
	in    r0, res[r3]
	bf    r8, SendToMemory

SendToChannel:
	out   res[r8], r0
	sub   r6, r4, 1
	bf    r6, EndOfChanLoop
ChanLoop:
	in    r4, res[r3]
	out   res[r8], r4
	sub   r6, r6, 1
	bt    r6, ChanLoop
EndOfChanLoop:
	outct res[r8], 1
	bu    EndOfFunction
    
SendToMemory:
	stw   r0, r6[0]
	sub   r4, r4, 1
	bf    r4, EndOfMemoryLoop
	add   r6, r6, 4
MemoryLoop:
	in    r5, res[r3]
	stw   r5, r6[0]
	add   r6, r6, 4
	sub   r4, r4, 1
	bt    r4, MemoryLoop
EndOfMemoryLoop:
EndOfFunction:
	setc  res[r2], 7
	out   res[r11], r1
	ldd   r4, r5, sp[1]
	ldd   r6, r7, sp[2]
	ldd   r8, r9, sp[3]
	ldw   r10, sp[1]
	retsp 8

	.cc_bottom fast_read_loop.function
	.set	fast_read_loop.nstackwords,8
	.set	fast_read_loop.maxcores,1
	.set	fast_read_loop.maxtimers,0
	.set	fast_read_loop.maxchanends,0
.Ltmp2:
	.size	fast_read_loop, .Ltmp2-fast_read_loop
