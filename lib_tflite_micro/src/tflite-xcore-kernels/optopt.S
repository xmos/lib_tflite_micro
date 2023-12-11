
	.section	.cp.rodata.cst4,"aMc",@progbits,4
	.cc_top .LCPI10_0.data,.LCPI10_0
	.align	4
	.type	.LCPI10_0,@object
	.size	.LCPI10_0, 4
.LCPI10_0:
	.long	4294967292
	.cc_bottom .LCPI10_0.data
	.text
	.globl	xc_transpose_conv2d_float_kw5xh2_stride_h3_packed_opt_S
	.align	4
	.type	xc_transpose_conv2d_float_kw5xh2_stride_h3_packed_opt_S,@function
	.cc_top xc_transpose_conv2d_float_kw5xh2_stride_h3_packed_opt_S.function,xc_transpose_conv2d_float_kw5xh2_stride_h3_packed_opt_S
xc_transpose_conv2d_float_kw5xh2_stride_h3_packed_opt_S:
	.issue_mode dual
	{
		nop
		dualentsp 30
	}
	std r5, r4, sp[11]
	{
		nop
		stw r6, sp[24]
	}
	{
		nop
		stw r7, sp[25]
	}
	{
		nop
		stw r8, sp[26]
	}
	{
		nop
		stw r9, sp[27]
	}
	{
		nop
		stw r10, sp[28]
	}
	{
		nop
		stw r2, sp[15]
	}
	{
		nop
		stw r1, sp[2]
	}
	{
		nop
		stw r0, sp[20]
	}
	{
		nop
		ldw r11, sp[36]
	}
	{
		nop
		ldw r1, sp[37]
	}
	{
		nop
		stw r1, sp[13]
	}
	{
		nop
		ldw r2, sp[34]
	}
	{
		nop
		ldw r1, sp[35]
	}
	{
		nop
		stw r2, sp[17]
	}
	{
		nop
		stw r1, sp[16]
	}
	{
		nop
		ldw r1, sp[32]
	}
	{
		nop
		ldw r2, sp[33]
	}
	{
		nop
		stw r2, sp[10]
	}
	{
		mov r1, r11
		stw r1, sp[12]
	}
	{
		zext r1, 2
		nop
	}
    ecallt   r1
	{
		mkmsk r2, 1
		ldw r1, sp[31]
	}
	{
		lss r2, r1, r2
		nop
	}
	bt r2, .LBB10_8
	{
		nop
		ldw r5, sp[12]
	}
	{
		sub r2, r5, 1
		sub r1, r1, 1
	}
	{
		nop
		stw r1, sp[19]
	}
	{
		nop
		ldw r1, sp[13]
	}
	{
		nop
		ldw r4, sp[38]
	}
	{
		sub r4, r4, r1
		nop
	}
	ldaw r10, r3[r1]
	{
		nop
		ldw r0, sp[20]
	}
	ldaw r0, r0[r1]
	{
		nop
		ldw r3, sp[10]
	}
	mul r1, r3, r5
	{
		shl r1, r1, 2
		nop
	}
	{
		shl r9, r3, 2
		stw r1, sp[18]
	}
	{
		ldc r1, 0
		nop
	}
.LBB10_3:
	{
		nop
		stw r1, sp[21]
	}
	{
		ldc r3, 0
		ldw r1, sp[12]
	}
	{
		lss r1, r3, r1
		mov r8, r0
	}
	{
		mov r5, r3
		nop
	}
	bf r1, .LBB10_7
.LBB10_4:
	{
		nop
		ldw r1, sp[13]
	}
	{
		nop
		ldw r3, sp[38]
	}
	{
		lss r7, r1, r3
		mov r1, r8
	}
	{
		mov r3, r10
		mov r6, r4
	}
	bf r7, .LBB10_6
.LBB10_5:
	{
		nop
		ldw r7, r3[0]
	}
	{
		sub r6, r6, 1
		stw r7, r1[0]
	}
	{
		add r3, r3, 4
		add r1, r1, 4
	}
	bt r6, .LBB10_5
.LBB10_6:
	{
		add r1, r5, 1
		add r8, r8, r9
	}
	{
		eq r3, r5, r2
		mov r5, r1
	}
	bf r3, .LBB10_4
.LBB10_7:
	{
		nop
		ldw r5, sp[21]
	}
	{
		add r1, r5, 1
		ldw r3, sp[18]
	}
	{
		add r0, r0, r3
		ldw r3, sp[19]
	}
	{
		eq r3, r5, r3
		nop
	}
	bf r3, .LBB10_3
.LBB10_8:
	{
		mkmsk r1, 1
		ldw r4, sp[17]
	}
	{
		lss r1, r4, r1
		ldw r2, sp[20]
	}
	bt r1, .LBB10_25
	lda16 r0, r11[r11]
	{
		nop
		stw r0, sp[14]
	}
	{
		nop
		ldw r0, sp[38]
	}
	{
		sub r0, r0, 1
		nop
	}
	{
		nop
		stw r0, sp[19]
	}
	{
		nop
		ldw r3, sp[16]
	}
	mul r0, r11, r3
	{
		sub r0, r3, 1
		stw r0, sp[1]
	}
	{
		sub r0, r4, 1
		stw r0, sp[8]
	}
	{
		shl r0, r11, 2
		stw r0, sp[4]
	}
	{
		ldc r1, 0
		stw r0, sp[7]
	}
	{
		mkmsk r0, 1
		nop
	}
	{
		lss r0, r3, r0
		nop
	}
	{
		ldc r5, 24
		stw r0, sp[3]
	}
	{
		ldc r7, 16
		ldc r4, 8
	}
	{
		mov r3, r1
		mov r6, r1
	}
.LBB10_10:
	{
		nop
		ldw r0, sp[3]
	}
	bt r0, .LBB10_24
	{
		shl r0, r6, 1
		nop
	}
	{
		nop
		stw r0, sp[11]
	}
	{
		nop
		stw r6, sp[5]
	}
	{
		nop
		ldw r0, sp[2]
	}
	ldaw r0, r0[r3]
	{
		nop
		stw r0, sp[17]
	}
	{
		ldc r0, 0
		stw r3, sp[6]
	}
	{
		mov r3, r0
		nop
	}
.LBB10_12:
	{
		mkmsk r0, 1
		stw r3, sp[9]
	}
	{
		sub r1, r0, r3
		nop
	}
	{
		lsu r1, r0, r1
		stw r1, sp[16]
	}
	{
		ldc r0, 0
		nop
	}
	bu .LBB10_13
.LBB10_21:
	{
		nop
		ldw r1, sp[18]
	}
	{
		add r0, r1, 1
		nop
	}
.xtaloop 3
	{
		eq r1, r1, 2
		nop
	}
.LBB10_13:
	{
		nop
		stw r0, sp[18]
	}
	bt r1, .LBB10_22
	{
		nop
		ldw r0, sp[13]
	}
	{
		nop
		ldw r1, sp[38]
	}
	{
		lss r1, r0, r1
		nop
	}
	bf r1, .LBB10_21
	{
		nop
		ldw r0, sp[11]
	}
	{
		nop
		ldw r1, sp[18]
	}
	{
		add r1, r1, r0
		ldw r0, sp[12]
	}
	mul r1, r1, r0
	{
		add r1, r1, 1
		ldw r0, sp[10]
	}
	mul r1, r1, r0
	{
		nop
		ldw r0, sp[13]
	}
	{
		add r8, r1, r0
		mov r3, r0
	}
.LBB10_16:
	{
	    shl r1, r3, 1
		ldw r2, r2[r8]
	}
	{
		add r1, r1, r3
		ldw r0, sp[18]
	}
	{
		nop
		stw r3, sp[21]
	}
	{
		add r1, r1, r0
		ldw r0, sp[16]
	}
	lda16 r1, r0[r1]
	{
		ldc r10, 30
		ldw r0, sp[14]
	}
	mul r1, r0, r1
	ashr r9, r1, 32
	{
		shr r9, r9, r10
		nop
	}
	{
		add r1, r1, r9
		nop
	}
	ldw r0, cp[.LCPI10_0]
	{
		and r1, r1, r0
		ldw r0, sp[15]
	}
	{
		add r1, r0, r1
		ldw r10, sp[17]
	}
	{
		shr r9, r11, 2
		nop
	}
.LBB10_19:
	{
		add r1, r1, 4
		ldw r6, r1[0]
	}
	{
		sub r9, r9, 1
		ldw r0, r10[0]
	}
	lextract r3, r6, r6, r5, 32
    
	fmacc r2, r2, r0, r3
	{
		add r1, r1, 4
		ldw r0, r1[0]
	}
	lextract r6, r0, r6, r7, 32
	{
	    ldc r4, 8
		ldw r3, r10[1]
	}
	fmacc r2, r2, r3, r6
	{
		add r1, r1, 4
		ldw r3, r1[0]
	}
	lextract r0, r3, r0, r4, 32
	ldd r6, r4, r10[1]
	fmacc r2, r2, r6, r0
	fmacc r2, r2, r4, r3
    
    {
		add r10, r10, r7
	    bt r9, .LBB10_19
    }
	{
		nop
		ldw r0, sp[20]
	}
	stw r2, r0[r8]
	{
		mov r2, r0
		add r8, r8, 1
	}
	{
		nop
		ldw r3, sp[21]
	}
	{
		add r0, r3, 1
		ldw r1, sp[19]
	}
	{
		eq r1, r3, r1
		mov r3, r0
	}
	bf r1, .LBB10_16
	bu .LBB10_21
.LBB10_22:
	{
		nop
		ldw r6, sp[9]
	}
	{
		add r0, r6, 1
		ldw r1, sp[7]
	}
	{
		nop
		ldw r3, sp[17]
	}
	{
		add r3, r3, r1
		nop
	}
	{
		nop
		stw r3, sp[17]
	}
	{
		nop
		ldw r1, sp[8]
	}
	{
		eq r1, r6, r1
		mov r3, r0
	}
	bf r1, .LBB10_12
	{
		nop
		ldw r0, sp[1]
	}
	{
		nop
		ldw r3, sp[6]
	}
	{
		add r3, r0, r3
		ldw r6, sp[5]
	}
.LBB10_24:
	{
		add r0, r6, 1
		ldw r1, sp[4]
	}
	{
		eq r1, r6, r1
		mov r6, r0
	}
	bf r1, .LBB10_10
.LBB10_25:
	{
		nop
		ldw r10, sp[28]
	}
	{
		nop
		ldw r8, sp[26]
	}
	{
		nop
		ldw r9, sp[27]
	}
	{
		nop
		ldw r6, sp[24]
	}
	{
		nop
		ldw r7, sp[25]
	}
	ldd r5, r4, sp[11]
	{
		nop
		retsp 30
	}
.LBB10_26:
	.cc_bottom xc_transpose_conv2d_float_kw5xh2_stride_h3_packed_opt_S.function
	.set	xc_transpose_conv2d_float_kw5xh2_stride_h3_packed_opt_S.nstackwords,(__assert_func.nstackwords + 30)
	.globl	xc_transpose_conv2d_float_kw5xh2_stride_h3_packed_opt_S.nstackwords
	.set	xc_transpose_conv2d_float_kw5xh2_stride_h3_packed_opt_S.maxcores,__assert_func.maxcores $M 1
	.globl	xc_transpose_conv2d_float_kw5xh2_stride_h3_packed_opt_S.maxcores
	.set	xc_transpose_conv2d_float_kw5xh2_stride_h3_packed_opt_S.maxtimers,__assert_func.maxtimers $M 0
	.globl	xc_transpose_conv2d_float_kw5xh2_stride_h3_packed_opt_S.maxtimers
	.set	xc_transpose_conv2d_float_kw5xh2_stride_h3_packed_opt_S.maxchanends,__assert_func.maxchanends $M 0
	.globl	xc_transpose_conv2d_float_kw5xh2_stride_h3_packed_opt_S.maxchanends
.Ltmp10:
	.size	xc_transpose_conv2d_float_kw5xh2_stride_h3_packed_opt_S, .Ltmp10-xc_transpose_conv2d_float_kw5xh2_stride_h3_packed_opt_S
