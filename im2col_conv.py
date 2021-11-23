import numpy as np
import numpy
import tvm.tir
from tvm import te
import tvm.testing
#定义inputs和filters的维度大小

Ci = 64
Co =60
Hi = 224
Wi = 224
kernel = 3
Pad = 1
Stride = 1
Ho = (Hi - kernel + 2 * Pad) // Stride + 1
Wo = (Wi - kernel + 2 * Pad) // Stride + 1

K = kernel * kernel * Ci
M = Co
N = Ho * Wo


mc = 60
nc = 224
kc = 144

mr = 5
nr = 16

# The default tensor type in tvm
dtype = "float32"

# using Intel AVX2(Advanced Vector Extensions) ISA for SIMD
# To get the best performance, please change the following line
# to llvm -mcpu=core-avx2, or specific type of CPU you use
target = "llvm -mcpu=znver2"
#target = "llvm"
dev = tvm.device(target, 0)

filter = tvm.nd.array(numpy.random.rand(Co,Ci,kernel,kernel).astype(dtype), dev)
inputimage = tvm.nd.array(numpy.random.rand(Ci,Hi,Wi).astype(dtype), dev)

#算法
InputImage = te.placeholder((Ci,Hi,Wi), name='Input')
Filter = te.placeholder((Co,Ci,kernel,kernel), name="Filter")
k = te.reduce_axis((0,kernel*kernel*Ci),name= "k")


#(Co,Ci,kernel,kernel)
packedFilter_pre = te.compute(
    (Co,kernel*kernel*Ci), lambda  co,h: Filter[co,h//(kernel*kernel),tvm.tir.indexmod(h,kernel*kernel)//kernel,tvm.tir.indexmod(tvm.tir.indexmod(h,kernel*kernel),kernel)],
    name= "packedFilter_pre"
)

s1 = te.create_schedule(packedFilter_pre.op)
packedFilter_func = tvm.build(s1, [Filter, packedFilter_pre], target=target, name='fpackedFilter')
assert packedFilter_func
packedfilter = tvm.nd.array(numpy.zeros((M,K), dtype=dtype), dev)
packedFilter_func (filter, packedfilter)
packedFilter = te.placeholder((M,K), name= "packedFilter")



InputPad = te.compute(
    (Ci,Hi+2*Pad,Wi+2*Pad),
    lambda c,h,w: tvm.tir.if_then_else(
        tvm.tir.all(w >= Pad, h >= Pad, w- Pad < Hi, h - Pad < Wi),
        InputImage[c, h-Pad, w-Pad],
        tvm.tir.const(0.0)
    ),
    name = "InputPad"
)



packedInput = te.compute(
    (K // kc, N // nc, kc, nc // nr,  nr), lambda ko, no, ki, nio, ni: InputPad[(ko*kc+ki)//(kernel*kernel),no+tvm.tir.indexmod(ko*kc+ki,kernel*kernel)//kernel,nio*nr+ni+tvm.tir.indexmod(tvm.tir.indexmod(ko*kc+ki,kernel*kernel),kernel)],
    name= "packedInput"
)

packedC  = te.compute(
    (N//nc, M, nc), lambda no, m, ni: te.sum(packedFilter[m, k]*packedInput[k//kc,no,tvm.tir.indexmod(k,kc),ni//nr,tvm.tir.indexmod(ni,nr)], axis = k), name="packedC"
)


C = te.compute(
    (M,N),
    lambda m, n: packedC[n//nc,m,tvm.tir.indexmod(n,nc)],
    name = "C"
)

s = te.create_schedule(C.op)

no,m,ni = s[packedC].op.axis

mo, nio, mi, nii = s[packedC].tile(m,ni,mr,nr)
k = s[packedC].op.reduce_axis[0]
ko, ki = s[packedC].split(k,kc)
s[packedC].reorder(no,ko,nio,mo,ki,mi,nii)
s[packedC].unroll(mi)
s[packedC].vectorize(nii)
bko, bno,bnio,bki,bni = s[packedInput].op.axis
s[packedInput].compute_at(s[packedC],ko)
s[packedInput].vectorize(s[packedInput].op.axis[4])


cm,cn = s[C].op.axis
cmo, cno, cmi, cni = s[C].tile(cm,cn,mc,nc)
s[C].reorder(cno,cmo,cmi,cni)
s[packedC].compute_at(s[C],cno)
s[C].vectorize(cni)



func = tvm.build(s,[packedFilter,InputImage, C], target=target, name='convolution')
assert func
print(tvm.lower(s,[packedFilter,InputImage, C],simple_mode= True))
c = tvm.nd.array(np.zeros((M,N), dtype=dtype), dev)
func(packedfilter, inputimage,  c)
#tvm.testing.assert_allclose(output.numpy(), output_np)
evaluator = func.time_evaluator(func.entry_name, dev, number=5,repeat=5)
mean_time = evaluator(packedfilter, inputimage, c).mean
print("Convolution: %f" % evaluator(packedfilter,inputimage, c).mean)
print("Opt1: %fms, %f GFLOPS" % (mean_time * 1000, (2.0 * Ci * kernel * kernel - 1 )* Hi * Wi * Co / mean_time / 1e9))
