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
Pad = 0
Stride = 1
Ho = (Hi - kernel + 2 * Pad) // Stride
Wo = (Wi - kernel + 2 * Pad) // Stride

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

input_np  = np.ones((Ci,Hi,Wi), dtype=dtype)
input = tvm.nd.array(input_np.astype(dtype), dev)
Input = te.placeholder((Ci,Hi,Wi), name='Input')
print(input.numpy() )
'''

packedInput = te.compute(
    (K,N), lambda k, n: Input[k//(kernel*kernel),n//Wo+tvm.tir.indexmod(k,kernel*kernel)//kernel,tvm.tir.indexmod(n,Wo)+tvm.tir.indexmod(tvm.tir.indexmod(k,kernel*kernel),kernel)],
    name= "packedInput"
)
'''
packedInput = te.compute(
    (K // kc, N // nc, nc // nr, kc, nr), lambda ko, no, nio, ki, ni: Input[(ko*kc+ki)//(kernel*kernel),no+tvm.tir.indexmod(ko*kc+ki,kernel*kernel)//kernel,nio*nr+ni+tvm.tir.indexmod(tvm.tir.indexmod(ko*kc+ki,kernel*kernel),kernel)],
    name= "packedInput"
)

s = te.create_schedule(packedInput.op)
func = tvm.build(s,[Input, packedInput], target=target, name='convolution')
assert func
print(tvm.lower(s,[Input, packedInput],simple_mode= True))
packedinput = tvm.nd.array(np.zeros((K // kc, N // nc, nc // nr, kc, nr), dtype=dtype), dev)
func(input, packedinput)
#tvm.testing.assert_allclose(output.numpy(), output_np)
evaluator = func.time_evaluator(func.entry_name, dev, number=10,repeat=10)
mean_time = evaluator(input, packedinput).mean
print("Convolution: %f" % evaluator(input, packedinput).mean)
print(packedinput.numpy())

#(K // kc, N // nc, nc // nr, kc, nr)


