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
nc = 64
kc = 192

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




input_np  = np.random.uniform(size=(Ci,Hi,Wi)).astype(dtype)
filter_np = np.random.uniform(size=(Co,Ci,kernel,kernel)).astype(dtype)

input = tvm.nd.array(input_np, dev)
filter = tvm.nd.array(filter_np, dev)
'''


input_np  = np.ones((Ci,Hi,Wi), dtype=dtype)
filter_np = np.ones((Co,Ci,kernel,kernel), dtype=dtype)
input = tvm.nd.array(input_np, dev)
filter = tvm.nd.array(filter_np, dev)

padinput = np.zeros((Ci,Hi+2*Pad,Wi+2*Pad),dtype=dtype)
output_np = np.zeros((Co,Ho, Wo), dtype=dtype)

for n in range(0,Ci):
    for i in range(0,Hi+2*Pad):
        for j in range(0,Wi+2*Pad):
            if (i>=Pad) and (j>=Pad) and (i-Pad < Hi) and (j-Pad < Wi):

                padinput[n,i,j] = input_np[n,i-Pad,j-Pad]
            else:


                padinput[n, i, j]= 0

for co in range(Co):
    for ci in range(Ci):
        for ho in range(Ho):
            for wo in range(Wo):
                for hf in range(kernel):
                    for wf in range(kernel):
                        output_np[co,ho,wo] = padinput[ci,ho+hf,wo+wf]*filter_np[co,ci,hf,wf]+output_np[co,ho,wo]


print(output_np)

'''
#算法
Input = te.placeholder((Ci,Hi,Wi), name='Input')
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
packedfilter = tvm.nd.array(numpy.zeros((Co,kernel*kernel*Ci), dtype=dtype), dev)
packedFilter_func (filter, packedfilter)
packedFilter = te.placeholder((Co,kernel*kernel*Ci), name= "packedB")

InputPad = te.compute(
    (Ci,Hi+2*Pad,Wi+2*Pad),
    lambda c,h,w: tvm.tir.if_then_else(
        tvm.tir.all(w >= Pad, h >= Pad, w- Pad < Hi, h - Pad < Wi),
        Input[c, h-Pad, w-Pad],
        tvm.tir.const(0.0)
    ),
    name = "InputPad"
)

'''
# (Ci,Hi,Wi)
packedInput = te.compute(
    (kernel*kernel*Ci,Ho*Wo), lambda h, w: InputPad[h//(kernel*kernel),(w//Wo)+ tvm.tir.indexmod(h,kernel*kernel)//kernel, tvm.tir.indexmod(w,Ho)+tvm.tir.indexmod(tvm.tir.indexmod(h,kernel*kernel),kernel)],
    name= "packedInput"
)



ppackedInput = te.compute(
    ((Ho*Wo)//nc,(kernel*kernel*Ci), nc), lambda no, K, ni: packedInput[k, no*nc+ni] ,  name="ppackedInput"
)
'''

packedInput = te.compute(
    (K,N), lambda k, n: InputPad[k//(kernel*kernel),n//Wo,tvm.tir.indexmod(n,Wo)+tvm.tir.indexmod(k,kernel*kernel)],
    name= "packedInput"
)

s = te.create_schedule(packedInput.op)
func = tvm.build(s,[Input, packedInput], target=target, name='convolution')
assert func
print(tvm.lower(s,[Input, packedInput],simple_mode= True))
output = tvm.nd.array(np.zeros((Co,Ho, Wo), dtype=dtype), dev)
func(input, packedInput)
#tvm.testing.assert_allclose(output.numpy(), output_np)
evaluator = func.time_evaluator(func.entry_name, dev, number=10,repeat=10)
mean_time = evaluator(input, packedInput).mean
print("Convolution: %f" % evaluator(input, packedInput).mean)
#print("Opt1: %fms, %f GFLOPS" % (mean_time * 1000, (2.0 * Ci * kernel * kernel - 1 )* Hi * Wi * Co / mean_time / 1e9))

'''
packedB = te.compute(
    (K // kc, N // nc, nc // nr, kc, nr), lambda ko, no, nio, ki, nii: InputPad[], 
    name = 'packedB'
)
'''
'''

im2col = te.compute(

)

packedOutput  = te.compute(
    (Co, Ho*Wo), lambda m, n: te.sum(packedFilter[m, k]*ppackedInput[n//nc,k,tvm.tir(n,nc)], axis = k), name="packedOutput"
)


Output = te.compute(
    (Co,Ho,Wo),
    lambda co,ho,wo: packedOutput[co,ho*Wo+wo],
    name = "Output"
)

s = te.create_schedule(Output.op)



func = tvm.build(s,[Input, packedFilter, Output], target=target, name='convolution')
assert func
print(tvm.lower(s,[Input, packedFilter,  Output],simple_mode= True))
output = tvm.nd.array(np.zeros((Co,Ho, Wo), dtype=dtype), dev)
func(input, packedfilter, output)
#tvm.testing.assert_allclose(output.numpy(), output_np)
evaluator = func.time_evaluator(func.entry_name, dev, number=10,repeat=10)
mean_time = evaluator(input, packedfilter, output).mean
print("Convolution: %f" % evaluator(input, packedfilter, output).mean)
print("Opt1: %fms, %f GFLOPS" % (mean_time * 1000, (2.0 * Ci * kernel * kernel - 1 )* Hi * Wi * Co / mean_time / 1e9))
'''