'''
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
'''

import tvm
import numpy
import tvm.testing
from tvm import te
import timeit
import time


# The size of the matrix
# (M, K) x (K, N)
# You are free to try out different shapes, sometimes TVM optimization outperforms numpy with MKL.

M = 60
K = 576
N = 50176

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

# Random generated tensor for testing
a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), dev)
b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), dev)

np_repeat = 10
np_runing_time = timeit.timeit(
    setup="import numpy\n"
    "M = " + str(M) + "\n"
    "K = " + str(K) + "\n"
    "N = " + str(N) + "\n"
    'dtype = "float32"\n'
    "a = numpy.random.rand(M, K).astype(dtype)\n"
    "b = numpy.random.rand(K, N).astype(dtype)\n",
    stmt="answer = numpy.dot(a, b)",
    number=np_repeat,
)
print("Numpy running time: %fms %f GFLOPS" % (1000.0 * np_runing_time / np_repeat, 2.0 * M * N * K / np_runing_time * np_repeat / 1e9 ))

answer = numpy.dot(a.numpy(), b.numpy())

time.sleep(1)



# Algorithm


A = te.placeholder((M, K), name='A')
B = te.placeholder((K, N), name='B')
k = te.reduce_axis((0, K), 'k')

packedB = te.compute(
    (K // kc, N // nc, nc // nr, kc, nr), lambda ko, no, nio, ki, ni: B[ko * kc + ki, no * nc + nio * nr + ni],
    name = 'packedB'
)
packedC = te.compute(
    (N // nc, M, nc),
    lambda no, m, ni: te.sum(A[ m,k] *
                                  packedB[k // kc,no,  ni // nr, tvm.tir.indexmod(k, kc), tvm.tir.indexmod(ni, nr)],
                                  axis=k),
    name = 'packedC'
)

C = te.compute(
    (M,N), lambda m,n: packedC[n//nc,m,tvm.tir.indexmod(n,nc)],
    name = 'C'
)

s = te.create_schedule(C.op)

no,m,ni = s[packedC].op.axis

mo, nio, mi, nii = s[packedC].tile(m,ni,mr,nr)
k = s[packedC].op.reduce_axis[0]
ko, ki = s[packedC].split(k,kc)
s[packedC].reorder(no,ko,nio,mo,ki,mi,nii)
s[packedC].unroll(mi)
s[packedC].vectorize(nii)
s[packedB].compute_at(s[packedC],ko)
s[packedB].vectorize(s[packedB].op.axis[4])

cm,cn = s[C].op.axis
cmo, cno, cmi, cni = s[C].tile(cm,cn,mc,nc)
s[C].reorder(cno,cmo,cmi,cni)
s[packedC].compute_at(s[C],cno)
s[C].vectorize(cni)

func = tvm.build(s, [A, B, C], target=target, name='mmult')
assert func
#print(func.get_source("asm"))
print(tvm.lower(s, [A, B, C], simple_mode=True))



c = tvm.nd.array(numpy.zeros((M,N), dtype=dtype), dev)
func(a, b, c)
tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, dev, number=10,repeat=5)
mean_time = evaluator(a, b, c).mean

print("Opt1: %fms, %f GFLOPS" % (mean_time * 1000, 2.0 * M * N * K / mean_time / 1e9))
'''
libpath = "/data/asm.so"
func.export_library(libpath)
'''
