#profle numcuda
import numcuda as nc 
import numpy as np 
import timeit


def test01(shape):
    data_a = (np.random.randn(*shape)*100).astype(np.float16)
    data_b = (np.random.randn(*shape)*100).astype(np.float16)

    cuda_array_a = nc.array(data_a)
    cuda_array_b = nc.array(data_b)
    x = lambda : data_a+data_b
    y = lambda : cuda_array_a+cuda_array_b

    t1 = timeit.timeit(stmt=x, number=100)
    t2 = timeit.timeit(stmt=y, number=100)

    print("numpy time is:{}s".format(t1))
    print("numcuda time is:{}s".format(t2))
    print("numcuda is {0:.4f}x faster than numpy".format(t1/t2))

if __name__ == "__main__":
     test01((200,200,200))   