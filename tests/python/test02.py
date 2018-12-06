import numpy as np
import numcuda as nc
import unittest


def test_add(shape,dtype):
    if dtype in (np.float16,np.float32,np.float64):
        data_a = np.random.randn(*shape)*100
        data_a = data_a.astype(dtype)
        data_b = np.random.randn(*shape)*100
        data_b = data_b.astype(dtype)
    else:
        data_a=np.random.randint(200,size=shape)
        data_b=np.random.randint(200,size=shape)   
    cuda_array_a = nc.array(data_a)
    cuda_array_b = nc.array(data_b)
    
    data_c = data_a + data_b
    cuda_array_c = cuda_array_a + cuda_array_b
    res_data_c = cuda_array_c.asnumpy()

    #assert two array are equal
    np.testing.assert_equal(data_c,res_data_c) 

class TEST(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        print("********start test********")

    def test01(self):
        test_add((4,23,16),np.float16)

    # def test02(self):
    #     test_add((4,23,32),np.float32)

    # def test03(self):
    #     test_array((4,23,48),np.float64)

    # def test04(self):
    #     test_array((4,64),np.int8)

    # def test05(self):
    #     test_array((4,64),np.int16)

    # def test06(self):
    #     test_array((4,64),np.int32)

    # def test07(self):
    #     test_array((4,64),np.int64)

    # def test08(self):
    #     test_array((4,64),np.uint8)

    # def test09(self):
    #     test_array((4,64),np.uint16)

    # def test10(self):
    #     test_array((4,64),np.uint32)

    # def test11(self):
    #     test_array((4,64),np.uint64)  

if __name__ == '__main__':
    unittest.main()            
   