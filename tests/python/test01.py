import numpy as np
import numcuda as nc
import unittest

def test_array(shape,dtype):
    if dtype in (np.float16,np.float32,np.float64):
        data_array=np.random.randn(*shape)*100
        data_array = data_array.astype(dtype)
    else:
        data_array=np.random.randint(200,size=shape)
    
    test_data = nc.array(data_array)
    res_data = test_data.asnumpy()

    #assert two array are equal
    np.testing.assert_equal(data_array,res_data)  


class TEST(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        print("********start test********")

    def test01(self):
        test_array((4,23,16),np.float16)

    def test02(self):
        test_array((4,23,32),np.float32)

    def test03(self):
        test_array((4,23,48),np.float64)

    def test04(self):
        test_array((4,64),np.int8)

    def test05(self):
        test_array((4,64),np.int16)

    def test06(self):
        test_array((4,64),np.int32)

    def test07(self):
        test_array((4,64),np.int64)

    def test08(self):
        test_array((4,64),np.uint8)

    def test09(self):
        test_array((4,64),np.uint16)

    def test10(self):
        test_array((4,64),np.uint32)

    def test11(self):
        test_array((4,64),np.uint64)           


if __name__ == '__main__':
    unittest.main()