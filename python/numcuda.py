import os
import sys

def _find_lib(lib_name="lib_numcuda.so"):
    path = os.path.dirname(os.path.abspath(__file__))
    basepath,_ = os.path.split(path)
    
    lib_path = os.path.join(basepath,lib_name)
    if os.path.exists(lib_path):
        sys.path.append(basepath)
        return True

    basepath = os.path.join(basepath,"build")
    lib_path = os.path.join(basepath,lib_name)
    if os.path.exists(lib_path):
        sys.path.append(basepath)
        return True  

    return False


if _find_lib():
    from lib_numcuda import *
else:
    raise ImportError("can not import lib_numcuda!!!")

