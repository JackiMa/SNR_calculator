import numpy as np
import scipy.signal as signal

class myTF(signal.TransferFunction):
    def __mul__(self, other):
        if isinstance(other, signal.TransferFunction):
            # 相乘后的分子和分母多项式
            new_num = np.polymul(self.num, other.num)
            new_den = np.polymul(self.den, other.den)
            return signal.TransferFunction(new_num, new_den)
        else:
            raise ValueError("Multiplication is only supported between TransferFunction instances")
    
    def to_scipy(self):
        return signal.TransferFunction(self.num, self.den)

# 重载signal.TransferFunction的__mul__方法，使之可以直接相乘       
signal.TransferFunction.__mul__ = myTF.__mul__

from .OPAMP import *
from .SiPM import *

