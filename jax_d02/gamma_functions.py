"""
gamma函数模块

Author: ZhouChk

"""

import numpy as np
from typing import Union, Tuple

# 全局变量
class GlobalParams:
    """存储全局参数的类"""
    J1plus: float = 0.0
    J2plus: float = 0.0  
    J3plus: float = 0.0
    Q1: float = 0.0
    Q2: float = 0.0

# 创建全局参数实例
_globals = GlobalParams()

def set_global_params(**kwargs):
    """设置全局参数"""
    for key, value in kwargs.items():
        if hasattr(_globals, key):
            setattr(_globals, key, value)
        else:
            raise ValueError(f"未知参数: {key}")

def gamma_A(k1: Union[float, np.ndarray], 
           k2: Union[float, np.ndarray], 
           A1: complex, A2: complex, A3: complex) -> Union[complex, np.ndarray]:
    """
    计算gamma_A函数
    
    Parameters:
    -----------
    k1, k2 : float or ndarray
        动量分量
    A1, A2, A3 : complex
        鞍点参数A
        
    Returns:
    --------
    complex or ndarray
        gamma_A的值
    """
    result = -1j * (_globals.J1plus * A1 * np.sin(k1) + 
                    _globals.J2plus * A2 * np.sin(k2) + 
                    _globals.J3plus * A3 * np.sin(k2 - k1))
    return result

def gamma_B(k1: Union[float, np.ndarray], 
           k2: Union[float, np.ndarray], 
           B1: complex, B2: complex, B3: complex) -> Union[complex, np.ndarray]:
    """
    计算gamma_B函数
    
    Parameters:
    -----------
    k1, k2 : float or ndarray
        动量分量
    B1, B2, B3 : float
        鞍点参数B
        
    Returns:
    --------
    float or ndarray
        gamma_B的值
    """
    result = (_globals.J1plus * B1 * np.cos(k1) + 
              _globals.J2plus * B2 * np.cos(k2) + 
              _globals.J3plus * B3 * np.cos(k2 - k1))
    return result
