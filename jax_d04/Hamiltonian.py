"""
Hamiltonian模块

Author: ZhouChk
"""

import numpy as np
import scipy.linalg as la
from typing import Union, Tuple
from gamma_functions import (gamma_A, gamma_B, set_global_params)

def Ham(omega: float, k1: Union[float, np.ndarray], k2: Union[float, np.ndarray], 
        Q1: float, Q2: float, A1: complex, A2: complex, A3: complex, 
        B1: float, B2: float, B3: float, lambda_param: float, h: float) -> np.ndarray:
    """
    构建哈密顿量矩阵
    
    Parameters:
    -----------
    omega : float
        频率参数
    k1, k2 : float or ndarray
        动量分量
    Q1, Q2 : float
        磁序波矢
    A1, A2, A3 : complex
        鞍点参数A
    B1, B2, B3 : float
        鞍点参数B
    lambda_param : float
        拉格朗日乘数
    h : float
        对称破缺场
        
    Returns:
    --------
    ndarray
        4x4哈密顿量矩阵 (形状为 (nk, 4, 4) 如果k是数组)
    """
    # 确保k1, k2是数组
    k1 = np.atleast_1d(k1)
    k2 = np.atleast_1d(k2)
    nk = len(k1)
    
    H = np.zeros((nk, 4, 4), dtype=complex)
    
    H[:, 0, 0] = omega + lambda_param + gamma_B(k1 + Q1/2, k2 + Q2/2, B1, B2, B3)
    H[:, 0, 1] = -gamma_A(k1 + Q1/2, k2 + Q2/2, A1, A2, A3)
    H[:, 0, 2] = -h/2
    H[:, 0, 3] = 0
    
    H[:, 1, 0] = -gamma_A(k1 + Q1/2, k2 + Q2/2, A1, A2, A3)
    H[:, 1, 1] = omega + lambda_param + gamma_B(k1 + Q1/2, k2 + Q2/2, B1, B2, B3)
    H[:, 1, 2] = 0
    H[:, 1, 3] = -h/2
    
    H[:, 2, 0] = -h/2
    H[:, 2, 1] = 0
    H[:, 2, 2] = omega + lambda_param + gamma_B(-k1 + Q1/2, -k2 + Q2/2, B1, B2, B3)
    H[:, 2, 3] = -gamma_A(-k1 + Q1/2, -k2 + Q2/2, A1, A2, A3)

    H[:, 3, 0] = 0
    H[:, 3, 1] = -h/2
    H[:, 3, 2] = -gamma_A(-k1 + Q1/2, -k2 + Q2/2, A1, A2, A3)
    H[:, 3, 3] = omega + lambda_param + gamma_B(-k1 + Q1/2, -k2 + Q2/2, B1, B2, B3)
    
    return H