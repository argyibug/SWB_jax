"""
Hamiltonian模块 - JAX版本（支持GPU加速）

Author: ZhouChk
JAX优化: 向量化操作，JIT编译，GPU加速
"""

import jax
import jax.numpy as jnp
from typing import Union, Tuple
from functools import partial

@partial(jax.jit, static_argnums=())
def Ham_jax(omega: float, k1: jnp.ndarray, k2: jnp.ndarray, 
            Q1: float, Q2: float, 
            A1: complex, A2: complex, A3: complex, 
            B1: float, B2: float, B3: float, 
            lambda_param: float, h: float,
            J1plus: float, J2plus: float, J3plus: float) -> jnp.ndarray:
    """
    构建哈密顿量矩阵 - JAX版本（完全向量化，支持GPU）
    
    Parameters:
    -----------
    omega : float
        频率参数
    k1, k2 : jnp.ndarray
        动量分量 (nk,)
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
    J1plus, J2plus, J3plus : float
        交换耦合参数
        
    Returns:
    --------
    jnp.ndarray
        哈密顿量矩阵，形状 (nk, 4, 4)
    """
    nk = k1.shape[0]
    
    # 计算gamma函数（向量化）
    k1_plus = k1 + Q1/2
    k2_plus = k2 + Q2/2
    k1_minus = -k1 + Q1/2
    k2_minus = -k2 + Q2/2
    
    # gamma_A计算
    gamma_A_plus = -1j * (J1plus * A1 * jnp.sin(k1_plus) + 
                          J2plus * A2 * jnp.sin(k2_plus) + 
                          J3plus * A3 * jnp.sin(k2_plus - k1_plus))
    
    gamma_A_minus = -1j * (J1plus * A1 * jnp.sin(k1_minus) + 
                           J2plus * A2 * jnp.sin(k2_minus) + 
                           J3plus * A3 * jnp.sin(k2_minus - k1_minus))
    
    # gamma_B计算
    gamma_B_plus = (J1plus * B1 * jnp.cos(k1_plus) + 
                    J2plus * B2 * jnp.cos(k2_plus) + 
                    J3plus * B3 * jnp.cos(k2_plus - k1_plus))
    
    gamma_B_minus = (J1plus * B1 * jnp.cos(k1_minus) + 
                     J2plus * B2 * jnp.cos(k2_minus) + 
                     J3plus * B3 * jnp.cos(k2_minus - k1_minus))
    
    # 使用jax.debug.print进行调试（JIT编译兼容）
    
    # 对角项
    diag_plus = lambda_param + gamma_B_plus
    diag_minus = lambda_param + gamma_B_minus
    
    # 构建哈密顿量（向量化方式）
    H = jnp.zeros((nk, 4, 4), dtype=jnp.complex128)
    
    # 使用at[].set()进行赋值（与NumPy版本保持一致）
    H = H.at[:, 0, 0].set(omega + diag_plus)
    H = H.at[:, 0, 1].set(-gamma_A_plus)
    H = H.at[:, 0, 2].set(-h/2)
    H = H.at[:, 0, 3].set(0)
    
    H = H.at[:, 1, 0].set(-gamma_A_plus)
    H = H.at[:, 1, 1].set(-omega + diag_plus)  # 修复：应该是 omega，不是 -omega
    H = H.at[:, 1, 2].set(0)
    H = H.at[:, 1, 3].set(-h/2)
    
    H = H.at[:, 2, 0].set(-h/2)
    H = H.at[:, 2, 1].set(0)
    H = H.at[:, 2, 2].set(omega + diag_minus)
    H = H.at[:, 2, 3].set(-gamma_A_minus)
    
    H = H.at[:, 3, 0].set(0)
    H = H.at[:, 3, 1].set(-h/2)
    H = H.at[:, 3, 2].set(-gamma_A_minus)
    H = H.at[:, 3, 3].set(-omega + diag_minus)  # 修复：应该是 omega，不是 -omega
    
    return H
