"""
Hamiltonian模块 - JAX版本（支持GPU加速）

Author: ZhouChk
JAX优化: 向量化操作，JIT编译，GPU加速
"""

import jax
import jax.numpy as jnp
from typing import Union, Tuple
from functools import partial
import IO

@partial(jax.jit, static_argnums=())
def Ham_jax(omega: float, k1: jnp.ndarray, k2: jnp.ndarray,
            A1: complex, A2: complex, A3: complex, 
            B1: float, B2: float, B3: float, 
            lambda_param: float, h: float,
            J1plus: float, J2plus: float, J3plus: float,
            bond_tab: jnp.ndarray, spin_n: int, bond_n: int) -> jnp.ndarray:
    """
    构建哈密顿量矩阵 - JAX版本（完全向量化，支持GPU）
    
    Parameters:
    -----------
    omega : float
        频率参数
    k1, k2 : jnp.ndarray
        动量分量 (nk,)
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
        哈密顿量矩阵，形状 (nk, 4 * unitcell_size, 4 * unitcell_size)
    """
    nk = k1.shape[0]
    
    # 使用jax.debug.print进行调试（JIT编译兼容）
    
    # 对角项
    diag_up = lambda_param
    diag_dw = lambda_param
    hx_term = h*0.5
    mat_dim = 4 * spin_n  # 哈密顿量矩阵的维度
    
    # 构建哈密顿量（向量化方式）
    H = jnp.zeros((nk, mat_dim, mat_dim), dtype=jnp.complex128)
    
    # 使用at[].set()进行赋值（与NumPy版本保持一致）
    H = H.at[:, :, :].set(0)


    Jplus = jnp.array([J1plus, J2plus, J3plus])
    B=jnp.array([B1, B2, B3])
    A=jnp.array([A1, A2, A3])

    for b in range(bond_n):
        s1_idx= bond_tab[b, 2]
        s2_idx= bond_tab[b, 3]

        s1_upd_idx = 

        rij= bond_tab[b, 5]
        kr = k1 * rij[0] + k2 * rij[1]

        H = H.at[:, s1up_0, s2up_0].set(Jplus[b] * B[b] * jnp.cos(kr))
        H = H.at[:, s1dw_0, s2dw_0].set(Jplus[b] * B[b] * jnp.cos(kr))
        H = H.at[:, s1up_1, s2up_1].set(Jplus[b] * B[b] * jnp.cos(kr))
        H = H.at[:, s1dw_1, s2dw_1].set(Jplus[b] * B[b] * jnp.cos(kr))

    # Diagonal terms
    H = H.at[:, 0, 0].set(diag_up)
    H = H.at[:, 1, 1].set(diag_up)
    H = H.at[:, 2, 2].set(diag_up)
    H = H.at[:, 3, 3].set(diag_dw)
    H = H.at[:, 4, 4].set(diag_dw)
    H = H.at[:, 5, 5].set(diag_dw)
    H = H.at[:, 6, 6].set(diag_up)
    H = H.at[:, 7, 7].set(diag_up)
    H = H.at[:, 8, 8].set(diag_up)
    H = H.at[:, 9, 9].set(diag_dw)
    H = H.at[:, 10, 10].set(diag_dw)
    H = H.at[:, 11, 11].set(diag_dw)

    # 添加对称破缺场项
    H = H.at[:, 0, 9].add(hx_term)
    H = H.at[:, 9, 0].add(hx_term)
    H = H.at[:, 1, 10].add(hx_term)
    H = H.at[:, 10, 1].add(hx_term)
    H = H.at[:, 2, 11].add(hx_term)
    H = H.at[:, 11, 2].add(hx_term)

    H = H.at[:, 3, 6].add(hx_term)
    H = H.at[:, 6, 3].add(hx_term)
    H = H.at[:, 4, 7].add(hx_term)
    H = H.at[:, 7, 4].add(hx_term)
    H = H.at[:, 5, 8].add(hx_term)
    H = H.at[:, 8, 5].add(hx_term)
    
    return H


