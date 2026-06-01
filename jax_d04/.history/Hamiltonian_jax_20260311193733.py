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
            bond_tab: jnp.ndarray, bond_n: int) -> jnp.ndarray:
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
    unitcell_size = 3  # 每个子晶格的自由度数
    mat_dim = 4 * unitcell_size  # 哈密顿量矩阵的维度
    
    # 构建哈密顿量（向量化方式）
    H = jnp.zeros((nk, mat_dim, mat_dim), dtype=jnp.complex128)
    
    # 使用at[].set()进行赋值（与NumPy版本保持一致）
    H = H.at[:, :, :].set(0)


    Jplus = jnp.array([J1plus, J2plus, J3plus])
    B=jnp.array([B1, B2, B3])
    A=jnp.array([A1, A2, A3])

    for b in range(bond_numbers):
        s1 = bond_indix[b, 0]
        s2 = bond_indix[b, 1]
        d1 = bond_vectors[b, 0]
        d2 = bond_vectors[b, 1]
        kr=k1*d1+k2*d2
        kv=-kr

        s1up_0 = s1
        s1dw_0 = s1 + unitcell_size
        s1up_1 = s1 + 3 * unitcell_size
        s1dw_1 = s1 + 2 * unitcell_size

        s2up_0 = s2
        s2dw_0 = s2 + unitcell_size
        s2up_1 = s2 + 3 * unitcell_size
        s2dw_1 = s2 + 2 * unitcell_size

        H = H.at[:, s1up_0, s2up_0].set(Jplus[b] * B[b] * jnp.cos(kr))
        H = H.at[:, s1dw_0, s2dw_0].set(Jplus[b] * B[b] * jnp.cos(kr))
        H = H.at[:, s1up_1, s2up_1].set(Jplus[b] * B[b] * jnp.cos(kr))
        H = H.at[:, s1dw_1, s2dw_1].set(Jplus[b] * B[b] * jnp.cos(kr))

    # B-sub matrix
    i=0
    j=0
    H = H.at[:, i+0, j+0].set(0)
    H = H.at[:, i+0, j+1].set(J1plus * B1 * jnp.cos(kr10))
    H = H.at[:, i+0, j+2].set(J3plus * B3 * jnp.cos(kr20))
    H = H.at[:, i+1, j+0].set(J1plus * B1 * jnp.cos(kr01))
    H = H.at[:, i+1, j+1].set(0)
    H = H.at[:, i+1, j+2].set(J2plus * B2 * jnp.cos(kr21))
    H = H.at[:, i+2, j+0].set(J3plus * B3 * jnp.cos(kr02))
    H = H.at[:, i+2, j+1].set(J2plus * B2 * jnp.cos(kr12))
    H = H.at[:, i+2, j+2].set(0)
    
    i=3
    j=3
    H = H.at[:, i+0, j+0].set(0)
    H = H.at[:, i+0, j+1].set(J1plus * B1 * jnp.cos(kr10))
    H = H.at[:, i+0, j+2].set(J3plus * B3 * jnp.cos(kr20))
    H = H.at[:, i+1, j+0].set(J1plus * B1 * jnp.cos(kr01))
    H = H.at[:, i+1, j+1].set(0)
    H = H.at[:, i+1, j+2].set(J2plus * B2 * jnp.cos(kr21))
    H = H.at[:, i+2, j+0].set(J3plus * B3 * jnp.cos(kr02))
    H = H.at[:, i+2, j+1].set(J2plus * B2 * jnp.cos(kr12))
    H = H.at[:, i+2, j+2].set(0)
    
    i=6
    j=6
    H = H.at[:, i+0, j+0].set(0)
    H = H.at[:, i+0, j+1].set(J2plus * B2 * jnp.cos(kr12))
    H = H.at[:, i+0, j+2].set(J3plus * B3 * jnp.cos(kr02))
    H = H.at[:, i+1, j+0].set(J2plus * B2 * jnp.cos(kr21))
    H = H.at[:, i+1, j+1].set(0)
    H = H.at[:, i+1, j+2].set(J1plus * B1 * jnp.cos(kr01))
    H = H.at[:, i+2, j+0].set(J3plus * B3 * jnp.cos(kr20))
    H = H.at[:, i+2, j+1].set(J1plus * B1 * jnp.cos(kr10))
    H = H.at[:, i+2, j+2].set(0)
    
    i=9
    j=9
    H = H.at[:, i+0, j+0].set(0)
    H = H.at[:, i+0, j+1].set(J2plus * B2 * jnp.cos(kr12))
    H = H.at[:, i+0, j+2].set(J3plus * B3 * jnp.cos(kr02))
    H = H.at[:, i+1, j+0].set(J2plus * B2 * jnp.cos(kr21))
    H = H.at[:, i+1, j+1].set(0)
    H = H.at[:, i+1, j+2].set(J1plus * B1 * jnp.cos(kr01))
    H = H.at[:, i+2, j+0].set(J3plus * B3 * jnp.cos(kr20))
    H = H.at[:, i+2, j+1].set(J1plus * B1 * jnp.cos(kr10))
    H = H.at[:, i+2, j+2].set(0)

    # A-sub matrix

    i=0
    j=9
    H = H.at[:, i+0, j+0].set(0)
    H = H.at[:, i+0, j+1].set(-J1plus * A1 * jnp.sin(kr10))
    H = H.at[:, i+0, j+2].set(-J3plus * A3 * jnp.sin(kr20))
    H = H.at[:, i+1, j+0].set(-J1plus * A1 * jnp.sin(kr01))
    H = H.at[:, i+1, j+1].set(0)
    H = H.at[:, i+1, j+2].set(-J2plus * A2 * jnp.sin(kr21))
    H = H.at[:, i+2, j+0].set(-J3plus * A3 * jnp.sin(kr02))
    H = H.at[:, i+2, j+1].set(-J2plus * A2 * jnp.sin(kr12))
    H = H.at[:, i+2, j+2].set(0)

    i=3
    j=6
    H = H.at[:, i+0, j+0].set(0)
    H = H.at[:, i+0, j+1].set(J1plus * A1 * jnp.sin(kr10))
    H = H.at[:, i+0, j+2].set(J1plus * A1 * jnp.sin(kr20))
    H = H.at[:, i+1, j+0].set(J3plus * A3 * jnp.sin(kr01))
    H = H.at[:, i+1, j+1].set(0)
    H = H.at[:, i+1, j+2].set(J3plus * A3 * jnp.sin(kr21))
    H = H.at[:, i+2, j+0].set(J2plus * A2 * jnp.sin(kr02))
    H = H.at[:, i+2, j+1].set(J2plus * A2 * jnp.sin(kr12))
    H = H.at[:, i+2, j+2].set(0)

    i=6
    j=3
    H = H.at[:, i+0, j+0].set(0)
    H = H.at[:, i+0, j+1].set(-J1plus * A1 * jnp.sin(kr10))
    H = H.at[:, i+0, j+2].set(-J1plus * A1 * jnp.sin(kr20))
    H = H.at[:, i+1, j+0].set(-J3plus * A3 * jnp.sin(kr01))
    H = H.at[:, i+1, j+1].set(0)
    H = H.at[:, i+1, j+2].set(-J3plus * A3 * jnp.sin(kr21))
    H = H.at[:, i+2, j+0].set(-J2plus * A2 * jnp.sin(kr02))
    H = H.at[:, i+2, j+1].set(-J2plus * A2 * jnp.sin(kr12))
    H = H.at[:, i+2, j+2].set(0)

    i=9
    j=0
    H = H.at[:, i+0, j+0].set(0)
    H = H.at[:, i+0, j+1].set(J1plus * A1 * jnp.sin(kr10))
    H = H.at[:, i+0, j+2].set(J3plus * A3 * jnp.sin(kr20))
    H = H.at[:, i+1, j+0].set(J1plus * A1 * jnp.sin(kr01))
    H = H.at[:, i+1, j+1].set(0)
    H = H.at[:, i+1, j+2].set(J2plus * A2 * jnp.sin(kr21))
    H = H.at[:, i+2, j+0].set(J3plus * A3 * jnp.sin(kr02))
    H = H.at[:, i+2, j+1].set(J2plus * A2 * jnp.sin(kr12))
    H = H.at[:, i+2, j+2].set(0)

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

@partial(jax.jit, static_argnums=())
def make_lattice() -> jnp.ndarray:
    """
    构建晶格 - JAX版本（完全向量化，支持GPU）
    
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
        bond_index: 
    """
    [dim, unit_vectors] = IO.read_unit_vectors(filepath='unit_vector.in')
    [spin_dim, spin_num, cell_spins] = IO.read_spin_in_cell(dim=dim, filepath='cellspin.in')
    spin_idx = jnp.arange(spin_num,dim)
    for i in range(spin_num):
        spin_idx = jnp.concatenate((spin_idx, jnp.arange(spin_num*i, spin_num*(i+1))))
    
    return H

