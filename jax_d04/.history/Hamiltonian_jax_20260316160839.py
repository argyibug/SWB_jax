"""
Hamiltonian模块 - JAX版本（支持GPU加速）

Author: ZhouChk
JAX优化: 向量化操作，JIT编译，GPU加速
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from typing import Union, Tuple
from functools import partial
import IO

@partial(jax.jit, static_argnums=(15, 16))
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
    B = jnp.array([B1, B2, B3])
    A = jnp.array([A1, A2, A3])

    # JAX 合法循环：使用 lax.fori_loop，避免在 jit 中使用 Python range(tracer)
    n_bond_eff = jnp.minimum(jnp.asarray(bond_n, dtype=jnp.int32), jnp.asarray(bond_tab.shape[0], dtype=jnp.int32))

    def bond_body(b, H_cur):
        s1_idx = jnp.asarray(bond_tab[b, 2], dtype=jnp.int32)
        s2_idx = jnp.asarray(bond_tab[b, 3], dtype=jnp.int32)

        s1_upd_idx = s1_idx
        s1_dwd_idx = s1_idx + spin_n
        s1_dwo_idx = (spin_n - 1 - s1_idx) + 2 * spin_n
        s1_upo_idx = (spin_n - 1 - s1_idx) + 3 * spin_n

        s2_upd_idx = s2_idx
        s2_dwd_idx = s2_idx + spin_n
        s2_upo_idx = (spin_n - 1 - s2_idx) + 2 * spin_n
        s2_dwo_idx = (spin_n - 1 - s2_idx) + 3 * spin_n

        rij_x = bond_tab[b, 4]
        rij_y = bond_tab[b, 5]
        kr = k1 * rij_x + k2 * rij_y

        if bond_tab.shape[1] > 6:
            bond_type = jnp.asarray(bond_tab[b, 6], dtype=jnp.int32)
        else:
            bond_type = jnp.mod(b, 3)

        cos_term = Jplus[bond_type] * B[bond_type] * jnp.cos(kr)
        sin_term = Jplus[bond_type] * A[bond_type] * jnp.sin(kr)

        H_cur = H_cur.at[:, s1_upo_idx, s2_upd_idx].add(cos_term)
        H_cur = H_cur.at[:, s1_dwo_idx, s2_dwd_idx].add(cos_term)
        H_cur = H_cur.at[:, s2_upo_idx, s1_upd_idx].add(cos_term)
        H_cur = H_cur.at[:, s2_dwo_idx, s1_dwd_idx].add(cos_term)

        H_cur = H_cur.at[:, s1_upo_idx, s2_upo_idx].add(sin_term)
        H_cur = H_cur.at[:, s1_dwo_idx, s2_dwo_idx].add(sin_term)
        H_cur = H_cur.at[:, s1_upd_idx, s2_upd_idx].add(sin_term)
        H_cur = H_cur.at[:, s1_dwd_idx, s2_dwd_idx].add(sin_term)
        return H_cur

    H = jax.lax.fori_loop(0, n_bond_eff, bond_body, H)

    # Diagonal terms
    idx = jnp.arange(spin_n, dtype=jnp.int32)
    idx_upd = idx
    idx_dwd = idx + spin_n
    idx_upo = (spin_n - 1 - idx) + 2 * spin_n
    idx_dwo = (spin_n - 1 - idx) + 3 * spin_n

    H = H.at[:, idx_upd, idx_upd].set(diag_up)
    H = H.at[:, idx_dwd, idx_dwd].set(diag_dw)
    H = H.at[:, idx_upo, idx_upo].set(diag_up)
    H = H.at[:, idx_dwo, idx_dwo].set(diag_dw)

    # 添加对称破缺场项
    H = H.at[:, idx_upd, idx_dwo].add(hx_term)
    H = H.at[:, idx_dwo, idx_upd].add(hx_term)
    H = H.at[:, idx_dwd, idx_upo].add(hx_term)
    H = H.at[:, idx_upo, idx_dwd].add(hx_term)
    
    return H


