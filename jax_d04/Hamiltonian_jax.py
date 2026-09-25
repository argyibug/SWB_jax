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
    spin_n = 1
    
    # 对角项
    diag_up = lambda_param
    diag_dw = lambda_param
    hx_term = h * 0.5
    mat_dim = 4 * spin_n  # 哈密顿量矩阵的维度
    
    # 构建哈密顿量（向量化方式）
    H = jnp.zeros((nk, mat_dim, mat_dim), dtype=jnp.complex128)
    
    # 使用at[].set()进行赋值（与NumPy版本保持一致）
    H = H.at[:, :, :].set(0)


    Jplus = jnp.array([J1plus, J2plus, J3plus])
    B = jnp.array([B1, B2, B3])
    A = jnp.array([A1, A2, A3])
    # jax.debug.print("B参数: B1={B1}, B2={B2}, B3={B3}", B1=B1, B2=B2, B3=B3, ordered=True)
    # jax.debug.print("A参数: A1={A1}, A2={A2}, A3={A3}", A1=A1, A2=A2, A3=A3, ordered=True)

    # jax.debug.print("A,B参数: A1={A1}, B1={B1}, lamdba={lambda_param}", A1=A1, B1=B1, lambda_param=lambda_param)

    # JAX 合法循环：使用 lax.fori_loop，避免在 jit 中使用 Python range(tracer)
    n_bond_eff = jnp.minimum(jnp.asarray(bond_n, dtype=jnp.int32), jnp.asarray(bond_tab.shape[0], dtype=jnp.int32))

    # 统一契约: [i, j, site_i, site_j, lattice_rij_x, lattice_rij_y, real_rij_x, real_rij_y, bond_type]
    # 即 bond_type 固定在索引 8。
    if bond_tab.ndim != 2 or bond_tab.shape[1] < 9:
        raise ValueError(
            f"bond_tab 需为至少9列二维数组，当前 shape={bond_tab.shape}；"
            "期望列顺序为 [i, j, site_i, site_j, lattice_rij_x, lattice_rij_y, real_rij_x, real_rij_y, bond_type]"
        )

    from Hamiltonian_jax import bond_body
    H = jax.lax.fori_loop(0, n_bond_eff, lambda b, H_cur: bond_body(b, H_cur, B, A, bond_tab, spin_n, mat_dim, k1, k2), H)

    # jax.debug.print("H={}", H[10, :, :])


    # Diagonal terms
    idx = jnp.arange(spin_n, dtype=jnp.int32)
    # jax.debug.print("计算对角项索引: idx={}", idx)

    # hx_term = 0
    # diag_up = 0
    # diag_dw = 0
    from Hamiltonian_jax import compute_diag_term
    H = jax.lax.fori_loop(0, spin_n, lambda i, H_cur: compute_diag_term(i, H_cur, diag_up, diag_dw, hx_term, spin_n, mat_dim), H)
    
    return H

def bond_body(b, H_cur, B_tab, A_tab, bond_tab, spin_n, mat_dim, k1, k2):
    # jax.debug.print("bond_table[{b}] = {bond}", b=b, bond=bond_tab[b, :])
    s1_idx = jnp.asarray(bond_tab[b, 2], dtype=jnp.int32)
    s2_idx = jnp.asarray(bond_tab[b, 3], dtype=jnp.int32)

    s1_test = s1_idx
    s2_test = s2_idx
    s1_ref = jnp.asarray(bond_tab[b, 0], dtype=jnp.int32)

    s1_idx = 0
    s2_idx = 0

    s1_upd_idx = s1_idx
    s1_dwd_idx = s1_idx + spin_n
    s1_dwo_idx = (spin_n - 1 - s1_idx) + 2 * spin_n
    s1_upo_idx = (spin_n - 1 - s1_idx) + 3 * spin_n

    s2_upd_idx = s2_idx
    s2_dwd_idx = s2_idx + spin_n
    s2_dwo_idx = (spin_n - 1 - s2_idx) + 2 * spin_n
    s2_upo_idx = (spin_n - 1 - s2_idx) + 3 * spin_n

    idx_ij_x = bond_tab[b, 4]
    idx_ij_y = bond_tab[b, 5]
    rij_x = bond_tab[b, 6]
    rij_y = bond_tab[b, 7]

    kr = k1 * idx_ij_x + k2 * idx_ij_y
    # kr = 0

    Q1 = jnp.pi/3
    Q2 = jnp.pi/jnp.sqrt(3)

    q_val = Q1 * rij_x + Q2 * rij_y
    qr = (q_val)
    bond_cond_0 = (b==1) | (b==3)
    bond_cond_1 = (b==12) | (b==16)
    bond_cond_2 = (b==11) | (b==10)
    bond_cond = bond_cond_0 | bond_cond_1 | bond_cond_2
    qr = jnp.where(bond_cond, qr - jnp.pi, qr)
    # qr = jnp.where(bond_cond, jnp.pi, 0)
    # jax.debug.print("bond={b}, kr={kr}, qr={qr}", b=b, kr=kr, qr=qr,ordered=True)
    # jax.debug.print("bond={b}, idx_ij=({idx_x}, {idx_y})", b=b, idx_x=idx_ij_x, idx_y=idx_ij_y, ordered=True)

    bond_type = jnp.asarray(bond_tab[b, 8], dtype=jnp.int32)
    B_val = B_tab[bond_type]
    B_val = B_val / 3
    A_val = A_tab[bond_type]
    A_val = A_val / 3
    # A_val = q_pi * A_val

    bond_type = jnp.asarray(bond_tab[b, 8], dtype=jnp.int32)

    bond_cond_0 = (b==0) | (b==3) | (b==2)
    bond_cond_1 = (b==14) | (b==12) | (b==15)
    bond_cond_2 = (b==9) | (b==11) | (b==7)

    bond_cond = bond_cond_0 | bond_cond_1 | bond_cond_2
    B_val = jnp.where(bond_cond, B_val, B_val)
    A_val = jnp.where(bond_cond, A_val, -A_val)
    # jax.debug.print("===================================",ordered=True)
    # B_val = jnp.where(bond_cond, 1/3, 1/3)
    # A_val = jnp.where(bond_cond, 1j/3, -1j/3)

    # bond_cond = (b==0) | (b==1) | (b==2) | (b==3) | (b==4) | (b==5)
    # # bond_cond = (b==0) |(b==4)
    # B_val = jnp.where(bond_cond, B_val, 0)
    # A_val = jnp.where(bond_cond, A_val, 0)

    # jax.debug.print("B_val={B}, A_val={A}", B=B_val, A=A_val,ordered=True)
    # b_check = (b == 1) | (b == 12)
    # b_check = (b==1)
    # B_val = jnp.where(b_check, B_val, 0)
    # A_val = jnp.where(b_check, A_val, 0)

    kk=kr+qr
    Bo_iou_jdu_real = jnp.cos(kk)
    Bo_iou_jdu_imag = jnp.sin(kk)
    sidx0 = s1_upo_idx
    sidx1 = s2_upd_idx
    value_to_add = Bo_iou_jdu_real + 1j * Bo_iou_jdu_imag
    value_to_add = 0.5 * B_val * value_to_add
    H_cur = H_cur.at[:, sidx0, (mat_dim -1 - sidx1)].add(value_to_add)

    kk=kr-qr
    Bo_iod_jdd_real = jnp.cos(kk)
    Bo_iod_jdd_imag = jnp.sin(kk)
    sidx0 = s1_dwo_idx
    sidx1 = s2_dwd_idx
    value_to_add = Bo_iod_jdd_real + 1j * Bo_iod_jdd_imag
    value_to_add = 0.5 * B_val * value_to_add
    H_cur = H_cur.at[:, sidx0, (mat_dim -1 - sidx1)].add(value_to_add)

    kk=-(-kr+qr)
    Bd_idu_jou_real = jnp.cos(kk)
    Bd_idu_jou_imag = jnp.sin(kk)
    sidx0 = s1_upd_idx
    sidx1 = s2_upo_idx
    value_to_add = Bd_idu_jou_real + 1j * Bd_idu_jou_imag
    value_to_add = 0.5 * B_val * value_to_add
    H_cur = H_cur.at[:, sidx0, (mat_dim -1 - sidx1)].add(value_to_add)

    kk=-(-kr-qr)
    Bd_idd_jod_real = jnp.cos(kk)
    Bd_idd_jod_imag = jnp.sin(kk)
    sidx0 = s1_dwd_idx
    sidx1 = s2_dwo_idx
    value_to_add = Bd_idd_jod_real + 1j * Bd_idd_jod_imag
    value_to_add = 0.5 * B_val * value_to_add
    H_cur = H_cur.at[:, sidx0, (mat_dim -1 - sidx1)].add(value_to_add)

    kk=kr+qr
    Ao_iou_jod_real = jnp.cos(kk)
    Ao_iou_jod_imag = jnp.sin(kk)
    sidx0 = s1_upo_idx
    sidx1 = s2_dwo_idx
    value_to_add = Ao_iou_jod_real + 1j * Ao_iou_jod_imag
    value_to_add = 0.5 * A_val * value_to_add
    H_cur = H_cur.at[:, sidx0, (mat_dim -1 - sidx1)].add(value_to_add)

    kk=kr-qr
    Ao_iod_jou_real = jnp.cos(kk)
    Ao_iod_jou_imag = jnp.sin(kk)
    sidx0 = s1_dwo_idx
    sidx1 = s2_upo_idx
    value_to_add = Ao_iod_jou_real + 1j * Ao_iod_jou_imag
    value_to_add = -0.5 * A_val * value_to_add
    H_cur = H_cur.at[:, sidx0, (mat_dim -1 - sidx1)].add(value_to_add)
        
    kk =-(kr+qr)
    Ad_idu_jdd_real = jnp.cos(kk)
    Ad_idu_jdd_imag = jnp.sin(kk)
    sidx0 = s2_dwd_idx
    sidx1 = s1_upd_idx
    value_to_add = Ad_idu_jdd_real + 1j * Ad_idu_jdd_imag
    value_to_add = -0.5 * A_val * value_to_add
    H_cur = H_cur.at[:, sidx0, (mat_dim -1 - sidx1)].add(value_to_add)

    kk =-(kr-qr)
    Ad_idd_jdu_real = jnp.cos(kk)
    Ad_idd_jdu_imag = jnp.sin(kk)
    sidx0 = s2_upd_idx
    sidx1 = s1_dwd_idx
    value_to_add = Ad_idd_jdu_real + 1j * Ad_idd_jdu_imag
    value_to_add = 0.5 * A_val * value_to_add
    H_cur = H_cur.at[:, sidx0, (mat_dim -1 - sidx1)].add(value_to_add)
        
    return H_cur

def compute_diag_term(i, H_cur, diag_up, diag_dw, hx_term, spin_n, mat_dim):
    idx_upd = i
    idx_dwd = i + spin_n
    idx_dwo = (spin_n - 1 - i) + 2 * spin_n
    idx_upo = (spin_n - 1 - i) + 3 * spin_n

    # jax.debug.print("--------------------------------------")
    # # jax.debug.print("添加对角项: lambda={}", lambda_param)
    # # jax.debug.print("idx_up={}, idx_dw={}", diag_up, diag_dw)
    # # jax.debug.print("idx_upd={}, idx_dwd={}, idx_dwo={}, idx_upo={}", idx_upd, idx_dwd, idx_dwo, idx_upo)
    # jax.debug.print("hx_term={}", hx_term)
    # jax.debug.print("--------------------------------------")
    # 添加对角项
    H_cur = H_cur.at[:, idx_upd, (mat_dim -1 - idx_upo)].add(diag_up)
    H_cur = H_cur.at[:, idx_dwd, (mat_dim -1 - idx_dwo)].add(diag_dw)
    H_cur = H_cur.at[:, idx_upo, (mat_dim -1 - idx_upd)].add(diag_up)
    H_cur = H_cur.at[:, idx_dwo, (mat_dim -1 - idx_dwd)].add(diag_dw)
    # 添加对称破缺场项
    H_cur = H_cur.at[:, idx_upd, (mat_dim -1 - idx_dwo)].add(hx_term)
    H_cur = H_cur.at[:, idx_dwd, (mat_dim -1 - idx_upo)].add(hx_term)
    H_cur = H_cur.at[:, idx_upo, (mat_dim -1 - idx_dwd)].add(hx_term)
    H_cur = H_cur.at[:, idx_dwo, (mat_dim -1 - idx_upd)].add(hx_term)

    return H_cur


