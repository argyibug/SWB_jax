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

    def bond_body(b, H_cur):
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
        # jax.debug.print("===================================")
        # jax.debug.print(
        #     "idx1 upd={upd} dwd={dwd} dwo={dwo} upo={upo}",
        #     upd=s1_upd_idx,
        #     dwd=s1_dwd_idx,
        #     dwo=s1_dwo_idx,
        #     upo=s1_upo_idx,
        # )
        # jax.debug.print(
        #     "idx2 upd={upd} dwd={dwd} dwo={dwo} upo={upo}",
        #     upd=s2_upd_idx,
        #     dwd=s2_dwd_idx,
        #     dwo=s2_dwo_idx,
        #     upo=s2_upo_idx,
        # )
        #jax.debug.print("bond_tab={}", bond_tab[b, 4:7])
        # jax.debug.print("===================================")
        # jax.debug.print(
        #     "Processing bond {b}: s1={s1}, s2={s2}, idx_ij=({rx}, {ry})",
        #     b=b,
        #     s1=s1_idx,
        #     s2=s2_idx,
        #     rx=idx_ij_x,
        #     ry=idx_ij_y,
        # )
        kr = k1 * idx_ij_x + k2 * idx_ij_y
        # kr = 0

        Q1 = 2 * jnp.pi/3
        Q2 = 2 * jnp.pi/jnp.sqrt(3)

        q_val = Q1 * rij_x + Q2 * rij_y
        bond_add_pi_0 = (b == 1) | (b == 3)
        bond_add_pi_1 = (b == 10) | (b == 11)
        bond_add_pi_2 = (b == 16) | (b == 12)
        bond_add_pi = bond_add_pi_0 | bond_add_pi_1 | bond_add_pi_2
        q_pi = jnp.where(bond_add_pi, -1, 1)
        qr = - (q_val) /2
        # qr = 0
        # jax.debug.print("bond={b}, rij=({rx}, {ry}), qr={qr}", b=b, rx=rij_x, ry=rij_y, qr=qr)

        bond_type = jnp.asarray(bond_tab[b, 8], dtype=jnp.int32)
        B_val = B[bond_type]
        B_val = B_val * q_pi / 3
        A_val = A[bond_type]
        A_val = A_val * q_pi / 3
        # jax.debug.print("B_val={B}, A_val={A}", B=B, A=A)
        # B_val = 1 * q_pi / 3
        # A_val = 1j * q_pi / 3
        # A_val = q_pi * A_val

        bond_type = jnp.asarray(bond_tab[b, 8], dtype=jnp.int32)
        
        # jax.debug.print("===================================",ordered=True)
        # jax.debug.print("bond={b}, q_pi={q_pi}, q_val={q_val}", b=b, q_pi=q_pi, q_val=q_val, ordered=True)
        
        bond_cond_0 = (s1_test == 0)
        bond_cond_1 = (s1_test == 1)
        bond_cond_2 = (s1_test == 2)

        bond_cond = bond_cond_0 | bond_cond_1 | bond_cond_2

        bond_cond_type_0 = (b == 0) | (b == 2) | (b == 1)
        bond_cond_type_1 = (b == 9) | (b == 10) | (b == 7)
        bond_cond_type_2 = (b == 14) | (b == 16) | (b == 15)

        bond_cond_type_01 = (s1_test == 0) & (s2_test == 1)
        bond_cond_type_12 = (s1_test == 1) & (s2_test == 2)
        bond_cond_type_20 = (s1_test == 2) & (s2_test == 0)

        bond_cond_type_10 = (s1_test == 1) & (s2_test == 0)
        bond_cond_type_21 = (s1_test == 2) & (s2_test == 1)
        bond_cond_type_02 = (s1_test == 0) & (s2_test == 2)

        bond_cond_type = bond_cond_type_01 | bond_cond_type_12 | bond_cond_type_20
        bond_fac = jnp.where(bond_cond_type, 1, -1)

        bond_cond_type = bond_cond_type_0 | bond_cond_type_1 | bond_cond_type_2
        sig0=jnp.where(bond_cond_type, 1, -1)
        sig1=jnp.where(bond_cond_type, -1, 1)

        Bo_iou_jdu_real = 0.5 * jnp.cos(kr + qr)
        Bo_iou_jdu_imag = -0.5 * jnp.sin(kr + qr)
        Bo_iod_jdd_imag = 0
        sidx0 = jnp.where(bond_cond, s1_upo_idx, s1_upo_idx)
        sidx1 = jnp.where(bond_cond, s2_upd_idx, s2_upd_idx)
        value_to_add = Bo_iou_jdu_real + 1j * Bo_iou_jdu_imag
        value_to_add = B_val * value_to_add
        # jax.debug.print("value_to_add={v}, B_val={B}", v=value_to_add, B=B_val, ordered=True)
        H_cur = H_cur.at[:, sidx0, (mat_dim -1 - sidx1)].add(value_to_add)
        # H_cur = H_cur.at[:, s1_upo_idx, (mat_dim -1 - s2_upd_idx)].add(Bo_iou_jdu_real + 1j * Bo_iou_jdu_imag)

        Bo_iod_jdd_real = 0.5 * jnp.cos(kr - qr)
        Bo_iod_jdd_imag = -0.5 * jnp.sin(kr - qr)
        Bo_iod_jdd_imag = 0
        sidx0 = jnp.where(bond_cond, s1_dwo_idx, s1_dwo_idx)
        sidx1 = jnp.where(bond_cond, s2_dwd_idx, s2_dwd_idx)
        value_to_add = Bo_iod_jdd_real + 1j * Bo_iod_jdd_imag
        value_to_add = B_val * value_to_add
        H_cur = H_cur.at[:, sidx0, (mat_dim -1 - sidx1)].add(value_to_add)
        # H_cur = H_cur.at[:, s1_dwo_idx, (mat_dim -1 - s2_dwd_idx)].add(Bo_iod_jdd_real + 1j * Bo_iod_jdd_imag)

        Bd_idu_jou_real = 0.5 * jnp.cos(-kr + qr)
        Bd_idu_jou_imag = 0.5 * jnp.sin(-kr + qr)
        Bd_idu_jou_imag = 0
        sidx0 = jnp.where(bond_cond, s1_upd_idx, s1_upd_idx)
        sidx1 = jnp.where(bond_cond, s2_upo_idx, s2_upo_idx)
        value_to_add = Bd_idu_jou_real + 1j * Bd_idu_jou_imag
        value_to_add = B_val * value_to_add
        H_cur = H_cur.at[:, sidx0, (mat_dim -1 - sidx1)].add(value_to_add)
        # H_cur = H_cur.at[:, s1_upd_idx, (mat_dim -1 - s2_upo_idx)].add(Bd_idu_jou_real + 1j * Bd_idu_jou_imag)
        
        Bd_idd_jod_real = 0.5 * jnp.cos(-kr - qr)
        Bd_idd_jod_imag = 0.5 * jnp.sin(-kr - qr)
        Bd_idd_jod_imag = 0
        sidx0 = jnp.where(bond_cond, s1_dwd_idx, s1_dwd_idx)
        sidx1 = jnp.where(bond_cond, s2_dwo_idx, s2_dwo_idx)
        value_to_add = Bd_idd_jod_real + 1j * Bd_idd_jod_imag
        value_to_add = B_val * value_to_add
        H_cur = H_cur.at[:, sidx0, (mat_dim -1 - sidx1)].add(value_to_add)
        # H_cur = H_cur.at[:, s1_dwd_idx, (mat_dim -1 - s2_dwo_idx)].add(Bd_idd_jod_real + 1j * Bd_idd_jod_imag)


        Ao_iou_jod_real = 0.5 * bond_fac * jnp.cos(- sig0 * kr - qr)
        Ao_iou_jod_imag = 0.5 * bond_fac * jnp.sin(- sig0 * kr - qr)
        Ao_iou_jod_real = 0
        sidx0 = jnp.where(sig0 == 1, s1_upo_idx, s2_dwo_idx)
        sidx1 = jnp.where(sig0 == 1, s2_dwo_idx, s1_upo_idx)
        value_to_add = Ao_iou_jod_real + 1j * Ao_iou_jod_imag
        value_to_add = A_val * value_to_add
        # jax.debug.print("sidx0={s0}, sidx1={s1}", s0=sidx0, s1=sidx1, ordered=True)
        # jax.debug.print("value_to_add={v}, A_val={A}", v=value_to_add * test_check, A=A_val, ordered=True)
        # jax.debug.print("1-1, fact={fact}, bond={b}, sig={sig0}, sidx0={s0}, sidx1={s1}", b=b, fact=demo_fac, sig0=sig0, s0=sidx0, s1=sidx1)
        H_cur = H_cur.at[:, sidx0, (mat_dim -1 - sidx1)].add(value_to_add)
        # H_cur = H_cur.at[:, s1_upo_idx, (mat_dim -1 - s2_dwo_idx)].add(Ao_iou_jod_real + 1j * Ao_iou_jod_imag)
        
        Ao_iod_jou_real = - 0.5 * bond_fac * jnp.cos(- sig1 * kr + qr)
        Ao_iod_jou_imag = - 0.5 * bond_fac * jnp.sin(- sig1 * kr + qr)
        Ao_iod_jou_real = 0
        sidx0 = jnp.where(sig1 == 1, s1_dwo_idx, s2_upo_idx)
        sidx1 = jnp.where(sig1 == 1, s2_upo_idx, s1_dwo_idx)
        value_to_add = Ao_iod_jou_real + 1j * Ao_iod_jou_imag
        value_to_add = A_val * value_to_add
        # jax.debug.print("sidx0={s0}, sidx1={s1}", s0=sidx0, s1=sidx1, ordered=True)
        # jax.debug.print("value_to_add={v}, A_val={A}", v=value_to_add * test_check, A=A_val, ordered=True)
        # jax.debug.print("1-2, fact={fact}, bond={b}, sig={sig1} sidx0={s0}, sidx1={s1}", b=b, fact=demo_fac, sig1=sig1, s0=sidx0, s1=sidx1)
        H_cur = H_cur.at[:, sidx0, (mat_dim -1 - sidx1)].add(value_to_add)
        # H_cur = H_cur.at[:, s2_dwo_idx, (mat_dim -1 - s1_upo_idx)].add(Ao_iod_jou_real + 1j * Ao_iod_jou_imag)


        Ad_idu_jdd_real = 0.5 * bond_fac * jnp.cos(sig0 * kr - qr)
        Ad_idu_jdd_imag = 0.5 * bond_fac * jnp.sin(sig0 * kr - qr)
        Ad_idu_jdd_real = 0
        sidx0 = jnp.where(sig0 == 1, s1_upd_idx, s2_dwd_idx)
        sidx1 = jnp.where(sig0 == 1, s2_dwd_idx, s1_upd_idx)
        value_to_add = Ad_idu_jdd_real + 1j * Ad_idu_jdd_imag
        value_to_add = A_val * value_to_add
        # jax.debug.print("sidx0={s0}, sidx1={s1}", s0=sidx0, s1=sidx1, ordered=True)
        # jax.debug.print("value_to_add={v}, A_val={A}", v=value_to_add * test_check, A=A_val, ordered=True)
        H_cur = H_cur.at[:, sidx0, (mat_dim -1 - sidx1)].add(value_to_add)
        # H_cur = H_cur.at[:, s1_upd_idx, (mat_dim -1 - s2_dwd_idx)].add(Ad_idu_jdd_real + 1j * Ad_idu_jdd_imag)
        
        Ad_idd_jdu_real = - 0.5 * bond_fac * jnp.cos(sig1 * kr + qr)
        Ad_idd_jdu_imag = - 0.5 * bond_fac * jnp.sin(sig1 * kr + qr)
        Ad_idd_jdu_real = 0
        sidx0 = jnp.where(sig1 == 1, s2_dwd_idx, s1_upd_idx)
        sidx1 = jnp.where(sig1 == 1, s1_upd_idx, s2_dwd_idx)
        value_to_add = Ad_idd_jdu_real + 1j * Ad_idd_jdu_imag
        value_to_add = A_val * value_to_add
        # jax.debug.print("sidx0={s0}, sidx1={s1}", s0=sidx0, s1=sidx1, ordered=True)
        # jax.debug.print("value_to_add={v}, A_val={A}", v=value_to_add * test_check, A=A_val, ordered=True)
        H_cur = H_cur.at[:, sidx0, (mat_dim -1 - sidx1)].add(value_to_add)
        # H_cur = H_cur.at[:, s2_dwd_idx, (mat_dim -1 - s1_upd_idx)].add(Ad_idd_jdu_real + 1j * Ad_idd_jdu_imag)

        # Bo_iou_jdu_real = 0.5 * jnp.cos(kr + qr) * Jplus[bond_type] * B[bond_type]
        # Bo_iou_jdu_imag = -0.5 * jnp.sin(kr + qr) * Jplus[bond_type] * B[bond_type]
        # Bo_iod_jdd_real = 0.5 * jnp.cos(kr - qr) * Jplus[bond_type] * B[bond_type]
        # Bo_iod_jdd_imag = -0.5 * jnp.sin(kr - qr) * Jplus[bond_type] * B[bond_type]

        # Bd_idu_jou_real = 0.5 * jnp.cos(-kr + qr) * Jplus[bond_type] * B[bond_type]
        # Bd_idu_jou_imag = 0.5 * jnp.sin(-kr + qr) * Jplus[bond_type] * B[bond_type]
        # Bd_idd_jod_real = 0.5 * jnp.cos(-kr - qr) * Jplus[bond_type] * B[bond_type]
        # Bd_idd_jod_imag = 0.5 * jnp.sin(-kr - qr) * Jplus[bond_type] * B[bond_type]

        # Ao_iou_jod_real = 0.5 * jnp.cos(kr + qr) * Jplus[bond_type] * A[bond_type]
        # Ao_iou_jod_imag = -0.5 * jnp.sin(kr + qr) * Jplus[bond_type] * A[bond_type]
        # Ao_iod_jou_real = -0.5 * jnp.cos(-kr - qr) * Jplus[bond_type] * A[bond_type]
        # Ao_iod_jou_imag = 0.5 * jnp.sin(-kr - qr) * Jplus[bond_type] * A[bond_type]

        # Ad_idu_jdd_real = 0.5 * jnp.cos(kr + qr) * Jplus[bond_type] * A[bond_type]
        # Ad_idu_jdd_imag = 0.5 * jnp.sin(kr + qr) * Jplus[bond_type] * A[bond_type]
        # Ad_idd_jdu_real = -0.5 * jnp.cos(-kr - qr) * Jplus[bond_type] * A[bond_type]
        # Ad_idd_jdu_imag = -0.5 * jnp.sin(-kr - qr) * Jplus[bond_type] * A[bond_type]

        # H_cur = H_cur.at[:, s1_upo_idx, (mat_dim -1 - s2_upd_idx)].add(Bo_iou_jdu_real + 1j * Bo_iou_jdu_imag)
        # H_cur = H_cur.at[:, s1_dwo_idx, (mat_dim -1 - s2_dwd_idx)].add(Bo_iod_jdd_real + 1j * Bo_iod_jdd_imag)

        # H_cur = H_cur.at[:, s1_upd_idx, (mat_dim -1 - s2_upo_idx)].add(Bd_idu_jou_real + 1j * Bd_idu_jou_imag)
        # H_cur = H_cur.at[:, s1_dwd_idx, (mat_dim -1 - s2_dwo_idx)].add(Bd_idd_jod_real + 1j * Bd_idd_jod_imag)

        # H_cur = H_cur.at[:, s1_upo_idx, (mat_dim -1 - s2_dwo_idx)].add(Ao_iou_jod_real + 1j * Ao_iou_jod_imag)
        # # H_cur = H_cur.at[:, s1_dwo_idx, (mat_dim -1 - s2_upo_idx)].add(Ao_iod_jou_real + 1j * Ao_iod_jou_imag)
        # H_cur = H_cur.at[:, s2_dwo_idx, (mat_dim -1 - s1_upo_idx)].add(Ao_iod_jou_real + 1j * Ao_iod_jou_imag)

        # H_cur = H_cur.at[:, s1_upd_idx, (mat_dim -1 - s2_dwd_idx)].add(Ad_idu_jdd_real + 1j * Ad_idu_jdd_imag)
        # # H_cur = H_cur.at[:, s1_dwd_idx, (mat_dim -1 - s2_upd_idx)].add(Ad_idd_jdu_real + 1j * Ad_idd_jdu_imag)
        # H_cur = H_cur.at[:, s2_dwd_idx, (mat_dim -1 - s1_upd_idx)].add(Ad_idd_jdu_real + 1j * Ad_idd_jdu_imag)


        ################################################################################################

        # bond_cond = (b == 1) | (b == 10) | (b == 16)
        # B_temp = jnp.where(bond_cond, -B[bond_type], B[bond_type])

        # cos_term_pkpq = Jplus[bond_type] * B_temp * jnp.cos(kr+qr)
        # cos_term_pknq = Jplus[bond_type] * B_temp * jnp.cos(kr-qr)
        # cos_term_nkpq = Jplus[bond_type] * B_temp * jnp.cos(-kr+qr)
        # cos_term_nknq = Jplus[bond_type] * B_temp * jnp.cos(-kr-qr)
        # sin_term_pkpq = (-1j) * Jplus[bond_type] * A[bond_type] * jnp.sin(kr+qr)
        # sin_term_pknq = (-1j) * Jplus[bond_type] * A[bond_type] * jnp.sin(kr-qr)
        # sin_term_nkpq = (-1j) * Jplus[bond_type] * A[bond_type] * jnp.sin(-kr+qr)
        # sin_term_nknq = (-1j) * Jplus[bond_type] * A[bond_type] * jnp.sin(-kr-qr)
        
        # bond_cond_0 = (b == 0) | (b == 2) | (b == 1)
        # bond_cond_1 = (b == 9) | (b == 7) | (b == 10)
        # bond_cond_2 = (b == 14) | (b == 15) | (b == 16)

        # bond_cond = bond_cond_0 | bond_cond_1 | bond_cond_2
        
        # cos_term_pkpq = jnp.where(bond_cond, cos_term_pkpq/3, jnp.zeros_like(cos_term_pkpq))
        # cos_term_pknq = jnp.where(bond_cond, cos_term_pknq/3, jnp.zeros_like(cos_term_pknq))
        # cos_term_nkpq = jnp.where(bond_cond, cos_term_nkpq/3, jnp.zeros_like(cos_term_nkpq))
        # cos_term_nknq = jnp.where(bond_cond, cos_term_nknq/3, jnp.zeros_like(cos_term_nknq))

        # sin_term_pkpq = jnp.where(bond_cond, sin_term_pkpq/3, jnp.zeros_like(sin_term_pkpq))
        # sin_term_pknq = jnp.where(bond_cond, sin_term_pknq/3, jnp.zeros_like(sin_term_pknq))
        # sin_term_nkpq = jnp.where(bond_cond, sin_term_nkpq/3, jnp.zeros_like(sin_term_nkpq))
        # sin_term_nknq = jnp.where(bond_cond, sin_term_nknq/3, jnp.zeros_like(sin_term_nknq))


        # H_cur = H_cur.at[:, s1_upo_idx, (mat_dim -1 - s2_upd_idx)].add(cos_term_pkpq)
        # H_cur = H_cur.at[:, s1_dwo_idx, (mat_dim -1 - s2_dwd_idx)].add(cos_term_pknq)
        # H_cur = H_cur.at[:, s1_upd_idx, (mat_dim -1 - s2_upo_idx)].add(cos_term_nkpq)
        # H_cur = H_cur.at[:, s1_dwd_idx, (mat_dim -1 - s2_dwo_idx)].add(cos_term_nknq)
        
        # H_cur = H_cur.at[:, s1_upo_idx, (mat_dim -1 - s2_dwo_idx)].add(sin_term_pkpq)
        # H_cur = H_cur.at[:, s1_dwo_idx, (mat_dim -1 - s2_upo_idx)].add(-sin_term_pknq)
        # H_cur = H_cur.at[:, s2_upd_idx, (mat_dim -1 - s1_dwd_idx)].add(-sin_term_pknq)
        # H_cur = H_cur.at[:, s2_dwd_idx, (mat_dim -1 - s1_upd_idx)].add(sin_term_pkpq)
        ################################################################################################
        
        return H_cur
        


    H = jax.lax.fori_loop(0, n_bond_eff, bond_body, H)

    # jax.debug.print("H={}", H[10, :, :])


    # Diagonal terms
    idx = jnp.arange(spin_n, dtype=jnp.int32)
    # jax.debug.print("计算对角项索引: idx={}", idx)

    # hx_term = 0
    # diag_up = 0
    # diag_dw = 0
    def compute_diag_term(i, H_cur):
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
    
    H = jax.lax.fori_loop(0, spin_n, compute_diag_term, H)
    
    return H


