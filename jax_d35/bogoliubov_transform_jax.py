"""
Bogoliubov变换模块 - JAX版本（支持GPU加速）

Author: ZhouChk
JAX优化: 批量矩阵运算，JIT编译，GPU/TPU加速
"""

import jax
import jax.numpy as jnp
from typing import Tuple
from functools import partial
from Hamiltonian_jax import Ham_jax

# 启用64位精度以匹配NumPy
jax.config.update("jax_enable_x64", True)

@partial(jax.jit, static_argnums=(1,))
def bogoliubov_single_k(H_k: jnp.ndarray, spin_n: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    对单个k点进行Bogoliubov变换 - JAX版本
    
    完全遵循NumPy版本的实现逻辑
    
    Parameters:
    -----------
    H_k : jnp.ndarray
        单个k点的哈密顿量矩阵 (4*spin_n, 4*spin_n)
        
    Returns:
    --------
    Ubov : jnp.ndarray
        Bogoliubov变换矩阵 (4*spin_n, 4*spin_n)
    ek : jnp.ndarray
        能谱 (4*spin_n,)
    """
    # 度规矩阵
    mat_dim = 4 * spin_n

    g = jnp.eye(mat_dim, dtype=jnp.float64)

    def make_g(n, carry):
        # jax.debug.print("设置度规矩阵 g 的元素 g[{n},{n}] = -1", n=n, ordered=True)
        return carry.at[n, n].set(-1)
    g = jax.lax.fori_loop(2*spin_n, 4*spin_n, make_g, g)
    
    # Cholesky分解 (upper triangular, 与NumPy版本一致)
    entest=jnp.linalg.eigvals(H_k)
    # jax.debug.print("输入矩阵H_k的特征值: {entest}", entest=entest)
    
    r_lower = jnp.linalg.cholesky(H_k)  # JAX默认返回下三角
    r = r_lower.T.conj()  # 转置得到上三角
    
    # 构造变换后的矩阵，并做一次数值厄米化
    ht = r @ g @ r.T.conj()
    # ht = 0.5 * (ht + ht.T.conj())

    # 对厄米矩阵使用eigh，返回正交归一特征向量
    enk_vals, ut = jnp.linalg.eigh(ht)
    sort_idx = jnp.argsort(enk_vals.real)[::-1]
    sort_enk = enk_vals.real[sort_idx]
    
    un = ut[:, sort_idx]

    def compute_eigenval(v):
        return v.conj().T @ ht @ v

    # jax.debug.print("排序后的特征值: {sort_enk}", sort_enk=sort_enk,ordered=True)
    ekk_diag = jax.vmap(compute_eigenval, in_axes=1)(un)
    # jax.debug.print("逐列验证特征值 v^H H v: {ekk_diag}", ekk_diag=ekk_diag, ordered=True)
    # jax.debug.print("验证误差 (v^H H v - sort_enk): {delta}", delta=(ekk_diag.real - sort_enk), ordered=True)

    # 计算最终的Bogoliubov矩阵
    ekk = jnp.diag(ekk_diag)
    ekk_sq = jnp.diag(jnp.sqrt(jnp.diag(g @ ekk)))  # sqrt(g @ ekk)
    r_inv = jnp.linalg.inv(r)
    Ubov = r_inv @ un @ ekk_sq
    
    # 提取最终能谱
    ek = jnp.array([g[j, j] * ekk_diag[j] for j in range(mat_dim)]).real
    
    return Ubov, ek

@partial(jax.jit, static_argnums=(15, 16))
def Bogoliubov_transform_jax_batch(omega: float, k1: jnp.ndarray, k2: jnp.ndarray,
                                    A1: complex, A2: complex, A3: complex, 
                                    B1: float, B2: float, B3: float, 
                                    lambda_param: float, h: float,
                                    J1plus: float, J2plus: float, J3plus: float,
                                    bond_tab: jnp.ndarray, spin_n: int, bond_n: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    批量Bogoliubov变换 - JAX版本（完全向量化，GPU加速）
    
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
    Ubov : jnp.ndarray
        Bogoliubov变换矩阵 (nk, 4, 4)
    ek : jnp.ndarray
        能谱 (nk, 4)
    """
    # 构建哈密顿量（批量）
    H = Ham_jax(omega, k1, k2, A1, A2, A3, B1, B2, B3, 
                lambda_param, h, J1plus, J2plus, J3plus, bond_tab, spin_n, bond_n)
    
    # spin_n 参与矩阵维度构造，必须作为静态参数显式传入单点变换。
    Ubov, ek = jax.vmap(lambda H_k: bogoliubov_single_k(H_k, spin_n))(H)
    
    return Ubov, ek

def Bogoliubov_transform_2_jax(omega: float, k1: jnp.ndarray, k2: jnp.ndarray,
                               A1: complex, A2: complex, A3: complex, 
                               B1: float, B2: float, B3: float, 
                               lambda_param: float, h: float,
                               J1plus: float, J2plus: float, J3plus: float,
                               bond_tab: jnp.ndarray, spin_n: int, bond_n: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Bogoliubov变换 - JAX版本（包装函数，兼容原接口）
    
    Returns:
    --------
    tuple
        (Ubov, ek) - Bogoliubov矩阵，能谱
    """
    # 确保输入是JAX数组
    k1 = jnp.atleast_1d(k1)
    k2 = jnp.atleast_1d(k2)
    
    # 执行批量计算
    spin_n = 1
    Ubov, ek = Bogoliubov_transform_jax_batch(
        omega, k1, k2, A1, A2, A3, B1, B2, B3, 
        lambda_param, h, J1plus, J2plus, J3plus, bond_tab, spin_n, bond_n
    )
    
    return Ubov, ek


@partial(jax.jit, static_argnums=(15, 16))
def Bogoliubov_constraint_jax_batch(omega: float, k1: jnp.ndarray, k2: jnp.ndarray, 
                                    A1: complex, A2: complex, A3: complex, 
                                    B1: float, B2: float, B3: float, 
                                    lambda_param: float, h: float,
                                    J1plus: float, J2plus: float, J3plus: float, 
                                    bond_tab: jnp.ndarray, spin_n: int, bond_n: int) -> jnp.ndarray:
    
    # 构建哈密顿量（批量）
    H = Ham_jax(omega, k1, k2, A1, A2, A3, B1, B2, B3, 
                lambda_param, h, J1plus, J2plus, J3plus, bond_tab, spin_n, bond_n)
    # jax.debug.print("lambda_param: {}", lambda_param, ordered=True)
    # jax.debug.print("H shape: {}", H.shape, ordered=True)
    # jax.debug.print("H[0]:\n{}", H[0], ordered=True)

    # 使用vmap进行向量化处理所有k点
    min_eng = jax.vmap(get_min_energy_jax)(H)
    
    return min_eng

def Bogoliubov_constraint_jax(omega: float, k1: jnp.ndarray, k2: jnp.ndarray,
                               A1: complex, A2: complex, A3: complex, 
                               B1: float, B2: float, B3: float, 
                               lambda_param: float, h: float,
                               J1plus: float, J2plus: float, J3plus: float,
                               bond_tab: jnp.ndarray, spin_n: int, bond_n: int) -> float:
    """
    Bogoliubov变换 - JAX版本（包装函数，兼容原接口）
    
    Returns:
    --------
    float
        所有 k 点最小本征值中的最小值
    """
    # 确保输入是JAX数组
    k1 = jnp.atleast_1d(k1)
    k2 = jnp.atleast_1d(k2)
    
    # 执行批量计算
    min_eng = Bogoliubov_constraint_jax_batch(
        omega, k1, k2, A1, A2, A3, B1, B2, B3, 
        lambda_param, h, J1plus, J2plus, J3plus, bond_tab, spin_n, bond_n
    )

    min_energy = jnp.min(min_eng)
    
    return min_energy

@partial(jax.jit, static_argnums=())
def get_min_energy_jax(H_k: jnp.ndarray) -> jnp.ndarray:
    
    # 计算特征值和特征向量 (使用eig以匹配NumPy版本)
    enk_vals, _ = jnp.linalg.eig(H_k)
    min_energies = jnp.min(enk_vals.real)
    # jax.debug.print("min_energies: {}", min_energies, ordered=True)

    return min_energies


@partial(jax.jit, static_argnums=(7, 8))
def saddle_point_sum_jax(Ubov: jnp.ndarray, k1: jnp.ndarray, k2: jnp.ndarray,
                         J1plus: float, J2plus: float, J3plus: float,
                         bond_tab: jnp.ndarray, spin_n: int, bond_n: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    计算鞍点求和 - JAX版本（完全向量化，与NumPy版本一致）
    
    Parameters:
    -----------
    Ubov : jnp.ndarray
        Bogoliubov变换矩阵 (nk, 4, 4)
    k1, k2 : jnp.ndarray
        动量分量 (nk,)
    Q1, Q2 : float
        磁序波矢
        
    Returns:
    --------
    tuple
        (lambda, AA, BB, Usum) - 各种求和结果
    """
    spin_n = 1
    mat_dim = 4 * spin_n # 哈密顿量矩阵的维度
    nk = Ubov.shape[0]
    bond_n_eff = min(bond_n, bond_tab.shape[0])
    Jplus = jnp.array([J1plus, J2plus, J3plus], dtype=jnp.float64)
    #Jplus = jnp.array([J1plus, 0, 0], dtype=jnp.float64)


    def compute_single_k_contribution(ut, k1_i, k2_i):
        """计算单个k点的贡献"""
        cc_init = jnp.zeros((mat_dim, mat_dim), dtype=jnp.complex128)
        ss_init = jnp.zeros((mat_dim, mat_dim), dtype=jnp.complex128)
        lam_template = jnp.zeros((mat_dim, mat_dim), dtype=jnp.complex128)

        diag_up = 1
        diag_dw = 1
        def compute_diag_contribution(i, lam_cur):
            idx_upd = i
            idx_dwd = i + spin_n
            idx_dwo = (spin_n - 1 - i) + 2 * spin_n
            idx_upo = (spin_n - 1 - i) + 3 * spin_n

            # jax.debug.print("--------------------------------------",ordered=True)
            # jax.debug.print("diag_up={}", diag_up,ordered=True)
            # jax.debug.print("diag_dw={}", diag_dw,ordered=True)
            # jax.debug.print("idx_1={},idx_2={}", idx_upd, (mat_dim -1 - idx_upo),ordered=True)
            # jax.debug.print("idx_1={},idx_2={}", idx_dwd, (mat_dim -1 - idx_dwo),ordered=True)
            # jax.debug.print("idx_1={},idx_2={}", idx_upo, (mat_dim -1 - idx_upd),ordered=True)
            # jax.debug.print("idx_1={},idx_2={}", idx_dwo, (mat_dim -1 - idx_dwd),ordered=True)
            # jax.debug.print("--------------------------------------",ordered=True)
            # 添加对角项
            lam_cur = lam_cur.at[idx_upd, (mat_dim -1 - idx_upo)].add(diag_up)
            lam_cur = lam_cur.at[idx_dwd, (mat_dim -1 - idx_dwo)].add(diag_dw)
            lam_cur = lam_cur.at[idx_upo, (mat_dim -1 - idx_upd)].add(diag_up)
            lam_cur = lam_cur.at[idx_dwo, (mat_dim -1 - idx_dwd)].add(diag_dw)
            
            return lam_cur
        
        lam_template = jax.lax.fori_loop(0, spin_n, compute_diag_contribution, lam_template)

        def compute_single_bond_contribution(b, carry):
            cc_cur, ss_cur = carry

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
            
            kr = k1_i * idx_ij_x + k2_i * idx_ij_y
            
            Q1 = 2 * jnp.pi/3
            Q2 = 2 * jnp.pi/jnp.sqrt(3)
            
            q_val = Q1 * rij_x + Q2 * rij_y
            bond_add_pi_0 = (b == 1) | (b == 3)
            bond_add_pi_1 = (b == 10) | (b == 11)
            bond_add_pi_2 = (b == 16) | (b == 12)
            bond_add_pi = bond_add_pi_0 | bond_add_pi_1 | bond_add_pi_2
            q_pi = jnp.where(bond_add_pi, -1, 1)
            qr = - (q_val) /2
            
            bond_type = jnp.asarray(bond_tab[b, 8], dtype=jnp.int32)

            just_test_0 = (bond_type == 0) | (bond_type == 4) 
            just_test_1 = (bond_type == 14) | (bond_type == 17)
            just_test_2 = (bond_type == 9) | (bond_type == 6)
            just_test = just_test_0 | just_test_1 | just_test_2

            B_val = jnp.where(just_test, 1, 0)
            B_val = B_val * q_pi / 3
            A_val = jnp.where(just_test, 1, 0)
            A_val = A_val * q_pi / 3
            
            bond_type = jnp.asarray(bond_tab[b, 8], dtype=jnp.int32)
            
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
            Bo_iou_jdu_imag = 0
            sidx0 = jnp.where(bond_cond, s1_upo_idx, s1_upo_idx)
            sidx1 = jnp.where(bond_cond, s2_upd_idx, s2_upd_idx)
            value_to_add = Bo_iou_jdu_real + 1j * Bo_iou_jdu_imag
            value_to_add = B_val * value_to_add
            cc_cur = cc_cur.at[sidx0, (mat_dim -1 - sidx1)].add(value_to_add)
            
            Bo_iod_jdd_real = 0.5 * jnp.cos(kr - qr)
            Bo_iod_jdd_imag = -0.5 * jnp.sin(kr - qr)
            Bo_iod_jdd_imag = 0
            sidx0 = jnp.where(bond_cond, s1_dwo_idx, s1_dwo_idx)
            sidx1 = jnp.where(bond_cond, s2_dwd_idx, s2_dwd_idx)
            value_to_add = Bo_iod_jdd_real + 1j * Bo_iod_jdd_imag
            value_to_add = B_val * value_to_add
            cc_cur = cc_cur.at[sidx0, (mat_dim -1 - sidx1)].add(value_to_add)
            
            Bd_idu_jou_real = 0.5 * jnp.cos(-kr + qr)
            Bd_idu_jou_imag = 0.5 * jnp.sin(-kr + qr)
            Bd_idu_jou_imag = 0
            sidx0 = jnp.where(bond_cond, s1_upd_idx, s1_upd_idx)
            sidx1 = jnp.where(bond_cond, s2_upo_idx, s2_upo_idx)
            value_to_add = Bd_idu_jou_real + 1j * Bd_idu_jou_imag
            value_to_add = B_val * value_to_add
            cc_cur = cc_cur.at[sidx0, (mat_dim -1 - sidx1)].add(value_to_add)
            
            Bd_idd_jod_real = 0.5 * jnp.cos(-kr - qr)
            Bd_idd_jod_imag = 0.5 * jnp.sin(-kr - qr)
            Bd_idd_jod_imag = 0
            sidx0 = jnp.where(bond_cond, s1_dwd_idx, s1_dwd_idx)
            sidx1 = jnp.where(bond_cond, s2_dwo_idx, s2_dwo_idx)
            value_to_add = Bd_idd_jod_real + 1j * Bd_idd_jod_imag
            value_to_add = B_val * value_to_add
            cc_cur = cc_cur.at[sidx0, (mat_dim -1 - sidx1)].add(value_to_add)
            
            Ao_iou_jod_real = 0.5 * bond_fac * jnp.cos(- sig0 * kr - qr)
            Ao_iou_jod_imag = 0.5 * bond_fac * jnp.sin(- sig0 * kr - qr)
            Ao_iou_jod_real = 0
            sidx0 = jnp.where(sig0 == 1, s1_upo_idx, s2_dwo_idx)
            sidx1 = jnp.where(sig0 == 1, s2_dwo_idx, s1_upo_idx)
            value_to_add = Ao_iou_jod_real + 1j * Ao_iou_jod_imag
            value_to_add = A_val * value_to_add
            ss_cur = ss_cur.at[sidx0, (mat_dim -1 - sidx1)].add(value_to_add)
            
            Ao_iod_jou_real = - 0.5 * bond_fac * jnp.cos(- sig1 * kr + qr)
            Ao_iod_jou_imag = - 0.5 * bond_fac * jnp.sin(- sig1 * kr + qr)
            Ao_iod_jou_real = 0
            sidx0 = jnp.where(sig1 == 1, s1_dwo_idx, s2_upo_idx)
            sidx1 = jnp.where(sig1 == 1, s2_upo_idx, s1_dwo_idx)
            value_to_add = Ao_iod_jou_real + 1j * Ao_iod_jou_imag
            value_to_add = A_val * value_to_add
            ss_cur = ss_cur.at[sidx0, (mat_dim -1 - sidx1)].add(value_to_add)
            
            Ad_idu_jdd_real = 0.5 * bond_fac * jnp.cos(sig0 * kr - qr)
            Ad_idu_jdd_imag = 0.5 * bond_fac * jnp.sin(sig0 * kr - qr)
            Ad_idu_jdd_real = 0
            sidx0 = jnp.where(sig0 == 1, s1_upd_idx, s2_dwd_idx)
            sidx1 = jnp.where(sig0 == 1, s2_dwd_idx, s1_upd_idx)
            value_to_add = Ad_idu_jdd_real + 1j * Ad_idu_jdd_imag
            value_to_add = A_val * value_to_add
            ss_cur = ss_cur.at[sidx0, (mat_dim -1 - sidx1)].add(value_to_add)
            
            Ad_idd_jdu_real = - 0.5 * bond_fac * jnp.cos(sig1 * kr + qr)
            Ad_idd_jdu_imag = - 0.5 * bond_fac * jnp.sin(sig1 * kr + qr)
            Ad_idd_jdu_real = 0
            sidx0 = jnp.where(sig1 == 1, s2_dwd_idx, s1_upd_idx)
            sidx1 = jnp.where(sig1 == 1, s1_upd_idx, s2_dwd_idx)
            value_to_add = Ad_idd_jdu_real + 1j * Ad_idd_jdu_imag
            value_to_add = A_val * value_to_add
            ss_cur = ss_cur.at[sidx0, (mat_dim -1 - sidx1)].add(value_to_add)
            
            return cc_cur, ss_cur

        cc_cur, ss_cur = jax.lax.fori_loop(0, bond_n_eff, compute_single_bond_contribution, (cc_init, ss_init))
        # jax.debug.print("lam_template:\n{}", lam_template, ordered=True)
        # jax.debug.print("ss_cur:\n{}", ss_cur, ordered=True)
        # jax.debug.print("cc_cur:\n{}", cc_cur, ordered=True)

        # 计算 ut @ ut^† 
        ut_uth = ut @ jnp.conj(ut).T

        # 计算三个组合矩阵
        combined_0 = ut_uth @ lam_template - lam_template
        combined_1 = ut_uth @ cc_cur
        combined_2 = ut_uth @ ss_cur
        # jax.debug.print("lam_template: {}", lam_template, ordered=True)
        # jax.debug.print("cc_cur: {}", cc_cur, ordered=True)
        # jax.debug.print("ss_cur: {}", ss_cur, ordered=True)
        # 返回shape (3, 4, 4)
        return jnp.stack([combined_0, combined_1, combined_2], axis=0)

    # 向量化计算所有k点
    all_contributions = jax.vmap(compute_single_k_contribution)(Ubov, k1, k2)
    
    # 对所有k点求和并归一化 (sum over k-points, result shape: (3, 4, 4))
    Usum = jnp.sum(all_contributions, axis=0) / nk
    
    # 计算各个量
    lam = jnp.trace(Usum[0, :, :])*0.25
    AA = jnp.trace(Usum[2, :, :])*0.125
    BB = jnp.trace(Usum[1, :, :])*0.125
    
    return jnp.real(lam), AA, BB, Usum
