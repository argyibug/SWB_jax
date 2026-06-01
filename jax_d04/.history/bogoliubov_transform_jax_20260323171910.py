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
def bogoliubov_single_k(H_k: jnp.ndarray, n_spin: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    对单个k点进行Bogoliubov变换 - JAX版本
    
    完全遵循NumPy版本的实现逻辑
    
    Parameters:
    -----------
    H_k : jnp.ndarray
        单个k点的哈密顿量矩阵 (4*n_spin, 4*n_spin)
        
    Returns:
    --------
    Ubov : jnp.ndarray
        Bogoliubov变换矩阵 (4*n_spin, 4*n_spin)
    ek : jnp.ndarray
        能谱 (4*n_spin,)
    """
    # 度规矩阵
    mat_dim = 4 * n_spin

    g = jnp.eye(mat_dim, dtype=jnp.float64)

    def make_g(n, carry):
        return carry.at[n, n].set(-1)
    g = jax.lax.fori_loop(2*n_spin, 4*n_spin, make_g, g)
    
    # Cholesky分解 (upper triangular, 与NumPy版本一致)
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

    # 计算重排后每个特征向量对应的特征值
    def compute_eigenval(v):
        return v.conj().T @ ht @ v
    
    ekk_diag = jax.vmap(compute_eigenval, in_axes=1)(un)

    # 计算最终的Bogoliubov矩阵
    ekk = jnp.diag(ekk_diag)
    ekk_sq = jnp.diag(jnp.sqrt(jnp.diag(g @ ekk)))  # sqrt(g @ ekk)
    r_inv = jnp.linalg.inv(r)
    Ubov = r_inv @ un @ ekk_sq
    
    # 提取最终能谱
    ek = jnp.array([g[j, j] * ekk_diag[j] for j in range(mat_dim)]).real
    
    return Ubov, ek

@partial(jax.jit, static_argnums=())
def Bogoliubov_transform_jax_batch(omega: float, k1: jnp.ndarray, k2: jnp.ndarray, 
                                    Q1: float, Q2: float, 
                                    A1: complex, A2: complex, A3: complex, 
                                    B1: float, B2: float, B3: float, 
                                    lambda_param: float, h: float,
                                    J1plus: float, J2plus: float, J3plus: float,
                                    bond_tab: jnp.ndarray, n_spin: int, n_bond: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    批量Bogoliubov变换 - JAX版本（完全向量化，GPU加速）
    
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
    Ubov : jnp.ndarray
        Bogoliubov变换矩阵 (nk, 4, 4)
    ek : jnp.ndarray
        能谱 (nk, 4)
    """
    # 构建哈密顿量（批量）
    H = Ham_jax(omega, k1, k2, Q1, Q2, A1, A2, A3, B1, B2, B3, 
                lambda_param, h, J1plus, J2plus, J3plus, bond_tab, n_spin, n_bond)
    
    # 使用vmap进行向量化处理所有k点
    Ubov, ek = jax.vmap(bogoliubov_single_k)(H)
    
    return Ubov, ek

def Bogoliubov_transform_2_jax(omega: float, k1: jnp.ndarray, k2: jnp.ndarray, 
                               Q1: float, Q2: float, 
                               A1: complex, A2: complex, A3: complex, 
                               B1: float, B2: float, B3: float, 
                               lambda_param: float, h: float,
                               J1plus: float, J2plus: float, J3plus: float,
                               bond_tab: jnp.ndarray, n_spin: int, n_bond: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Bogoliubov变换 - JAX版本（包装函数，兼容原接口）
    
    Returns:
    --------
    tuple
        (Ubov, ek, H) - Bogoliubov矩阵，能谱，哈密顿量
    """
    # 确保输入是JAX数组
    k1 = jnp.atleast_1d(k1)
    k2 = jnp.atleast_1d(k2)
    
    # 执行批量计算
    Ubov, ek = Bogoliubov_transform_jax_batch(
        omega, k1, k2, Q1, Q2, A1, A2, A3, B1, B2, B3, 
        lambda_param, h, J1plus, J2plus, J3plus, bond_tab, n_spin, n_bond
    )
    
    # 重新计算H（如果需要）
    # H = Ham_jax(omega, k1, k2, Q1, Q2, A1, A2, A3, B1, B2, B3, 
    #             lambda_param, h, J1plus, J2plus, J3plus)
    
    return Ubov, ek


@partial(jax.jit, static_argnums=())
def Bogoliubov_constraint_jax_batch(omega: float, k1: jnp.ndarray, k2: jnp.ndarray, 
                                    Q1: float, Q2: float, 
                                    A1: complex, A2: complex, A3: complex, 
                                    B1: float, B2: float, B3: float, 
                                    lambda_param: float, h: float,
                                    J1plus: float, J2plus: float, J3plus: float,
                                    bond_tab: jnp.ndarray, n_spin: int, n_bond: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    批量Bogoliubov变换 - JAX版本（完全向量化，GPU加速）
    
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
    Ubov : jnp.ndarray
        Bogoliubov变换矩阵 (nk, 4, 4)
    ek : jnp.ndarray
        能谱 (nk, 4)
    """
    # 构建哈密顿量（批量）
    H = Ham_jax(omega, k1, k2, Q1, Q2, A1, A2, A3, B1, B2, B3, 
                lambda_param, h, J1plus, J2plus, J3plus, bond_tab, n_spin, n_bond)
    
    # 使用vmap进行向量化处理所有k点
    Ubov, ek = jax.vmap(bogoliubov_single_k)(H)
    
    return Ubov, ek

def Bogoliubov_constraint_jax(omega: float, k1: jnp.ndarray, k2: jnp.ndarray, 
                               Q1: float, Q2: float, 
                               A1: complex, A2: complex, A3: complex, 
                               B1: float, B2: float, B3: float, 
                               lambda_param: float, h: float,
                               J1plus: float, J2plus: float, J3plus: float,
                               bond_tab: jnp.ndarray, n_spin: int, n_bond: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Bogoliubov变换 - JAX版本（包装函数，兼容原接口）
    
    Returns:
    --------
    tuple
        (Ubov, ek, H) - Bogoliubov矩阵，能谱，哈密顿量
    """
    # 确保输入是JAX数组
    k1 = jnp.atleast_1d(k1)
    k2 = jnp.atleast_1d(k2)
    
    # 执行批量计算
    Ubov, ek = Bogoliubov_constraint_jax_batch(
        omega, k1, k2, Q1, Q2, A1, A2, A3, B1, B2, B3, 
        lambda_param, h, J1plus, J2plus, J3plus, bond_tab, n_spin, n_bond
    )
    
    # 重新计算H（如果需要）
    H = Ham_jax(omega, k1, k2, Q1, Q2, A1, A2, A3, B1, B2, B3, 
                lambda_param, h, J1plus, J2plus, J3plus, bond_tab, n_spin, n_bond)
    
    return Ubov, ek, H


@partial(jax.jit, static_argnums=())
def Bogoliubov_constraint_jax_batch(omega: float, k1: jnp.ndarray, k2: jnp.ndarray, 
                                    Q1: float, Q2: float, 
                                    A1: complex, A2: complex, A3: complex, 
                                    B1: float, B2: float, B3: float, 
                                    lambda_param: float, h: float,
                                    J1plus: float, J2plus: float, J3plus: float, 
                                    bond_tab: jnp.ndarray, n_spin: int, n_bond: int) -> Tuple[jnp.ndarray]:
    
    # 构建哈密顿量（批量）
    H = Ham_jax(omega, k1, k2, Q1, Q2, A1, A2, A3, B1, B2, B3, 
                lambda_param, h, J1plus, J2plus, J3plus, bond_tab, n_spin, n_bond)
    
    # 使用vmap进行向量化处理所有k点
    min_eng = jax.vmap(get_min_energy_jax)(H)
    
    return min_eng

def Bogoliubov_constraint_jax(omega: float, k1: jnp.ndarray, k2: jnp.ndarray, 
                               Q1: float, Q2: float, 
                               A1: complex, A2: complex, A3: complex, 
                               B1: float, B2: float, B3: float, 
                               lambda_param: float, h: float,
                               J1plus: float, J2plus: float, J3plus: float,
                               bond_tab: jnp.ndarray, n_spin: int, n_bond: int) -> float:
    """
    Bogoliubov变换 - JAX版本（包装函数，兼容原接口）
    
    Returns:
    --------
    tuple
        (Ubov, ek, H) - Bogoliubov矩阵，能谱，哈密顿量
    """
    # 确保输入是JAX数组
    k1 = jnp.atleast_1d(k1)
    k2 = jnp.atleast_1d(k2)
    
    # 执行批量计算
    min_eng = Bogoliubov_constraint_jax_batch(
        omega, k1, k2, Q1, Q2, A1, A2, A3, B1, B2, B3, 
        lambda_param, h, J1plus, J2plus, J3plus, bond_tab, n_spin, n_bond
    )

    min_energy = jnp.min(min_eng)
    
    # 重新计算H（如果需要）
    # H = Ham_jax(omega, k1, k2, Q1, Q2, A1, A2, A3, B1, B2, B3, 
    #             lambda_param, h, J1plus, J2plus, J3plus, bond_tab, n_spin, n_bond)
    
    return min_energy

@partial(jax.jit, static_argnums=())
def get_min_energy_jax(H_k: jnp.ndarray) -> Tuple[jnp.ndarray]:
    
    # 计算特征值和特征向量 (使用eig以匹配NumPy版本)
    enk_vals, _ = jnp.linalg.eig(H_k)
    min_energies = jnp.min(enk_vals.real)

    return min_energies


@partial(jax.jit, static_argnums=(7, 8))
def saddle_point_sum_jax(Ubov: jnp.ndarray, k1: jnp.ndarray, k2: jnp.ndarray,
                         J1plus: float, J2plus: float, J3plus: float,
                         bond_tab: jnp.ndarray, n_spin: int, n_bond: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
    mat_dim = 4 * n_spin # 哈密顿量矩阵的维度
    nk = Ubov.shape[0]
    n_bond_eff = min(n_bond, bond_tab.shape[0])
    has_bond_type = bond_tab.shape[1] > 6
    Jplus = jnp.array([J1plus, J2plus, J3plus], dtype=jnp.float64)

    idx = jnp.arange(n_spin, dtype=jnp.int32)
    idx_upd = idx
    idx_dwd = idx + n_spin
    idx_dwo = (n_spin - 1 - idx) + 2 * n_spin
    idx_upo = (n_spin - 1 - idx) + 3 * n_spin

    lam_template = jnp.zeros((mat_dim, mat_dim), dtype=jnp.complex128)
    lam_template = lam_template.at[idx_upd, mat_dim - 1 - idx_upo].set(1.0)
    lam_template = lam_template.at[idx_dwd, mat_dim - 1 - idx_dwo].set(1.0)
    lam_template = lam_template.at[idx_upo, mat_dim - 1 - idx_upd].set(1.0)
    lam_template = lam_template.at[idx_dwo, mat_dim - 1 - idx_dwd].set(1.0)

    def compute_single_k_contribution(ut, k1_i, k2_i):
        """计算单个k点的贡献"""
        cc_init = jnp.zeros((mat_dim, mat_dim), dtype=jnp.complex128)
        ss_init = jnp.zeros((mat_dim, mat_dim), dtype=jnp.complex128)

        def compute_single_bond_contribution(b, carry):
            cc_cur, ss_cur = carry

            s1_idx = jnp.asarray(bond_tab[b, 2], dtype=jnp.int32)
            s2_idx = jnp.asarray(bond_tab[b, 3], dtype=jnp.int32)

            s1_upd_idx = s1_idx
            s1_dwd_idx = s1_idx + n_spin
            s1_dwo_idx = (n_spin - 1 - s1_idx) + 2 * n_spin
            s1_upo_idx = (n_spin - 1 - s1_idx) + 3 * n_spin

            s2_upd_idx = s2_idx
            s2_dwd_idx = s2_idx + n_spin
            s2_dwo_idx = (n_spin - 1 - s2_idx) + 2 * n_spin
            s2_upo_idx = (n_spin - 1 - s2_idx) + 3 * n_spin

            rij_x = bond_tab[b, 4]
            rij_y = bond_tab[b, 5]
            kr = k1_i * rij_x + k2_i * rij_y

            if has_bond_type:
                bond_type = jnp.asarray(bond_tab[b, 6], dtype=jnp.int32)
            else:
                bond_type = jnp.mod(b, 3)

            cos_term = 0.5 * Jplus[bond_type] * jnp.cos(kr)
            sin_term = 0.5 * Jplus[bond_type] * jnp.sin(kr)

            cc_cur = cc_cur.at[s1_upo_idx, mat_dim - 1 - s2_upd_idx].add(cos_term)
            cc_cur = cc_cur.at[s1_dwo_idx, mat_dim - 1 - s2_dwd_idx].add(cos_term)
            cc_cur = cc_cur.at[s1_upd_idx, mat_dim - 1 - s2_upo_idx].add(cos_term)
            cc_cur = cc_cur.at[s1_dwd_idx, mat_dim - 1 - s2_dwo_idx].add(cos_term)

            ss_cur = ss_cur.at[s1_upo_idx, mat_dim - 1 - s2_dwo_idx].add(-sin_term)
            ss_cur = ss_cur.at[s1_dwo_idx, mat_dim - 1 - s2_upo_idx].add(sin_term)
            ss_cur = ss_cur.at[s1_upd_idx, mat_dim - 1 - s2_dwd_idx].add(sin_term)
            ss_cur = ss_cur.at[s1_dwd_idx, mat_dim - 1 - s2_upd_idx].add(-sin_term)

            return cc_cur, ss_cur

        cc_cur, ss_cur = jax.lax.fori_loop(0, n_bond_eff, compute_single_bond_contribution, (cc_init, ss_init))
        # 计算 ut @ ut^† 
        ut_uth = ut @ jnp.conj(ut).T
        
        # 计算三个组合矩阵
        combined_0 = ut_uth @ lam_template - lam_template
        combined_1 = ut_uth @ cc_cur
        # combined_2 = ut_uth @ ss_cur
        combined_2 = ss_cur
        
        # 返回shape (3, 4, 4)
        return jnp.stack([combined_0, combined_1, combined_2], axis=0)

    # 向量化计算所有k点
    all_contributions = jax.vmap(compute_single_k_contribution)(Ubov, k1, k2)
    
    # 对所有k点求和并归一化 (sum over k-points, result shape: (3, 4, 4))
    Usum = jnp.sum(all_contributions, axis=0) / nk
    
    # 计算各个量
    lam = jnp.trace(Usum[0, :, :])
    AA = jnp.trace(Usum[2, :, :])
    BB = jnp.trace(Usum[1, :, :])
    
    return jnp.real(lam), AA, BB, Usum
