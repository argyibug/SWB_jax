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
    g = g.at[2*n_spin:, 2*n_spin:].set(-1)
    jax.debug.print("test g:\n{}", g)
    
    # Cholesky分解 (upper triangular, 与NumPy版本一致)
    r_lower = jnp.linalg.cholesky(H_k)  # JAX默认返回下三角
    r = r_lower.T.conj()  # 转置得到上三角
    
    # 构造变换后的矩阵
    ht = r @ g @ r.T.conj()

    jax.debug.print(":test ht:\n{}", ht-ht.T.conj())
    
    # 计算特征值和特征向量，并按 sort_enk 的顺序同步重排特征向量列
    enk_vals, ut = jnp.linalg.eig(ht)
    jax.debug.print("原始特征值 enk_vals:\n{}", jnp.round(enk_vals, 2))
    sort_idx = jnp.argsort(enk_vals.real)[::-1]
    sort_enk = enk_vals.real[sort_idx]
    jax.debug.print("sort_enk: {}", jnp.round(sort_enk, 2))
    
    jax.debug.print("sort_idx:\n{}", sort_idx)
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

    # jax.debug.print("test Un:\n{}", jnp.round(un[0:2,:], 4))
    # jax.debug.print("test Ut:\n{}", jnp.round(ut[0:2,:], 4))
    # jax.debug.print("test g:\n{}", jnp.round(jnp.linalg.inv(ut) @ ht @ ut, 2))
    # jax.debug.print("test g:\n{}", jnp.round(jnp.linalg.inv(un) @ ht @ un, 2))

    jax.debug.print("test ekk_diag:\n{}", jnp.round(Ubov.conj().T @ H_k @ Ubov, 4))
    
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


@partial(jax.jit, static_argnums=())
def saddle_point_sum_jax(Ubov: jnp.ndarray, k1: jnp.ndarray, k2: jnp.ndarray, 
                         Q1: float, Q2: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
    nk = Ubov.shape[0]
    
    def compute_single_k_contribution(ut, k1_i):
        """计算单个k点的贡献"""
        # 构建三角函数矩阵 cc
        cc = jnp.diag(jnp.array([
            jnp.cos(k1_i + Q1/2),
            jnp.cos(k1_i + Q1/2),
            jnp.cos(-k1_i + Q1/2),
            jnp.cos(-k1_i + Q1/2)
        ]))
        
        # 构建三角函数矩阵 ss (与NumPy版本完全一致)
        ss_diag_upper = jnp.array([
            [0.0, jnp.sin(k1_i + Q1/2), 0.0, 0.0],
            [jnp.sin(k1_i + Q1/2), 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, jnp.sin(-k1_i + Q1/2)],
            [0.0, 0.0, jnp.sin(-k1_i + Q1/2), 0.0]
        ])
        
        # 计算 ut @ ut^† 
        ut_uth = ut @ jnp.conj(ut).T
        
        # 计算三个组合矩阵
        combined_0 = ut_uth @ jnp.eye(4) - jnp.eye(4)
        combined_1 = ut_uth @ cc
        combined_2 = ut_uth @ ss_diag_upper  # ss矩阵
        
        # 返回shape (3, 4, 4)
        return jnp.stack([combined_0, combined_1, combined_2], axis=0)
    
    # 向量化计算所有k点
    all_contributions = jax.vmap(compute_single_k_contribution)(Ubov, k1)
    
    # 对所有k点求和并归一化 (sum over k-points, result shape: (3, 4, 4))
    Usum = jnp.sum(all_contributions, axis=0) / nk
    
    # 计算各个量
    lam = jnp.trace(Usum[0, :, :]) / 4
    AA = 1j * jnp.trace(Usum[2, :, :]) / 8
    BB = jnp.trace(Usum[1, :, :]) / 8
    
    return jnp.real(lam), AA, BB, Usum
