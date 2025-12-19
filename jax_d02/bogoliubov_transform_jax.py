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

@partial(jax.jit, static_argnums=())
def bogoliubov_single_k(H_k: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    对单个k点进行Bogoliubov变换 - JAX版本
    
    完全遵循NumPy版本的实现逻辑
    
    Parameters:
    -----------
    H_k : jnp.ndarray
        单个k点的哈密顿量矩阵 (4, 4)
        
    Returns:
    --------
    Ubov : jnp.ndarray
        Bogoliubov变换矩阵 (4, 4)
    ek : jnp.ndarray
        能谱 (4,)
    """
    # 度规矩阵
    g = jnp.eye(4)
    g = g.at[1, 1].set(-1)
    g = g.at[3, 3].set(-1)
    
    # Cholesky分解 (upper triangular, 与NumPy版本一致)
    r_lower = jnp.linalg.cholesky(H_k)  # JAX默认返回下三角
    r = r_lower.T.conj()  # 转置得到上三角
    
    # 构造变换后的矩阵
    ht = r @ g @ r.T.conj()
    
    # 计算特征值和特征向量 (使用eig以匹配NumPy版本)
    enk_vals, ut = jnp.linalg.eig(ht)
    sort_enk = jnp.sort(enk_vals.real)  # 按实部排序
    
    # 重新排序特征向量 - 按照NumPy版本的idxtab逻辑
    idxtab = jnp.array([3, 1, 2, 0])
    
    # 计算所有特征向量对应的特征值 (用于匹配)
    # eigenval_test[j] = ut[:, j]^† @ ht @ ut[:, j]
    def compute_eigenval(v):
        return v.conj().T @ ht @ v
    
    eigenval_test = jax.vmap(compute_eigenval, in_axes=1)(ut)
    
    # 使用贪心匹配: 依次为每个目标特征值找到最佳特征向量
    # 使用jax.lax.fori_loop实现高效循环
    def match_step(k, carry):
        """单步匹配函数:为第k个目标特征值找到最佳特征向量"""
        used, un, ekk_diag = carry
        
        # 获取目标特征值
        idx = idxtab[k]
        target_val = sort_enk[idx]
        
        # 计算距离(已使用的特征向量设为无穷大)
        diff = jnp.abs(eigenval_test.real - target_val)
        diff_masked = jnp.where(used, jnp.inf, diff)
        
        # 找到最佳匹配
        best_j = jnp.argmin(diff_masked)
        
        # 更新状态
        used_new = used.at[best_j].set(True)
        un_new = un.at[:, k].set(ut[:, best_j])
        ekk_diag_new = ekk_diag.at[k].set(target_val)
        
        return (used_new, un_new, ekk_diag_new)
    
    # 初始状态
    init_used = jnp.zeros(4, dtype=bool)
    init_un = jnp.zeros((4, 4), dtype=jnp.complex128)
    init_ekk_diag = jnp.zeros(4, dtype=jnp.float64)
    
    # 执行4次迭代(k=0,1,2,3)
    _, un, ekk_diag = jax.lax.fori_loop(
        0, 4, match_step, (init_used, init_un, init_ekk_diag)
    )
    
    # 计算最终的Bogoliubov矩阵
    ekk = jnp.diag(ekk_diag)
    ekk_sq = jnp.diag(jnp.sqrt(jnp.diag(g @ ekk)))  # sqrt(g @ ekk)
    r_inv = jnp.linalg.inv(r)
    Ubov = r_inv @ un @ ekk_sq
    
    # 提取最终能谱
    ek = jnp.array([g[j, j] * ekk_diag[j] for j in range(4)]).real
    
    return Ubov, ek

@partial(jax.jit, static_argnums=())
def Bogoliubov_transform_jax_batch(omega: float, k1: jnp.ndarray, k2: jnp.ndarray, 
                                    Q1: float, Q2: float, 
                                    A1: complex, A2: complex, A3: complex, 
                                    B1: float, B2: float, B3: float, 
                                    lambda_param: float, h: float,
                                    J1plus: float, J2plus: float, J3plus: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
                lambda_param, h, J1plus, J2plus, J3plus)
    
    # 使用vmap进行向量化处理所有k点
    Ubov, ek = jax.vmap(bogoliubov_single_k)(H)
    
    return Ubov, ek

def Bogoliubov_transform_2_jax(omega: float, k1: jnp.ndarray, k2: jnp.ndarray, 
                               Q1: float, Q2: float, 
                               A1: complex, A2: complex, A3: complex, 
                               B1: float, B2: float, B3: float, 
                               lambda_param: float, h: float,
                               J1plus: float, J2plus: float, J3plus: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
        lambda_param, h, J1plus, J2plus, J3plus
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
                                    J1plus: float, J2plus: float, J3plus: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
                lambda_param, h, J1plus, J2plus, J3plus)
    
    # 使用vmap进行向量化处理所有k点
    Ubov, ek = jax.vmap(bogoliubov_single_k)(H)
    
    return Ubov, ek

def Bogoliubov_constraint_jax(omega: float, k1: jnp.ndarray, k2: jnp.ndarray, 
                               Q1: float, Q2: float, 
                               A1: complex, A2: complex, A3: complex, 
                               B1: float, B2: float, B3: float, 
                               lambda_param: float, h: float,
                               J1plus: float, J2plus: float, J3plus: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
        lambda_param, h, J1plus, J2plus, J3plus
    )
    
    # 重新计算H（如果需要）
    H = Ham_jax(omega, k1, k2, Q1, Q2, A1, A2, A3, B1, B2, B3, 
                lambda_param, h, J1plus, J2plus, J3plus)
    
    return Ubov, ek, H


@partial(jax.jit, static_argnums=())
def Bogoliubov_constraint_jax_batch(omega: float, k1: jnp.ndarray, k2: jnp.ndarray, 
                                    Q1: float, Q2: float, 
                                    A1: complex, A2: complex, A3: complex, 
                                    B1: float, B2: float, B3: float, 
                                    lambda_param: float, h: float,
                                    J1plus: float, J2plus: float, J3plus: float) -> Tuple[jnp.ndarray]:
    
    # 构建哈密顿量（批量）
    H = Ham_jax(omega, k1, k2, Q1, Q2, A1, A2, A3, B1, B2, B3, 
                lambda_param, h, J1plus, J2plus, J3plus)
    
    # 使用vmap进行向量化处理所有k点
    min_eng = jax.vmap(get_min_energy_jax)(H)
    
    return min_eng

def Bogoliubov_constraint_jax(omega: float, k1: jnp.ndarray, k2: jnp.ndarray, 
                               Q1: float, Q2: float, 
                               A1: complex, A2: complex, A3: complex, 
                               B1: float, B2: float, B3: float, 
                               lambda_param: float, h: float,
                               J1plus: float, J2plus: float, J3plus: float) -> float:
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
        lambda_param, h, J1plus, J2plus, J3plus
    )

    min_energy = jnp.min(min_eng)
    
    # 重新计算H（如果需要）
    # H = Ham_jax(omega, k1, k2, Q1, Q2, A1, A2, A3, B1, B2, B3, 
    #             lambda_param, h, J1plus, J2plus, J3plus)
    
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
