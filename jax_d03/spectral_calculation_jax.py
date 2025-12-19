"""
光谱计算模块 - JAX版本（支持GPU加速）

Author: ZhouChk
JAX优化: 完全向量化，消除Python循环，GPU并行计算
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple
from functools import partial
from bogoliubov_transform_jax import Bogoliubov_transform_jax_batch

def calculate_spectral_single_omega(omega: float, kpath: jnp.ndarray, eta: float,
                                    channel: jnp.ndarray,
                                    A1: complex, A2: complex, A3: complex,
                                    B1: float, B2: float, B3: float,
                                    lambda_param: float, h: float,
                                    k1_all: jnp.ndarray, k2_all: jnp.ndarray,
                                    Q1: float, Q2: float,
                                    J1plus: float, J2plus: float, J3plus: float,
                                    U_k: jnp.ndarray, eng_k: jnp.ndarray,
                                    beta: float = 100) -> jnp.ndarray:
    """    
    Parameters:
    -----------
    omega : float
        频率
    kpath : jnp.ndarray
        k路径，形状 (kpath_len, 2)
    eta : float
        展宽参数
    channel : jnp.ndarray
        通道矩阵 (3, 3)
    A1, A2, A3 : complex
        鞍点参数A
    B1, B2, B3 : float
        鞍点参数B
    lambda_param : float
        拉格朗日乘数
    h : float
        对称破缺场
    k1_all, k2_all : jnp.ndarray
        全布里渊区动量网格
    Q1, Q2 : float
        磁序波矢
    J1plus, J2plus, J3plus : float
        交换耦合参数
        
    Returns:
    --------
    jnp.ndarray
        光谱强度 (kpath_len,)
    """
    
    # 定义自旋算符
    uz = 0.5 * jnp.array([[1, 0, 0, 0],
                          [0, -1, 0, 0],
                          [0, 0, -1, 0],
                          [0, 0, 0, 1]], dtype=jnp.complex128)
    
    ux = 0.5 * jnp.array([[0, 0, 1, 0],
                          [0, 0, 0, 1],
                          [1, 0, 0, 0],
                          [0, 1, 0, 0]], dtype=jnp.complex128)
    
    uy = 0.5 * jnp.array([[0, 0, 1j, 0],
                          [0, 0, 0, 1j],
                          [-1j, 0, 0, 0],
                          [0, -1j, 0, 0]], dtype=jnp.complex128)
    
    u = jnp.stack([ux, uy, uz], axis=0)

    
    order_vec = jnp.array([[[Q1, Q2],[0, 0],[0, 0]],
                          [[0, 0],[Q1, Q2],[0, 0]],
                          [[0, 0],[0, 0],[0, 0]]], dtype=jnp.complex128)
    
    # 度规矩阵
    g = jnp.eye(4)
    g = g.at[1, 1].set(-1)
    g = g.at[3, 3].set(-1)
        
    Nsites = k1_all.shape[0]
    kpath_len = kpath.shape[0]
    spectrum = jnp.zeros(kpath_len, dtype=jnp.complex128)
    
    # JIT 编译的核心计算函数（单个通道和单个k点）
    @partial(jax.jit, static_argnums=(2, 3))
    def compute_channel_kpoint(kx, ky, mu_idx, nu_idx):
        """计算单个通道(mu, nu)在单个k点的贡献"""
        omega_modified = 0

        kq1_all = k1_all + kx + order_vec[mu_idx, nu_idx, 0]
        kq2_all = k2_all + ky + order_vec[mu_idx, nu_idx, 1]

        # 计算k+q点的Bogoliubov变换
        U_kq, eng_kq = Bogoliubov_transform_jax_batch(
            omega_modified, kq1_all, kq2_all, Q1, Q2,
            A1, A2, A3, B1, B2, B3, lambda_param, h,
            J1plus, J2plus, J3plus
        )
        
        se_kq = eng_kq @ g  # (Nsites, 4)
        se_k = eng_k @ g    # (Nsites, 4)
        
        # 对单个idx计算
        def compute_for_idx(idx):
            uu_kq = U_kq[idx, :, :]  # (4, 4)
            uu_k = U_k[idx, :, :]
            uu_kq_d = uu_kq.conj().T
            uu_k_d = uu_k.conj().T
            
            se_kq_idx = se_kq[idx, :]  # (4,)
            se_k_idx = se_k[idx, :]
            
            # 对 m in range(15) 进行向量化
            def compute_for_m(m):
                x_kq = m % 4
                x_k = (m // 4) % 4
                
                # 构建 T 矩阵
                T_kq = jnp.outer(uu_kq[:, x_kq], uu_kq_d[x_kq, :]) * g[x_kq, x_kq]
                T_k = jnp.outer(uu_k[:, x_k], uu_k_d[x_k, :]) * g[x_k, x_k]
                
                # 计算 f 和 f1
                nb_kq = 1.0 / (jnp.exp(beta * se_kq_idx[x_kq]) - 1.0)
                nb_k = 1.0 / (jnp.exp(beta * se_k_idx[x_k]) - 1.0)
                
                f = nb_kq - nb_k
                f1 = omega + 1j * eta + se_kq_idx[x_kq] - se_k_idx[x_k]
                f = f / f1
                
                # 计算该通道和 m 的贡献
                trace_val = jnp.trace(T_kq @ u[mu_idx, :, :] @ T_k @ u[nu_idx, :, :])
                return f * trace_val
            
            # 对 m 维度向量化
            m_array = jnp.arange(15)
            contributions = jax.vmap(compute_for_m)(m_array)
            
            return jnp.sum(contributions)
        
        # 对所有idx求和
        idx_array = jnp.arange(U_kq.shape[0])
        total_contributions = jax.vmap(compute_for_idx)(idx_array)
        return jnp.sum(total_contributions)
    
    # ====== JIT 外的 Python 循环：处理通道和k点 ======
    for mu in range(3):
        for nu in range(3):
            # 在 Python 层面检查通道是否为零
            if channel[mu, nu] == 0:
                continue
            
            print(f"Calculating channel ({mu}, {nu}) contribution...")
            # 对每个k点计算该通道的贡献
            channel_spectrum = jnp.array([
                compute_channel_kpoint(float(kpath[i, 0]), float(kpath[i, 1]), mu, nu)
                for i in range(kpath_len)
            ], dtype=jnp.complex128)
            
            spectrum = spectrum + channel[mu, nu] * channel_spectrum
    
    # 取虚部作为光谱强度（动态磁化率虚部）
    spectrum = -jnp.imag(spectrum) / Nsites
    
    return spectrum

def calculate_spectral_jax_vectorized(kpath: jnp.ndarray, omega_array: jnp.ndarray,
                                      eta: float, channel: jnp.ndarray,
                                      A1: complex, A2: complex, A3: complex,
                                      B1: float, B2: float, B3: float,
                                      lambda_param: float, h: float,
                                      k1_all: jnp.ndarray, k2_all: jnp.ndarray,
                                      Q1: float, Q2: float,
                                      J1plus: float, J2plus: float, J3plus: float,
                                      batch_size: int = 10, beta: float = 100) -> jnp.ndarray:
    """
    分批计算光谱函数 - JAX版本（内存高效）
    
    Parameters:
    -----------
    kpath : jnp.ndarray
        k路径，形状 (kpath_len, 2)
    omega_array : jnp.ndarray
        频率数组 (omega_len,)
    eta : float
        展宽参数
    channel : jnp.ndarray
        通道矩阵 (3, 3)
    batch_size : int
        每批处理的频率数（默认10，可根据内存调整）
    其他参数同上
        
    Returns:
    --------
    jnp.ndarray
        光谱函数 (kpath_len, omega_len)
    """
    omega_modified = 0  # 修正频率初始化
    kpath_len = kpath.shape[0]
    omega_len = omega_array.shape[0]
    
    U_k, eng_k = Bogoliubov_transform_jax_batch(
        omega_modified, k1_all, k2_all, Q1, Q2,
        A1, A2, A3, B1, B2, B3, lambda_param, h,
        J1plus, J2plus, J3plus
    )

    # 分批处理频率，避免内存溢出
    spec_batches = []
    for start_idx in range(0, omega_len, batch_size):
        end_idx = min(start_idx + batch_size, omega_len)
        omega_batch = omega_array[start_idx:end_idx]
        
        # 不使用 JIT，直接计算以提高编译速度
        def compute_batch(omega_batch):
            def compute_single_omega(w):
                return calculate_spectral_single_omega(
                    w, kpath, eta, channel, A1, A2, A3, B1, B2, B3,
                    lambda_param, h, k1_all, k2_all, Q1, Q2,
                    J1plus, J2plus, J3plus, U_k, eng_k,
                    beta
                )
            
            return jax.vmap(compute_single_omega)(omega_batch)
        
        # 计算该批次 (batch_size, kpath_len)
        spec_batch = compute_batch(omega_batch)
        spec_batches.append(spec_batch)
    
    # 拼接所有批次 (omega_len, kpath_len)
    spec = jnp.concatenate(spec_batches, axis=0)
    
    # 转置为 (kpath_len, omega_len)
    return spec.T
