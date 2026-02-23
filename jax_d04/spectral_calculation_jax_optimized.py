#!/usr/bin/env python
"""
spectral_calculation_jax.py 的优化版本

主要优化:
1. 消除 Python 列表推导式，使用 jax.vmap 替代
2. 整合多个循环，一次性计算所有通道和k点
3. 确保所有数据都在GPU上，避免频繁转移
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple
from functools import partial
from bogoliubov_transform_jax import Bogoliubov_transform_jax_batch

def calculate_spectral_single_omega_optimized(
    omega: float, kpath: jnp.ndarray, eta: float,
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
    优化版本：消除Python循环，完全向量化
    
    关键改进:
    - 使用 jax.vmap 替代 Python for 循环
    - 一次性处理所有k点和通道
    - 避免频繁的GPU同步和数据转移
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
    
    # ===== 优化策略：完全向量化计算 =====
    
    @jax.jit
    def compute_all_channels_and_kpoints():
        """一次性计算所有通道和k点，避免Python循环"""
        
        # 初始化频谱
        spectrum = jnp.zeros(kpath_len, dtype=jnp.complex128)
        
        # 定义计算单个通道在所有k点的贡献的函数
        def compute_channel_all_kp(mu_nu_pair):
            """计算单个通道(mu, nu)在所有k点的贡献"""
            mu, nu = mu_nu_pair[0], mu_nu_pair[1]
            
            # 检查通道是否为零
            channel_coeff = channel[mu, nu]
            
            # 对所有k点向量化计算
            def compute_for_kpoint(kp):
                kx, ky = kp[0], kp[1]
                omega_modified = 0

                kq1_all = k1_all + kx + order_vec[mu, nu, 0]
                kq2_all = k2_all + ky + order_vec[mu, nu, 1]

                # 计算k+q点的Bogoliubov变换
                U_kq, eng_kq = Bogoliubov_transform_jax_batch(
                    omega_modified, kq1_all, kq2_all, Q1, Q2,
                    A1, A2, A3, B1, B2, B3, lambda_param, h,
                    J1plus, J2plus, J3plus
                )
                
                se_kq = eng_kq @ g  # (Nsites, 4)
                se_k = eng_k @ g    # (Nsites, 4)
                
                # 对所有idx和m计算贡献
                def compute_for_idx_m(idx_m):
                    idx = idx_m[0]
                    m = idx_m[1]
                    
                    uu_kq = U_kq[idx, :, :]  # (4, 4)
                    uu_k = U_k[idx, :, :]
                    uu_kq_d = uu_kq.conj().T
                    uu_k_d = uu_k.conj().T
                    
                    se_kq_idx = se_kq[idx, :]  # (4,)
                    se_k_idx = se_k[idx, :]
                    
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
                    trace_val = jnp.trace(T_kq @ u[mu, :, :] @ T_k @ u[nu, :, :])
                    return f * trace_val
                
                # 对 idx 和 m 维度向量化
                Nsites_range = jnp.arange(U_kq.shape[0])
                m_range = jnp.arange(15)
                
                # 创建 idx_m 的所有组合
                idx_mesh, m_mesh = jnp.meshgrid(Nsites_range, m_range, indexing='ij')
                idx_m_pairs = jnp.stack([idx_mesh.flatten(), m_mesh.flatten()], axis=1)
                
                contributions = jax.vmap(compute_for_idx_m)(idx_m_pairs)
                return jnp.sum(contributions)
            
            # 对所有k点向量化计算 ← 这里替代了Python列表推导式！
            channel_spectrum = jax.vmap(compute_for_kpoint)(kpath)
            
            return channel_coeff * channel_spectrum
        
        # 对所有通道对向量化计算
        mu_range = jnp.arange(3)
        nu_range = jnp.arange(3)
        mu_mesh, nu_mesh = jnp.meshgrid(mu_range, nu_range, indexing='ij')
        mu_nu_pairs = jnp.stack([mu_mesh.flatten(), nu_mesh.flatten()], axis=1)
        
        # 计算所有通道的贡献
        channel_contributions = jax.vmap(compute_channel_all_kp)(mu_nu_pairs)
        
        # 求和所有通道
        spectrum = jnp.sum(channel_contributions, axis=0)
        
        # 取虚部作为光谱强度（动态磁化率虚部）
        spectrum = -jnp.imag(spectrum) / Nsites
        
        return spectrum
    
    return compute_all_channels_and_kpoints()


def calculate_spectral_jax_vectorized_optimized(
    kpath: jnp.ndarray, omega_array: jnp.ndarray,
    eta: float, channel: jnp.ndarray,
    A1: complex, A2: complex, A3: complex,
    B1: float, B2: float, B3: float,
    lambda_param: float, h: float,
    k1_all: jnp.ndarray, k2_all: jnp.ndarray,
    Q1: float, Q2: float,
    J1plus: float, J2plus: float, J3plus: float,
    batch_size: int = 10, beta: float = 100) -> jnp.ndarray:
    """
    优化版本：对频率轴使用 jax.vmap，而不是Python循环
    
    改进:
    - 对频率轴使用 jax.vmap 而不是 for 循环
    - 减少Python↔JAX的交界处往返
    - 最大化GPU并行度
    """
    
    omega_modified = 0  
    
    U_k, eng_k = Bogoliubov_transform_jax_batch(
        omega_modified, k1_all, k2_all, Q1, Q2,
        A1, A2, A3, B1, B2, B3, lambda_param, h,
        J1plus, J2plus, J3plus
    )

    # 使用 vmap 替代循环处理频率数组
    @jax.jit
    def compute_spectral_all_omegas(omega_array):
        """对所有频率向量化计算"""
        def compute_single_omega(w):
            return calculate_spectral_single_omega_optimized(
                w, kpath, eta, channel, A1, A2, A3, B1, B2, B3,
                lambda_param, h, k1_all, k2_all, Q1, Q2,
                J1plus, J2plus, J3plus, U_k, eng_k,
                beta
            )
        
        # 对频率数组向量化
        return jax.vmap(compute_single_omega)(omega_array)
    
    # 为了避免编译超时，仍然分批处理频率
    spec_batches = []
    for start_idx in range(0, len(omega_array), batch_size):
        end_idx = min(start_idx + batch_size, len(omega_array))
        omega_batch = omega_array[start_idx:end_idx]
        
        print(f"计算频率 {start_idx}-{end_idx}...")
        spec_batch = compute_spectral_all_omegas(omega_batch)
        spec_batches.append(spec_batch)
    
    # 拼接所有批次 (omega_len, kpath_len)
    spec = jnp.concatenate(spec_batches, axis=0)
    
    # 转置为 (kpath_len, omega_len)
    return spec.T
