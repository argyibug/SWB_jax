"""
gamma函数模块 - JAX版本（支持GPU加速）

Author: ZhouChk
JAX优化: 支持GPU/TPU加速和自动微分
"""

import jax
import jax.numpy as jnp
from typing import Union, Tuple
from functools import partial

# 全局变量
class GlobalParams:
    """存储全局参数的类"""
    J1plus: float = 0.0
    J2plus: float = 0.0  
    J3plus: float = 0.0
    Q1: float = 0.0
    Q2: float = 0.0

# 创建全局参数实例
_globals = GlobalParams()

def set_global_params(**kwargs):
    """设置全局参数"""
    for key, value in kwargs.items():
        if hasattr(_globals, key):
            setattr(_globals, key, value)
        else:
            raise ValueError(f"未知参数: {key}")

@partial(jax.jit, static_argnums=())
def gamma_A_jax(k1: jnp.ndarray, k2: jnp.ndarray, 
                A1: complex, A2: complex, A3: complex,
                J1plus: float, J2plus: float, J3plus: float) -> jnp.ndarray:
    """
    计算gamma_A函数 - JAX版本（JIT编译，支持GPU）
    
    Parameters:
    -----------
    k1, k2 : jnp.ndarray
        动量分量
    A1, A2, A3 : complex
        鞍点参数A
    J1plus, J2plus, J3plus : float
        交换耦合参数
        
    Returns:
    --------
    jnp.ndarray
        gamma_A的值
    """
    result = -1j * (J1plus * A1 * jnp.sin(k1) + 
                    J2plus * A2 * jnp.sin(k2) + 
                    J3plus * A3 * jnp.sin(k2 - k1))
    return result

@partial(jax.jit, static_argnums=())
def gamma_B_jax(k1: jnp.ndarray, k2: jnp.ndarray, 
                B1: float, B2: float, B3: float,
                J1plus: float, J2plus: float, J3plus: float) -> jnp.ndarray:
    """
    计算gamma_B函数 - JAX版本（JIT编译，支持GPU）
    
    Parameters:
    -----------
    k1, k2 : jnp.ndarray
        动量分量
    B1, B2, B3 : float
        鞍点参数B
    J1plus, J2plus, J3plus : float
        交换耦合参数
        
    Returns:
    --------
    jnp.ndarray
        gamma_B的值
    """
    result = (J1plus * B1 * jnp.cos(k1) + 
              J2plus * B2 * jnp.cos(k2) + 
              J3plus * B3 * jnp.cos(k2 - k1))
    return result

# 兼容接口（使用全局参数）
def gamma_A(k1: Union[float, jnp.ndarray], 
           k2: Union[float, jnp.ndarray], 
           A1: complex, A2: complex, A3: complex) -> Union[complex, jnp.ndarray]:
    """
    计算gamma_A函数（兼容原接口）
    """
    k1 = jnp.atleast_1d(k1)
    k2 = jnp.atleast_1d(k2)
    return gamma_A_jax(k1, k2, A1, A2, A3, 
                       _globals.J1plus, _globals.J2plus, _globals.J3plus)

def gamma_B(k1: Union[float, jnp.ndarray], 
           k2: Union[float, jnp.ndarray], 
           B1: float, B2: float, B3: float) -> Union[float, jnp.ndarray]:
    """
    计算gamma_B函数（兼容原接口）
    """
    k1 = jnp.atleast_1d(k1)
    k2 = jnp.atleast_1d(k2)
    return gamma_B_jax(k1, k2, B1, B2, B3,
                       _globals.J1plus, _globals.J2plus, _globals.J3plus)
