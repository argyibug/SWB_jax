import IO
from Hamiltonian_jax import Ham_jax
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# 模型参数
J1xy = J2xy = J3xy = 1.0
J1z = J2z = J3z = 1.0  # 各向同性
S = 0.5
    
# 计算组合参数
J1plus = (J1z + J1xy) / 2
J2plus = (J2z + J2xy) / 2
J3plus = (J3z + J3xy) / 2
    
# 晶格参数
L1 = 10
L2 = L1
k1 = jnp.array([10])
k2 = jnp.array([10])
Nsites = 10
h = 0

A1 = A2 = A3 = 0.49126303j
B1 = 0.22640955
B2 = -B1
B3 = B1
lambda_param = 0.94176189
    

Q1 = 2*np.pi/3
Q2 = 4*np.pi/3
# 读取晶格信息
# JAX 不能直接处理 object dtype，转为数值表: [i, j, site_i, site_j, rij_x, rij_y, type]

from Hamiltonian_jax import Ham_jax
# 构建哈密顿量
H = Ham_jax(
	omega=0.0,
	k1=k1,
	k2=k2,
	Q1=Q1,
	Q2=Q2,
	A1=A1,
	A2=A2,
	A3=A3,
	B1=B1,
	B2=B2,
	B3=B3,
	lambda_param=lambda_param,
	h=h,
	J1plus=J1plus,
	J2plus=J2plus,
	J3plus=J3plus,
)
print(f"哈密顿量形状: {H.shape}")
H0 = np.asarray(H[0, :, :])
print("哈密顿量示例元素 (H[0, :, :]):")
print(np.array2string(H0, precision=6, suppress_small=True, max_line_width=160))

import bogoliubov_transform_jax
# test Bogoliubov transform
from bogoliubov_transform_jax import bogoliubov_single_k
print(f"Bogoliubov变换输入矩阵 H[0] 形状: {H[0].shape}")
Ubov, ek = bogoliubov_single_k(H[0])
# print("Bogoliubov变换矩阵 Ubov:")
# print(np.array2string(np.asarray(Ubov), precision=6, suppress_small=True, max_line_width=160))
print("能谱 ek:")
print(np.array2string(np.asarray(ek), precision=6, suppress_small=True, max_line_width=160))

g = jnp.eye(4)
g = g.at[1, 1].set(-1)
g = g.at[3, 3].set(-1)
Ubov_dag = jnp.conjugate(jnp.transpose(Ubov))
check = Ubov_dag @ g @ Ubov-g
print("Bogoliubov变换的正则化检查 (Ubov^† g Ubov - g):")
print(np.array2string(np.asarray(check), precision=6, suppress_small=True, max_line_width=160))

# test compute_single_k_contribution
from bogoliubov_transform_jax import saddle_point_sum_jax
lambda_t, A_t, B_t, Usum_t = saddle_point_sum_jax(H[0], Ubov, ek)
print("单个k点的贡献:")
print(np.array2string(np.asarray(contribution), precision=6, suppress_small=True, max_line_width=160))