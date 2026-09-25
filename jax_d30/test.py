import IO
from Hamiltonian_jax import Ham_jax
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from saddle_point_optimization_jax import saddle_point_sum_jax

# 模型参数
J1xy = J2xy = J3xy = 1.0
J1z = J2z = J3z = 1.0  # 各向同性
S = 0.5

g = jnp.eye(4)
g = g.at[1, 1].set(-1)
g = g.at[3, 3].set(-1)
Q1 = 2*np.pi/3
Q2 = 4*np.pi/3

J1plus = (J1z + J1xy) / 2
J2plus = (J2z + J2xy) / 2
J3plus = (J3z + J3xy) / 2
A1 = A2 = A3 = 0.4934543715111092j
B1 = 0.2277674726161695
B2 = -B1
B3 = B1
lambda_param = 0.9608546444218063

L1 = 5
L2 = L1
k1_1d = 2*np.pi/L1 * np.arange(L1)
k2_1d = 2*np.pi/L2 * np.arange(L2)
k1_2d, k2_2d = np.meshgrid(k1_1d, k2_1d, indexing='ij')
k1 = k1_2d.flatten()
k2 = k2_2d.flatten()
Nsites = len(k1)
h = 1/Nsites

from Hamiltonian_jax import Ham_jax
# 构建哈密顿量
print(f"+++++++++++++++++++++++++=====================+++++++++++++++++++++++++=====================")
print(f"k1= {k1[0]}, k2= {k2[0]}, Q1= {Q1}, Q2= {Q2}, A1= {A1}, A2= {A2}, A3= {A3}, B1= {B1}, B2= {B2}, B3= {B3}, lambda_param= {lambda_param}, h= {h}")
print(f"+++++++++++++++++++++++++=====================+++++++++++++++++++++++++=====================")
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

H0 = np.asarray(H[0, :, :])
print("k", k1[0], k2[0])
print("哈密顿量示例元素 (H[0, :, :]):")
print(np.array2string(H0, precision=6, suppress_small=True, max_line_width=160))

from bogoliubov_transform_jax import Bogoliubov_transform_2_jax
from bogoliubov_transform_jax import Bogoliubov_constraint_jax_batch
min_eng = Bogoliubov_constraint_jax_batch(
	0, k1, k2, Q1, Q2, A1, A2, A3, B1, B2, B3, lambda_param, h,
	J1plus, J2plus, J3plus)
print(f"当前能量: {min_eng}")
idx=8
print(f"min_eng: {min_eng[idx]}, at k1: {k1[idx]}, k2: {k2[idx]}")

H0 = np.asarray(H[idx, :, :])
print("k", k1[idx], k2[idx])
print("哈密顿量示例元素 (H[idx, :, :]):")
print(np.array2string(H0, precision=6, suppress_small=True, max_line_width=160))
Ubov = Bogoliubov_transform_2_jax(
        0, k1, k2, Q1, Q2, A1, A2, A3, B1, B2, B3, lambda_param, h,
        J1plus, J2plus, J3plus)[0]

lam, AA, BB = saddle_point_sum_jax(Ubov, k1, k2, Q1, Q2)[0:3]  # 修复：[1:3]返回2个元素
print(f"===============================================+++++++++++++++++++++++++=====================")
print(f"Current  lam: {lam}, A: {AA}, B: {BB}")
print(f"inital  lam: {lambda_param}, A: {A1}, B: {B1}")
print(f"===============================================+++++++++++++++++++++++++=====================")

print(f"+++++++++++++++++++++++++=====================+++++++++++++++++++++++++=====================")
print(f"k1= {k1[0]}, k2= {k2[0]}, Q1= {Q1}, Q2= {Q2}, A1= {A1}, A2= {A2}, A3= {A3}, B1= {B1}, B2= {B2}, B3= {B3}, lambda_param= {lambda_param}, h= {h}")
print(f"+++++++++++++++++++++++++=====================+++++++++++++++++++++++++=====================")
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

H0 = np.asarray(H[0, :, :])
print("k", k1[0], k2[0])
print("哈密顿量示例元素 (H[0, :, :]):")
print(np.array2string(H0, precision=6, suppress_small=True, max_line_width=160))
print(f"===============================================+++++++++++++++++++++++++=====================")
from bogoliubov_transform_jax import Bogoliubov_constraint_jax
con_eig = Bogoliubov_constraint_jax(0, k1, k2, A1, A2, A3, B1, B2, B3, Q1, Q2,
									lambda_param, h, J1plus, J2plus, J3plus)

tolerance = 1e-5 / np.sqrt(Nsites)
c = [tolerance - float(con_eig)]

print(f"约束条件值: {con_eig}, 目标值: {tolerance}, 差值: {c[0]}")
