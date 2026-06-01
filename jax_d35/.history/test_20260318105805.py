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
B1 = B2 = B3 = 0.22640955
lambda_param = 0.94176189
    
# 读取晶格信息
# JAX 不能直接处理 object dtype，转为数值表: [i, j, site_i, site_j, rij_x, rij_y, type]

from Hamiltonian_jax import Ham_jax
# 构建哈密顿量
H = Ham_jax(
	omega=0.0,
	k1=k1,
	k2=k2,
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
	bond_tab=bond_tab_jax,
	spin_n=n_spin,
	bond_n=n_bond,
)
print(f"哈密顿量形状: {H.shape}")
H0 = np.asarray(H[0, :, :])
print("哈密顿量示例元素 (H[0, :, :]):")
print(np.array2string(H0, precision=6, suppress_small=True, max_line_width=160))

# 绘制 12x12 热力图：分别展示实部和虚部
def _annotate_heatmap(ax, data, fmt=".2f"):
	"""在热力图每个单元格中标注数值"""
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			val = data[i, j]
			label = f"{val:{fmt}}"
			txt = ax.text(
				j, i, label,
				ha="center", va="center",
				fontsize=8, fontweight="bold", color="white",
			)
			txt.set_path_effects([pe.withStroke(linewidth=1.5, foreground="black")])

fig, axes = plt.subplots(1, 2, figsize=(26, 12))

# --- 实部 ---
real_data = H0.real
vabs_r = max(np.abs(real_data).max(), 1e-10)
im_r = axes[0].imshow(real_data, cmap="coolwarm", origin="upper",
                      vmin=-vabs_r, vmax=vabs_r)
axes[0].set_title("Re(H[0])  —  Real Part")
axes[0].set_xlabel("j")
axes[0].set_ylabel("i")
axes[0].set_xticks(np.arange(H0.shape[1]))
axes[0].set_yticks(np.arange(H0.shape[0]))
_annotate_heatmap(axes[0], real_data)
cbar_r = fig.colorbar(im_r, ax=axes[0])
cbar_r.set_label("Re(H_ij)")

# --- 虚部 ---
imag_data = H0.imag
vabs_i = max(np.abs(imag_data).max(), 1e-10)
im_i = axes[1].imshow(imag_data, cmap="PuOr", origin="upper",
                      vmin=-vabs_i, vmax=vabs_i)
axes[1].set_title("Im(H[0])  —  Imaginary Part")
axes[1].set_xlabel("j")
axes[1].set_ylabel("i")
axes[1].set_xticks(np.arange(H0.shape[1]))
axes[1].set_yticks(np.arange(H0.shape[0]))
_annotate_heatmap(axes[1], imag_data)
cbar_i = fig.colorbar(im_i, ax=axes[1])
cbar_i.set_label("Im(H_ij)")

fig.suptitle("Hamiltonian H[0] Heatmap — Real & Imaginary Parts", fontsize=14)
fig.tight_layout()
fig.savefig("results/H0_heatmap_12x12.png", dpi=200)
plt.close(fig)
print("已生成热力图: results/H0_heatmap_12x12.png")

import bogoliubov_transform_jax
# test Bogoliubov transform
from bogoliubov_transform_jax import bogoliubov_single_k
print(f"Bogoliubov变换输入矩阵 H[0] 形状: {H[0].shape}")
print("spin_n:", n_spin)
Ubov, ek = bogoliubov_single_k(H[0], n_spin)
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