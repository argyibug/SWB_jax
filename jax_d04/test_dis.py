import os
import numpy as np
import matplotlib.pyplot as plt

from IO import read_results_from_file
from IO import create_bond_table_file, convert_bond_table_to_jax_array
from gamma_functions_jax import set_global_params
from bogoliubov_transform_jax import Bogoliubov_transform_2_jax
from spectral_calculation import get_triangular_lattice_path, convert_to_cartesian_coordinates
from visualization import plot_dispersion


def main() -> None:
	L1 = 5
	load = f"results/swb_L{L1}.dat"

	A1, A2, A3, B1, B2, B3, lambda_param = read_results_from_file(load)
	print(f"lambda: {lambda_param}")

	# 与主流程保持一致的模型设置
	J1xy = J2xy = J3xy = 1.0
	J1z = J2z = J3z = 1.0
	J1plus = (J1z + J1xy) / 2
	J2plus = (J2z + J2xy) / 2
	J3plus = (J3z + J3xy) / 2

	# 读取并转换键表，确保传入 JAX 的是纯数值数组
	bond_tab_raw, n_spin, n_bond = create_bond_table_file(
		unit_vector_filepath="unit_vector.in", cellspin_filepath="cellspin.in"
	)
	bond_tab = convert_bond_table_to_jax_array(bond_tab_raw)

	Nsites = L1 * L1
	h = 1.0 / Nsites

	set_global_params(
		J1plus=J1plus,
		J2plus=J2plus,
		J3plus=J3plus,
		bond_tab=bond_tab,
		bond_n=n_bond,
		spin_n=n_spin,
	)

	k_path, k_distances, k_tick_positions = get_triangular_lattice_path(L1)
	k_cartesian = convert_to_cartesian_coordinates(k_path)

	n_points = min(len(k_cartesian), max(50, 5 * L1))
	indices = np.linspace(0, len(k_cartesian) - 1, n_points, dtype=int)
	k_path_selected = k_cartesian[indices]
	kx_batch = k_path_selected[:, 0]
	ky_batch = k_path_selected[:, 1]

	# JAX 批量计算色散本征值
	_, ek_batch = Bogoliubov_transform_2_jax(
		0,
		kx_batch,
		ky_batch,
		A1,
		A2,
		A3,
		B1,
		B2,
		B3,
		lambda_param,
		h,
		J1plus,
		J2plus,
		J3plus,
		bond_tab,
		n_spin,
		n_bond,
	)
	eigenvalues = np.array(ek_batch[:, :])

	os.makedirs("results", exist_ok=True)
	output = f"results/dispersion_L{L1}.png"

	plot_dispersion(
		k_distances[indices],
		eigenvalues,
		k_tick_positions,
		["X", "M", "Γ", "K'", "M", "Y", "K"],
		title=f"Spinon Dispersion (L={L1})",
		save_path=output,
	)
	print(f"dispersion saved: {output}")
	plt.show()


if __name__ == "__main__":
    main()

