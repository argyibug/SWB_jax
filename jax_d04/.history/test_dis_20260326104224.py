import os
import numpy as np
import matplotlib.pyplot as plt

from IO import read_results_from_file
from gamma_functions import set_global_params
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

	Nsites = L1 * L1
	h = 1.0 / Nsites

	set_global_params(J1plus=J1plus, J2plus=J2plus, J3plus=J3plus, Q1=Q1, Q2=Q2)

	k_path, k_distances, k_tick_positions = get_triangular_lattice_path(L1)
	k_cartesian = convert_to_cartesian_coordinates(k_path)

	n_points = min(len(k_cartesian), max(50, 5 * L1))
	indices = np.linspace(0, len(k_cartesian) - 1, n_points, dtype=int)
	eigenvalues = np.zeros((n_points, 2))

	for i, idx in enumerate(indices):
		kx, ky = k_cartesian[idx]
		_, ek, _, _, _, _ = Bogoliubov_transform_2_jax(
			0, np.array([kx]), np.array([ky]),
			A1, A2, A3, B1, B2, B3, lambda_param, h
		)
		eigenvalues[i, 0] = ek[0, 0]
		eigenvalues[i, 1] = ek[0, 1]

	os.makedirs("results", exist_ok=True)
	output = f"results/dispersion_L{L1}.png"

	plot_dispersion(
		k_distances[indices],
		eigenvalues,
		k_tick_positions,
		['X', 'M', 'Γ', "K'", 'M', 'Y', 'K'],
		title=f"Spinon Dispersion (L={L1})",
		save_path=output,
	)
	print(f"dispersion saved: {output}")
	plt.show()


if __name__ == "__main__":
	main()

