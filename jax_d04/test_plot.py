from pathlib import Path

from IO import plot_point_bond_check


def main() -> None:
	base_dir = Path(__file__).resolve().parent
	plot_point_bond_check(
		filepath_unit_vector=str(base_dir / 'unit_vector.in'),
		filepath_bond=str(base_dir / 'bond.log'),
		filepath_spin=str(base_dir / 'cellspin.in'),
	)


if __name__ == '__main__':
	main()
