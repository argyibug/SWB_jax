from IO import *
from visualization import *


load = f'results/swb_L{5}.dat'
A1, A2, A3, B1, B2, B3, lambda_param = read_results_from_file(load)
print(f'lambda: {lambda_param}')
plot_dispersion(A1, A2, A3, B1, B2, B3)

