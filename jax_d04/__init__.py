"""
三角格子反铁磁体自旋子(Spinon)物理的鞍点计算和光谱函数计算Python包

这是一个从MATLAB代码转写而来的Python包，用于计算：
1. 自旋子的鞍点(saddle point)优化
2. Bogoliubov变换
3. 动态磁化率和光谱函数
4. 磁振子色散关系

作者：从MATLAB代码转写
版本：1.0.0
"""

__version__ = "1.0.0"
__author__ = "Converted from MATLAB"

# 导入主要模块
from gamma_functions import gamma_functions
from matrix_elements import matrix_elements
from bogoliubov_transform import bogoliubov_transform
from saddle_point_optimization import saddle_point_optimization
from spectral_calculation import spectral_calculation
from visualization import visualization

# 导入常用函数
from main_calculations import SWBSystem
from visualization import plot_dispersion, plot_spectral_intensity