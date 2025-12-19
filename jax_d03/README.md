# 三角格子反铁磁体自旋子物理计算Python包

## 功能特性

- 🔬 **鞍点优化**：使用NLopt进行高效的鞍点参数优化
- 📊 **Bogoliubov变换**：计算自旋子的准粒子态  
- 🌊 **光谱函数计算**：计算动态磁化率和光谱函数
- 📈 **可视化**：使用matplotlib生成专业的物理图表
- 🚀 **高性能**：使用numpy/scipy优化的数值计算
- ⚡ **GPU加速**：支持JAX GPU加速，计算速度提升10-100倍（可选）

## 🎯 快速开始GPU加速

如果您有NVIDIA GPU，**5分钟内**即可启用GPU加速：

```bash
# 1. 安装JAX
pip install jax[cuda12]  # CUDA 12.x

# 2. 运行测试
python test_jax_performance.py

# 3. 运行加速计算
python run_swb_jax.py
```

详细步骤请参考 → [JAX 5分钟快速入门](JAX_QUICKSTART.md)

## 安装

### 快速安装（Windows）

运行安装脚本自动安装所有依赖：

```bash
# 方式1: 批处理脚本（推荐）
install_dependencies.bat

# 方式2: PowerShell脚本
.\install_dependencies.ps1

# 方式3: Python脚本
python install_dependencies.py
```

详细安装说明请参考 [INSTALL_GUIDE.md](INSTALL_GUIDE.md)

### 主要依赖

- numpy (>=1.21.0) - 数组计算
- scipy (>=1.7.0) - 科学计算 
- matplotlib (>=3.5.0) - 绘图
- seaborn (>=0.11.0) - 高级可视化
- nlopt (>=2.7.0) - 非线性优化
- jax (>=0.4.20) - GPU加速（可选，强烈推荐）

### GPU加速安装（可选）

启用GPU加速可获得显著性能提升：

```bash
# CUDA 12.x用户
pip install jax[cuda12]

# CUDA 11.x用户  
pip install jax[cuda11]

# CPU版本（测试用）
pip install jax[cpu]
```

详细说明请参考 [JAX_ACCELERATION_GUIDE.md](JAX_ACCELERATION_GUIDE.md)

## 快速开始

### 运行示例计算

```bash
# 标准版本（NumPy/SciPy）
python run_swb_calculation.py
python run_spectra_calculation.py

# JAX加速版本（推荐，需要安装JAX）
python run_swb_jax.py

# 性能测试
python test_jax_performance.py
```

### Python API使用

**标准版本（NumPy）：**

```python
from main_calculations import SWBSystem

# 创建SWB系统
system = SWBSystem()

# 设置系统参数
system.set_model_parameters(J1xy=1.0, J2xy=1.0, J3xy=1.0, S=0.5)
system.set_lattice_size(L1=20, L2=20)

# 进行鞍点优化
results = system.saddle_point_optimization()

# 计算并绘制色散关系
system.plot_dispersion()

# 计算并绘制光谱函数
system.plot_spectral_intensity()
```

**JAX加速版本（GPU）：**

```python
import jax.numpy as jnp
from bogoliubov_transform_jax import Bogoliubov_transform_jax_batch
from gamma_functions import set_global_params

# 设置参数
set_global_params(J1plus=1.0, J2plus=1.0, J3plus=1.0, Q1=4*np.pi/3, Q2=0)

# 准备数据（JAX数组）
k1 = jnp.linspace(-np.pi, np.pi, 100)
k2 = jnp.linspace(-np.pi, np.pi, 100)

# 执行计算（自动在GPU上运行）
Ubov, ek = Bogoliubov_transform_jax_batch(
    omega=0.0, k1=k1, k2=k2, Q1=4*np.pi/3, Q2=0.0,
    A1=0.5, A2=0.5, A3=0.5, B1=0.3, B2=-0.3, B3=0.3,
    lambda_param=1.0, h=0.01,
    J1plus=1.0, J2plus=1.0, J3plus=1.0
)
```

## 文件结构

```
python_d02/
├── __init__.py                    # 包初始化
├── gamma_functions.py             # γ函数计算
├── Hamiltonian.py                 # 哈密顿量构建
├── bogoliubov_transform.py        # Bogoliubov变换
├── saddle_point_optimization.py   # 鞍点优化
├── spectral_calculation.py        # 光谱函数计算
├── visualization.py               # 可视化工具
├── main_calculations.py           # 主要计算类（SWBSystem）
├── IO.py                          # 输入输出工具
│
├── gamma_functions_jax.py         # γ函数（JAX版本）⚡
├── Hamiltonian_jax.py             # 哈密顿量（JAX版本）⚡
├── bogoliubov_transform_jax.py    # Bogoliubov变换（JAX版本）⚡
├── spectral_calculation_jax.py    # 光谱计算（JAX版本）⚡
│
├── run_swb_calculation.py         # SWB计算示例
├── run_spectra_calculation.py     # 光谱计算示例
├── run_swb_jax.py                 # JAX加速示例⚡
├── test_jax_performance.py        # 性能测试⚡
│
├── requirements.txt               # 依赖库列表
├── setup.py                       # 安装配置
├── install_dependencies.bat       # Windows安装脚本
├── install_dependencies.ps1       # PowerShell安装脚本
├── install_dependencies.py        # Python安装脚本
│
├── README.md                      # 本文档
├── INSTALL_GUIDE.md              # 详细安装指南
└── JAX_ACCELERATION_GUIDE.md     # JAX加速使用指南⚡
```

⚡ = GPU加速相关文件

1. **面向对象设计**：使用类来组织相关功能
2. **现代Python特性**：类型提示、文档字符串等
3. **更好的错误处理**：详细的异常信息
4. **模块化设计**：更清晰的代码组织
5. **性能优化**：向量化计算，减少循环

## 物理背景

该代码计算三角格子反铁磁体的自旋子物理性质：

- **鞍点方法**：使用平均场理论处理量子自旋液体
- **Bogoliubov变换**：对角化有效哈密顿量
- **120°磁序**：三角格子的磁挫败基态
- **动态磁化率**：中子散射实验的理论对比

## 贡献

欢迎提交Issue和Pull Request来改进代码。
s