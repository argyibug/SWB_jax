# 安装说明

## 依赖库安装

本项目提供了三种安装依赖库的方式，请根据您的环境选择：

### 方式一：Windows批处理脚本（推荐）

双击运行 `install_dependencies.bat` 文件，或在命令行中执行：

```cmd
install_dependencies.bat
```

### 方式二：PowerShell脚本

在PowerShell中执行：

```powershell
.\install_dependencies.ps1
```

如果遇到权限问题，可能需要先设置执行策略：

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 方式三：Python脚本

```bash
python install_dependencies.py
```

### 方式四：手动安装

如果上述方式都不适用，可以手动执行以下命令：

```bash
# 升级pip
python -m pip install --upgrade pip

# 安装依赖库
pip install -r requirements.txt

# 安装项目包（开发模式，可选）
pip install -e .
```

## 依赖库列表

- **numpy** (>=1.21.0) - 数组计算和数值运算
- **scipy** (>=1.7.0) - 科学计算和线性代数
- **matplotlib** (>=3.5.0) - 数据可视化
- **seaborn** (>=0.11.0) - 高级可视化
- **nlopt** (>=2.7.0) - 非线性优化

## 验证安装

安装完成后，可以运行以下命令验证：

```python
python -c "import numpy, scipy, matplotlib, seaborn, nlopt; print('所有依赖库安装成功！')"
```

## 快速测试

运行示例计算：

```bash
python run_swb_calculation.py
```

## 常见问题

### nlopt安装失败

如果nlopt安装失败，可以尝试：

1. 使用conda安装：`conda install -c conda-forge nlopt`
2. 从源码编译安装
3. 下载预编译的wheel包

### 其他依赖问题

如果某个库安装失败，可以尝试单独安装：

```bash
pip install numpy
pip install scipy
pip install matplotlib
pip install seaborn
```
