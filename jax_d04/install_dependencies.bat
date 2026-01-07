@echo off
REM ====================================
REM 安装SWB项目依赖库 - Windows批处理脚本
REM ====================================

echo ========================================
echo 开始安装SWB项目依赖库...
echo ========================================
echo.

REM 升级pip到最新版本
echo [1/3] 升级pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo 错误: pip升级失败！
    pause
    exit /b 1
)
echo.

REM 安装依赖库
echo [2/3] 安装依赖库...
pip install -r requirements.txt
if errorlevel 1 (
    echo 错误: 依赖库安装失败！
    pause
    exit /b 1
)
echo.

REM 安装开发模式包（可选）
echo [3/3] 安装项目包（开发模式）...
pip install -e .
if errorlevel 1 (
    echo 警告: 项目包安装失败，但核心依赖已安装。
)
echo.

echo ========================================
echo 安装完成！
echo ========================================
echo.
echo 已安装的依赖库：
pip list | findstr "numpy scipy matplotlib seaborn nlopt"
echo.

pause
