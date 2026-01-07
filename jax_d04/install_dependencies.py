#!/usr/bin/env python3
"""
SWB项目依赖安装脚本
使用pip安装所有必需的依赖库
"""

import subprocess
import sys

def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"\n{'='*50}")
    print(f"{description}")
    print(f"{'='*50}")
    try:
        subprocess.check_call(cmd, shell=True)
        print(f"✓ {description}完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description}失败: {e}")
        return False

def main():
    print("\n" + "="*50)
    print("SWB项目依赖安装脚本")
    print("="*50)
    
    # 1. 升级pip
    if not run_command(
        f"{sys.executable} -m pip install --upgrade pip",
        "[1/3] 升级pip"
    ):
        print("\n错误: pip升级失败！")
        return 1
    
    # 2. 安装依赖
    if not run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "[2/3] 安装依赖库"
    ):
        print("\n错误: 依赖库安装失败！")
        return 1
    
    # 3. 安装项目包（开发模式）
    if not run_command(
        f"{sys.executable} -m pip install -e .",
        "[3/3] 安装项目包（开发模式）"
    ):
        print("\n警告: 项目包安装失败，但核心依赖已安装。")
    
    # 显示安装结果
    print("\n" + "="*50)
    print("安装完成！")
    print("="*50)
    print("\n已安装的依赖库：")
    subprocess.run(
        f"{sys.executable} -m pip list | grep -E 'numpy|scipy|matplotlib|seaborn|nlopt'",
        shell=True
    )
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
