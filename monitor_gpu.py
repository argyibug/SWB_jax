#!/usr/bin/env python
"""
GPU使用监控脚本
在另一个终端运行此脚本，同时运行计算，观察GPU内存和加载
"""

import subprocess
import time
import sys

def monitor_gpu(interval=1, duration=60):
    """
    监控GPU使用情况
    
    Parameters:
    -----------
    interval : int
        刷新间隔（秒）
    duration : int
        监控时长（秒），0表示无限
    """
    
    print("GPU 监控开始...")
    print("继续运行你的计算脚本，这个脚本会实时显示GPU使用情况")
    print("按 Ctrl+C 停止监控")
    print("="*80)
    
    start_time = time.time()
    
    try:
        while True:
            # 运行nvidia-smi并格式化输出
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu', 
                     '--format=csv,noheader'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0:
                    # 清屏（跨平台）
                    subprocess.run(['clear' if sys.platform != 'win32' else 'cls'], shell=True)
                    
                    print("GPU 使用情况监控")
                    print("="*80)
                    print(f"监控时长: {time.time() - start_time:.0f}s")
                    print()
                    
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        print(f"  {line}")
                    
                    # 检查GPU是否在使用
                    if lines:
                        stats = lines[0].split(', ')
                        if len(stats) >= 5:
                            gpu_util = stats[4].strip().rstrip('%')
                            mem_used = stats[2].strip()
                            try:
                                if float(gpu_util) > 10:
                                    print(f"\n✓ GPU正在使用中 (利用率: {gpu_util}%)")
                                else:
                                    print(f"\n⚠️ GPU利用率低 (利用率: {gpu_util}%)")
                            except:
                                pass
                    
                    # 检查持续时间
                    if duration > 0 and (time.time() - start_time) > duration:
                        break
                        
            except subprocess.TimeoutExpired:
                print("nvidia-smi 查询超时")
            except Exception as e:
                print(f"错误: {e}")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n监控已停止")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='监控GPU使用情况')
    parser.add_argument('--interval', type=int, default=1, help='刷新间隔（秒）')
    parser.add_argument('--duration', type=int, default=0, help='监控时长（秒，0=无限）')
    
    args = parser.parse_args()
    monitor_gpu(interval=args.interval, duration=args.duration)
