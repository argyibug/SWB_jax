#!/bin/bash
# 修复JAX GPU显存不足问题的启动脚本

export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1"

# 启用动态内存分配（重要！）
export TF_FORCE_GPU_ALLOW_GROWTH=true

# 限制显存使用（可选）
export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_PER_DEVICE=2

echo "【GPU内存修复】已启用动态内存分配"
echo "运行计算..."
python "$@"
