# ====================================
# 安装SWB项目依赖库 - PowerShell脚本
# ====================================

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "开始安装SWB项目依赖库..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 升级pip到最新版本
Write-Host "[1/3] 升级pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Host "错误: pip升级失败！" -ForegroundColor Red
    Read-Host "按Enter键退出"
    exit 1
}
Write-Host ""

# 安装依赖库
Write-Host "[2/3] 安装依赖库..." -ForegroundColor Yellow
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "错误: 依赖库安装失败！" -ForegroundColor Red
    Read-Host "按Enter键退出"
    exit 1
}
Write-Host ""

# 安装开发模式包（可选）
Write-Host "[3/3] 安装项目包（开发模式）..." -ForegroundColor Yellow
pip install -e .
if ($LASTEXITCODE -ne 0) {
    Write-Host "警告: 项目包安装失败，但核心依赖已安装。" -ForegroundColor Yellow
}
Write-Host ""

Write-Host "========================================" -ForegroundColor Green
Write-Host "安装完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

Write-Host "已安装的依赖库：" -ForegroundColor Cyan
pip list | Select-String "numpy|scipy|matplotlib|seaborn|nlopt"
Write-Host ""

Read-Host "按Enter键退出"
