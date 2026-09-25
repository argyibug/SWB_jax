import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from pathlib import Path

def plot_opt(ax, logfile, yidx, color, lab):
    # Your plotting logic here
    if not logfile.exists():
        raise FileNotFoundError(f"Log file not found: {logfile}")
    logdata = np.loadtxt(logfile, unpack = True)

    ax.plot(logdata[0], logdata[yidx], '-', linewidth=3, label=lab, color=color)
    ax.plot(logdata[0], logdata[yidx], 'o', markersize=5, color=color)

    ax.tick_params(axis='both', which='major', labelsize=text_size)
    ax.tick_params(direction='in', width=3, length=8, top=True, bottom=True, left=True, right=True)
    ax.set_xlabel(r'$\mathrm{Optimizing\,time}$', fontsize=text_size)

    return ax

# 设置LaTeX字体
plt.rcParams.update({
    "text.usetex": True,
    "font.family": 'Times New Roman',
    "font.weight": 900,
    "text.latex.preamble": r"""
        \usepackage{amsmath}
        \usepackage{amsfonts}
        \usepackage{bm}
        \usepackage{amssymb}
    """
})

font = {'family': 'Times New Roman',
        'size': 40}

text_size = 30
show_plots = os.environ.get("SWB_SHOW_PLOTS") == "1"

# 创建N的范围
m = np.linspace(1, 100, 1000)

# 计算两条曲线
fig = plt.figure(figsize=(20, 7), constrained_layout=True)

# 创建图形
base_dir = Path(__file__).resolve().parent
logfile=[base_dir / "jax_d35" / "opt.log",
         base_dir / "jax_d04" / "opt.log"]


gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.25, hspace=0.1)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])

# 绘制曲线
print(f"读取日志文件: {logfile[0]}, {logfile[1]}")
plot_opt(ax0, logfile[0], yidx=1, color='blue', lab=r'$A$')
plot_opt(ax0, logfile[0], yidx=2, color='red', lab=r'$B$')
plot_opt(ax0, logfile[0], yidx=4, color='orange', lab=r'$(\lambda-2S)^2$')

plot_opt(ax1, logfile[1], yidx=2, color='blue', lab=r'$A$')
plot_opt(ax1, logfile[1], yidx=3, color='red', lab=r'$B$')
plot_opt(ax1, logfile[1], yidx=5, color='orange', lab=r'$(\lambda-2S)^2$')

# 使用 constrained_layout，避免 tight_layout 与当前轴组合产生兼容性警告

ax0.legend(fontsize=25, loc='upper right', frameon=True, 
           facecolor='white', edgecolor='black', framealpha=1)  
ax1.legend(fontsize=25, loc='upper right', frameon=True, 
          facecolor='white', edgecolor='black', framealpha=1)

ax0.text(0.5, 0.95, '(a)', transform=ax0.transAxes, fontsize=30, fontweight='bold', va='top', ha='center')
ax1.text(0.5, 0.95, '(b)', transform=ax1.transAxes, fontsize=30, fontweight='bold', va='top', ha='center')

# 保存图形（固定保存到脚本目录，避免因运行目录不同产生多份文件）
fig_test_path = base_dir / 'Fig_test.pdf'
plt.savefig(fig_test_path, dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', format="pdf", bbox_inches='tight')

# 显示图形
if show_plots:
    plt.show()

print(f"Figure saved as: {fig_test_path}")
fig_opt_pdf_path = base_dir / 'Fig_opt.pdf'
plt.savefig(fig_opt_pdf_path, dpi=None, facecolor='w', edgecolor='w', orientation='portrait', format="pdf", bbox_inches='tight')
fig_opt_jpg_path = base_dir / 'Fig_opt.jpg'
plt.savefig(fig_opt_jpg_path, dpi=300, facecolor='w', edgecolor='w', orientation='portrait', format="jpg", bbox_inches='tight')
print(f"Figure saved as: {fig_opt_pdf_path}")
print(f"Figure saved as: {fig_opt_jpg_path}")
if show_plots:
    plt.show()
