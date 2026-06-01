import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_opt(ax, logfile, yidx, color, lab):
    # Your plotting logic here
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

# 创建N的范围
m = np.linspace(1, 100, 1000)

# 计算两条曲线
fig = plt.figure(figsize = (20, 7))

# 创建图形
logfile=["D:\\0DOCUMENTS\\DSL\\MATLAB\\SWB\\SWB_jax\\jax_d35\\opt.log",
         "D:\\0DOCUMENTS\\DSL\\MATLAB\\SWB\\SWB_jax\\jax_d04\\opt.log"]


gs = gridspec.GridSpec(1, 2, wspace = 0.25, hspace = 0.1)
ax0=plt.subplot(gs[0, 0])
ax1=plt.subplot(gs[0, 1])

# 绘制曲线
plot_opt(ax0, logfile[0], yidx=1, color='blue', lab=r'$A$')
plot_opt(ax0, logfile[0], yidx=2, color='red', lab=r'$B$')
plot_opt(ax0, logfile[0], yidx=4, color='orange', lab=r'$(\lambda-2S)^2$')

plot_opt(ax1, logfile[1], yidx=2, color='blue', lab=r'$A$')
plot_opt(ax1, logfile[1], yidx=3, color='red', lab=r'$B$')
plot_opt(ax1, logfile[1], yidx=5, color='orange', lab=r'$(\lambda-2S)^2$')

# 调整布局
plt.tight_layout()

ax0.legend(fontsize=25, loc='upper right', frameon=True, 
          facecolor='white', edgecolor='black', framealpha=1)  
ax1.legend(fontsize=25, loc='upper right', frameon=True, 
          facecolor='white', edgecolor='black', framealpha=1)

ax0.text(0.5, 0.95, '(a)', transform=ax0.transAxes, fontsize=30, fontweight='bold', va='top', ha='center')
ax1.text(0.5, 0.95, '(b)', transform=ax1.transAxes, fontsize=30, fontweight='bold', va='top', ha='center')

# 保存图形
pdf_file_name = 'Fig_test.pdf'
plt.savefig(pdf_file_name, dpi=300, facecolor='w', edgecolor='w', 
            orientation='portrait', format="pdf", bbox_inches='tight')

# 显示图形
plt.show()

print(f"Figure saved as: {pdf_file_name}")
pdf_file_name = 'Fig_opt.pdf'
plt.savefig(pdf_file_name, dpi=None, facecolor='w', edgecolor='w', orientation='portrait', format="pdf", bbox_inches='tight')
pdf_file_name = 'Fig_opt.jpg'
plt.savefig(pdf_file_name, dpi=300, facecolor='w', edgecolor='w', orientation='portrait', format="jpg", bbox_inches='tight')
plt.show()
