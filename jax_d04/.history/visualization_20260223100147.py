"""
可视化模块

Author: ZhouChk
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import MultipleLocator
from typing import Optional, List, Tuple, Union
import seaborn as sns

# 设置matplotlib样式
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def plot_dispersion(k_distances: np.ndarray, eigenvalues: np.ndarray, 
                   k_tick_positions: List[float], k_labels: List[str],
                   title: str = "Spinon Dispersion", 
                   ylabel: str = "Energy (meV)",
                   figsize: Tuple[int, int] = (10, 6),
                   save_path: Optional[str] = None) -> plt.Figure:
    """
    绘制色散关系图
    
    Parameters:
    -----------
    k_distances : ndarray
        k路径距离
    eigenvalues : ndarray
        本征值 (形状: n_k_points x n_bands)
    k_tick_positions : list
        高对称点位置
    k_labels : list
        高对称点标签
    title : str
        图标题
    ylabel : str
        y轴标签
    figsize : tuple
        图尺寸
    save_path : str, optional
        保存路径
        
    Returns:
    --------
    Figure
        matplotlib图对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制每个能带
    n_bands = eigenvalues.shape[1] if eigenvalues.ndim > 1 else 1
    
    if eigenvalues.ndim == 1:
        ax.plot(k_distances, eigenvalues, 'b-', linewidth=2, label='Band 1')
    else:
        colors_list = plt.cm.tab10(np.linspace(0, 1, n_bands))
        for i in range(n_bands):
            ax.plot(k_distances, eigenvalues[:, i], color=colors_list[i], 
                   linewidth=2, label=f'Band {i+1}')
    
    # 添加高对称点的垂直线
    for pos in k_tick_positions:
        ax.axvline(pos, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    # 设置x轴标签
    ax.set_xticks(k_tick_positions)
    ax.set_xticklabels(k_labels)
    ax.set_xlabel('')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # 如果有多个能带，显示图例
    if n_bands > 1:
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_spectral_intensity(k_distances: np.ndarray, omega: np.ndarray, 
                           spectral_data: np.ndarray,
                           k_tick_positions: List[float], k_labels: List[str],
                           title: str = "Spectral Intensity",
                           xlabel: str = "", ylabel: str = "ω/J",
                           omega_range: Optional[Tuple[float, float]] = None,
                           intensity_range: Optional[Tuple[float, float]] = None,
                           colormap: str = 'jet',
                           figsize: Tuple[int, int] = (12, 8),
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    绘制光谱强度热力图
    
    Parameters:
    -----------
    k_distances : ndarray
        k路径距离
    omega : ndarray
        频率数组
    spectral_data : ndarray
        光谱强度数据 (形状: n_omega x n_k)
    k_tick_positions : list
        高对称点位置
    k_labels : list
        高对称点标签
    title : str
        图标题
    xlabel, ylabel : str
        轴标签
    omega_range : tuple, optional
        频率范围
    intensity_range : tuple, optional
        强度范围
    colormap : str
        颜色映射
    figsize : tuple
        图尺寸
    save_path : str, optional
        保存路径
        
    Returns:
    --------
    Figure
        matplotlib图对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 创建网格
    K_grid, Omega_grid = np.meshgrid(k_distances, omega)
    
    # 绘制热力图
    if intensity_range:
        vmin, vmax = intensity_range
        im = ax.pcolormesh(K_grid, Omega_grid, spectral_data, 
                          cmap=colormap, vmin=vmin, vmax=vmax, shading='auto')
    else:
        im = ax.pcolormesh(K_grid, Omega_grid, spectral_data, 
                          cmap=colormap, shading='auto')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Intensity')
    
    # 设置频率范围
    if omega_range:
        ax.set_ylim(omega_range)
    
    # 设置轴标签和标题
    ax.set_xticks(k_tick_positions)
    ax.set_xticklabels(k_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_multiple_spectral_channels(k_distances: np.ndarray, omega: np.ndarray,
                                   spectral_data_list: List[np.ndarray],
                                   channel_names: List[str],
                                   k_tick_positions: List[float], k_labels: List[str],
                                   main_title: str = "Spectral Intensity",
                                   omega_range: Optional[Tuple[float, float]] = None,
                                   intensity_ranges: Optional[List[Tuple[float, float]]] = None,
                                   colormap: str = 'jet',
                                   figsize: Tuple[int, int] = (15, 10),
                                   save_path: Optional[str] = None) -> plt.Figure:
    """
    绘制多通道光谱强度图
    
    Parameters:
    -----------
    k_distances : ndarray
        k路径距离
    omega : ndarray
        频率数组
    spectral_data_list : list
        多个光谱强度数据的列表
    channel_names : list
        通道名称列表
    k_tick_positions : list
        高对称点位置
    k_labels : list
        高对称点标签
    main_title : str
        主标题
    omega_range : tuple, optional
        频率范围
    intensity_ranges : list, optional
        各通道的强度范围
    colormap : str
        颜色映射
    figsize : tuple
        图尺寸
    save_path : str, optional
        保存路径
        
    Returns:
    --------
    Figure
        matplotlib图对象
    """
    n_channels = len(spectral_data_list)
    
    # 计算子图布局
    if n_channels <= 2:
        nrows, ncols = 1, n_channels
    elif n_channels <= 4:
        nrows, ncols = 2, 2
    else:
        nrows = int(np.ceil(n_channels / 3))
        ncols = 3
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_channels == 1:
        axes = [axes]
    elif nrows == 1:
        axes = list(axes)
    else:
        axes = axes.flatten()
    
    # 创建网格
    K_grid, Omega_grid = np.meshgrid(k_distances, omega)
    
    for i, (spectral_data, channel_name) in enumerate(zip(spectral_data_list, channel_names)):
        ax = axes[i]
        
        # 设置强度范围
        if intensity_ranges and i < len(intensity_ranges):
            vmin, vmax = intensity_ranges[i]
            im = ax.pcolormesh(K_grid, Omega_grid, spectral_data, 
                              cmap=colormap, vmin=vmin, vmax=vmax, shading='auto')
        else:
            im = ax.pcolormesh(K_grid, Omega_grid, spectral_data, 
                              cmap=colormap, shading='auto')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Intensity', fontsize=14)
        cbar.ax.tick_params(labelsize=12)
        
        # 设置频率范围
        if omega_range:
            ax.set_ylim(omega_range)
        
        # 设置轴标签和标题
        ax.set_xticks(k_tick_positions)
        ax.set_xticklabels(k_labels, fontsize=14)
        ax.set_xlabel('', fontsize=14)
        ax.set_ylabel('ω/J', fontsize=14)
        ax.set_title(f'Spectral Intensity ({channel_name})', fontsize=16)
        ax.tick_params(axis='both', labelsize=12)
    
    # 隐藏多余的子图
    for i in range(n_channels, len(axes)):
        axes[i].set_visible(False)
    
    fig.suptitle(main_title, fontsize=18)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_2d_brillouin_zone(kx: np.ndarray, ky: np.ndarray, data: np.ndarray,
                          title: str = "2D Brillouin Zone",
                          xlabel: str = r"$k_x/\pi$", ylabel: str = r"$k_y/\pi$",
                          colormap: str = 'viridis',
                          figsize: Tuple[int, int] = (8, 6),
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    绘制二维布里渊区图
    
    Parameters:
    -----------
    kx, ky : ndarray
        动量网格
    data : ndarray
        要绘制的数据
    title : str
        图标题
    xlabel, ylabel : str
        轴标签
    colormap : str
        颜色映射
    figsize : tuple
        图尺寸
    save_path : str, optional
        保存路径
        
    Returns:
    --------
    Figure
        matplotlib图对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制等高线图
    im = ax.contourf(kx/np.pi, ky/np.pi, data, levels=50, cmap=colormap)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    
    # 设置轴标签和标题
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_3d_surface(k_distances: np.ndarray, omega: np.ndarray, spectral_data: np.ndarray,
                   k_tick_positions: List[float], k_labels: List[str],
                   title: str = "3D Spectral Function",
                   xlabel: str = "k-path", ylabel: str = "ω (meV)", zlabel: str = "S(k,ω)",
                   figsize: Tuple[int, int] = (12, 8),
                   elevation: float = 30, azimuth: float = 45,
                   save_path: Optional[str] = None) -> plt.Figure:
    """
    绘制三维光谱函数图
    
    Parameters:
    -----------
    k_distances : ndarray
        k路径距离
    omega : ndarray
        频率数组
    spectral_data : ndarray
        光谱数据
    k_tick_positions : list
        高对称点位置
    k_labels : list
        高对称点标签
    title : str
        图标题
    xlabel, ylabel, zlabel : str
        轴标签
    figsize : tuple
        图尺寸
    elevation, azimuth : float
        视角参数
    save_path : str, optional
        保存路径
        
    Returns:
    --------
    Figure
        matplotlib图对象
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # 创建网格
    K_grid, Omega_grid = np.meshgrid(k_distances, omega)
    
    # 绘制三维表面
    surf = ax.plot_surface(K_grid, Omega_grid, spectral_data, 
                          cmap='jet', alpha=0.8, edgecolor='none')
    
    # 添加颜色条
    fig.colorbar(surf, ax=ax, shrink=0.5)
    
    # 设置视角
    ax.view_init(elev=elevation, azim=azimuth)
    
    # 设置轴标签和标题
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    
    # 设置x轴刻度
    ax.set_xticks(k_tick_positions)
    ax.set_xticklabels(k_labels)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_saddle_point_convergence(iteration_data: List[float],
                                 title: str = "Saddle Point Convergence",
                                 ylabel: str = "Objective Function Value",
                                 figsize: Tuple[int, int] = (10, 6),
                                 save_path: Optional[str] = None) -> plt.Figure:
    """
    绘制鞍点优化收敛图
    
    Parameters:
    -----------
    iteration_data : list
        迭代数据
    title : str
        图标题
    ylabel : str
        y轴标签
    figsize : tuple
        图尺寸
    save_path : str, optional
        保存路径
        
    Returns:
    --------
    Figure
        matplotlib图对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.semilogy(iteration_data, 'b-o', linewidth=2, markersize=4)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_momentum_point_spectrum(data, L1, k_index=None, k_value=None, save_path=None, name=None):
    """
    绘制特定动量点的谱函数对频率的函数
    
    Parameters:
    -----------
    data : dict
        光谱数据
    L1 : int
        晶格大小
    k_index : int, optional
        k点的索引
    k_value : float, optional
        k路径上的距离值
    save_path : str, optional
        保存路径
    """
    
    k_distances = data['k_distances']
    omega_idx = data['omega_idx']
    spectral_intensity = data['spectral_intensity']  # (n_omega, n_points)
    k_tick_positions = data['k_tick_positions']
    k_tick_labels = data['k_tick_labels']
    
    # 确定k点索引
    if k_index is None:
        if k_value is None:
            # 使用第一个高对称点（X点）
            k_index = np.argmin(np.abs(k_distances - k_tick_positions[0]))
            k_value = k_distances[k_index]
            k_label = k_tick_labels[0]
        else:
            # 根据k_value找到最接近的索引
            k_index = np.argmin(np.abs(k_distances - k_value))
            k_value = k_distances[k_index]
            # 找到最接近的高对称点标签
            nearest_tick_idx = np.argmin(np.abs(k_tick_positions - k_value))
            k_label = k_tick_labels[nearest_tick_idx]
    else:
        k_value = k_distances[k_index]
        # 找到最接近的高对称点标签
        nearest_tick_idx = np.argmin(np.abs(k_tick_positions - k_value))
        k_label = k_tick_labels[nearest_tick_idx]
    
    # 提取特定k点的谱函数
    spectrum_at_k = spectral_intensity[:, k_index]
    
    # 绘图
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(omega_idx, spectrum_at_k, 'o-', linewidth=2, markersize=4, color='#e74c3c')
    ax.fill_between(omega_idx, spectrum_at_k, alpha=0.3, color='#e74c3c')
    
    ax.set_xlabel('Frequency ω', fontsize=12)
    ax.set_ylabel('Spectral Intensity', fontsize=12)
    ax.set_title(f'{name} Spectrum at k={k_label} (L={L1})', 
                 fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"单点谱函数已保存: {save_path}")
    
    plt.show()
    
    return fig, ax

def create_publication_ready_plots():
    """
    创建发表质量的图表样式设置
    """
    # 设置字体
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'text.usetex': False,  # 如果有LaTeX可以设为True
        'figure.figsize': [8, 6],
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'axes.linewidth': 1.2,
        'lines.linewidth': 2,
        'grid.alpha': 0.3,
        'legend.frameon': True,
        'legend.fancybox': True,
        'legend.shadow': True
    })

# 初始化发表质量设置
create_publication_ready_plots()