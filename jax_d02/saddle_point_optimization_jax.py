"""
鞍点优化模块

Author: ZhouChk
"""

import numpy as np
import jax.numpy as jnp
import nlopt
from typing import Tuple, Optional, Callable, List
import scipy.optimize as opt
from bogoliubov_transform_jax import Bogoliubov_transform_2_jax, Bogoliubov_constraint_jax, saddle_point_sum_jax
from gamma_functions import set_global_params

class SaddlePointOptimizer:
    """鞍点优化器类"""
    
    def __init__(self, S: float = 0.5, J1plus: float = 0.5, J2plus: float = 0.5, J3plus: float = 0.5):
        """
        初始化优化器
        
        Parameters:
        -----------
        S : float
            自旋量子数，默认0.5
        J1plus, J2plus, J3plus : float
            交换耦合参数
        """
        self.S = S
        self.k1 = None
        self.k2 = None
        self.Nsites = None
        self.h = None
        self.Q1 = None
        self.Q2 = None
        self.J1plus = J1plus
        self.J2plus = J2plus
        self.J3plus = J3plus
        self.test = 1  # 默认使用改进版本
        
    def set_lattice(self, k1: np.ndarray, k2: np.ndarray, h: float, Q1: float, Q2: float):
        """设置晶格参数"""
        self.k1 = k1
        self.k2 = k2
        self.Nsites = len(k1)
        self.h = h
        self.Q1 = Q1
        self.Q2 = Q2
        
    def saddle_point_eq_number(self, lambda_param: float, A1: complex, B1: complex) -> float:
        """
        粒子数约束方程
        
        Parameters:
        -----------
        lambda_param : float
            拉格朗日乘数
        A1 : complex
            鞍点参数A1
        B1 : float
            鞍点参数B1
            
        Returns:
        --------
        float
            约束方程的值
        """
        A2 = A1
        A3 = A1
        B2 = -B1
        B3 = B1

        # res_eq=Bogoliubov_constraint_jax(0, self.k1, self.k2, self.Q1, self.Q2, A1, A2, A3, B1, B2, B3, lambda_param, self.h)
        # print("check eq number constraint:", res_eq, lambda_param)
        # print("A1:", A1, "B1:", B1, "lambda:", lambda_param)
        # print("--------------------------------------------------")

        Ubov = Bogoliubov_transform_2_jax(
            0, self.k1, self.k2, self.Q1, self.Q2, A1, A2, A3, B1, B2, B3, lambda_param, self.h,
            self.J1plus, self.J2plus, self.J3plus)[0]
        # print("--------------------------------------------------")
            
        lam = saddle_point_sum_jax(Ubov, self.k1, self.k2, self.Q1, self.Q2)[0]
        z = (lam - 2.0 * self.S)**2
            
        return z
    
    def saddle_point_constraint(self, lambda_param: float, A1: complex, B1: complex) -> List[float]:
        """
        鞍点约束条件 (确保所有本征值为正)
        
        Parameters:
        -----------
        lambda_param : float
            拉格朗日乘数
        A1 : complex
            鞍点参数A1
        B1 : float
            鞍点参数B1
            
        Returns:
        --------
        List[float]
            约束条件值 (应该 <= 0)
        """
        A2 = A1
        A3 = A1
        B2 = -B1
        B3 = B1
        
        con_eig = Bogoliubov_constraint_jax(0, self.k1, self.k2, self.Q1, self.Q2, A1, A2, A3, B1, B2, B3, 
                                        lambda_param, self.h, self.J1plus, self.J2plus, self.J3plus)
        
        # 约束容差调节策略
        # ========================================
        # 约束形式: con_eig ≥ tolerance  =>  constraint = tolerance - con_eig ≤ 0
        # 
        # tolerance 的物理意义:
        # - 确保 Bogoliubov 哈密顿量正定（最小本征值 > 0）
        # - 值越大，约束越严格，但数值求解越困难
        # - 值越小，数值稳定，但可能允许非物理解
        #
        # 推荐的 tolerance 设置:
        # 1. 固定容差（简单但可能过严）:
        #    tolerance = 1e-6  # 适用于 Nsites < 50
        #    tolerance = 1e-7  # 适用于 Nsites = 100-200
        #    tolerance = 1e-8  # 适用于 Nsites > 500
        #
        # 2. 自适应容差（推荐，随系统规模调整）:
        #    当前使用: max(1e-8, min(1e-5, 1e-4/sqrt(Nsites)))
        #    - Nsites=100: tolerance ≈ 1e-5
        #    - Nsites=400: tolerance ≈ 5e-6
        #    - Nsites=10000: tolerance ≈ 1e-6
        #
        # 3. 放宽约束（如果优化困难收敛）:
        #    tolerance = max(1e-7, 1e-4 / self.Nsites)
        #    或调节 SLSQP 的 eps 参数（见 saddle_point_optimization_number）
        #
        # 4. 约束缩放（提高 SLSQP 敏感度）:
        #    可返回 [scale_factor * (tolerance - con_eig)]
        #    其中 scale_factor = 1e3 到 1e5
        # ========================================
        
        # 当前配置（自适应容差）
        # tolerance = max(1e-8, 1e-5 / np.sqrt(self.Nsites))
        tolerance = 1e-5 / np.sqrt(self.Nsites)
        # tolerance = 1e-2
        
        # 替代配置示例（取消注释以使用）:
        # tolerance = 1e-6  # 固定容差
        # tolerance = max(1e-7, 1e-4 / self.Nsites)  # 放宽约束
        
        c = [float(tolerance - con_eig)]
        # print(f"  当前约束最小本征值: {con_eig:.6e}, 设定容差: {tolerance:.6e}, 约束值: {c[0]:.6e}")
        
        # 可选：约束缩放（提高 SLSQP 对约束违反的敏感度）
        # scale_factor = 1e4
        # c = [float(scale_factor * (tolerance - con_eig))]
            
        return c
    
    def saddle_point_gapless_condition(self, A1: complex, B1: complex) -> float:
        """
        无能隙条件
        
        Parameters:
        -----------
        A1 : complex
            鞍点参数A1
        B1 : float
            鞍点参数B1
            
        Returns:
        --------
        float
            无能隙条件的lambda值
        """
        def constraint_max(lambda_param):
            constraints = self.saddle_point_constraint(lambda_param, A1, B1)
            return max(constraints)
        
        # lambda_gapless = opt.brentq(constraint_max, 0.5, 1.5)
        lambda_gapless = opt.fsolve(constraint_max, 0.9685)[0]
        # try:
        #     lambda_gapless = opt.brentq(constraint_max, 0.5, 1.5)
        # except ValueError:
        #     # 如果brentq失败，尝试其他方法
        #     lambda_gapless = opt.fsolve(constraint_max, 0.9685)[0]
        
        return lambda_gapless
    
    def saddle_point_optimization_number(self, x0: float, A1: complex, B1: complex) -> float:
        """
        优化拉格朗日乘数lambda
        
        Parameters:
        -----------
        x0 : float
            初始猜测
        A1 : complex
            鞍点参数A1
        B1 : float
            鞍点参数B1
            
        Returns:
        --------
        float
            优化的lambda值
        """
        
            
        def scipy_constraint(lambda_param):
            constraint_val = self.saddle_point_constraint(lambda_param, A1, B1)[0]
            return -constraint_val
            
        constraint = {'type': 'ineq', 'fun': scipy_constraint}
        
        # ========================================
        result = opt.minimize(
            lambda x: self.saddle_point_eq_number(x, A1, B1),
            x0,
            method='trust-constr',
            constraints=constraint,
            options={'xtol': 1e-15, 'gtol': 1e-15, 'maxiter': 1000}
        )
        lambda_opt = result.x
        # ========================================

        # print("+++++++++++++++++++++++++++++++++++")
        # print("x0 for lambda optimization:", x0)
        # result = opt.minimize(
        #     lambda x: self.saddle_point_eq_number(x, A1, B1),
        #     x0,
        #     method='COBYLA',
        #     constraints=constraint,
        #     options={
        #         'rhobeg': 0.05,     #初始步长（约为 lambda 的 5%）
        #         'tol': 1e-15,        #收敛容差（高精度）
        #         'catol': 0,      #约束容差（比 tolerance 严格）
        #         'maxiter': 1000,    #最大迭代次数
        #         'disp': False       #不显示详细信息
        #     }
        # )
        
        # lambda_opt = result.x
        # print("lam for lambda optimization:", lambda_opt)
        # check=scipy_constraint(lambda_opt)
        # print("Constraint check after COBYLA:", check)
        # if check > 0:
        #     print("-----------------------------------")
        #     print("x0 for lambda optimization:", x0)
        #     lambda_opt = x0
        #     print(f"Constraint violated after COBYLA: {check}, trying trust-constr...")
        #     result_tc = opt.minimize(
        #         lambda x: self.saddle_point_eq_number(x, A1, B1),
        #         x0,
        #         method='trust-constr',
        #         constraints=constraint,
        #         options={'xtol': 1e-15, 'gtol': 1e-15, 'maxiter': 1000}
        #     )
        #     lambda_opt = result_tc.x
        #     print("lam for lambda optimization:", lambda_opt)
        
        # SLSQP 约束容差调节说明:
        # ========================================
        # 1. ftol: 目标函数收敛容差（默认1e-6）
        #    - 控制何时停止优化（相邻迭代目标函数变化 < ftol）
        #    - 值越小，收敛越精确，但可能导致过多迭代
        #    - 推荐: 1e-10 到 1e-12（平衡精度与效率）
        #
        # 2. eps: 梯度有限差分步长（默认sqrt(machine_eps)≈1.5e-8）
        #    - 影响约束函数梯度的数值计算精度
        #    - 值越小，梯度越精确，但数值误差可能放大
        #    - 推荐: 1e-8（对于约束条件值在1e-5级别时）
        #
        # 3. 约束违反容差:
        #    - SLSQP 内部约束违反容差约为 sqrt(machine_eps) ≈ 1e-8
        #    - 无法直接设置，但可通过以下方式间接调节:
        #      a) saddle_point_constraint 中的 tolerance 参数
        #      b) 约束函数返回值的缩放（如乘以系数）
        #
        # 4. 当前配置适用场景:
        #    - 约束值在 1e-5 到 1e-8 量级
        #    - 目标函数为二次型（近似线性）
        #    - Nsites = 100, tolerance = 1e-5
        # ========================================
        
        # result = opt.minimize(
        #     lambda x: self.saddle_point_eq_number(x, A1, B1),
        #     x0,
        #     method='SLSQP',
        #     constraints=constraint,
        #     options={
        #         'ftol': 1e-10,      # 目标函数容差（适中，避免过拟合）
        #         'eps': 1e-8,        # 梯度有限差分步长
        #         'maxiter': 1000,    # 最大迭代次数
        #         'disp': False,      # 不显示优化过程
        #         'iprint': 1         # 静默模式（0=无输出，1=每次迭代，2=详细）
        #     }
        # )
        
        # ========================================
        # result = opt.minimize(
        #     lambda x: self.saddle_point_eq_number(x, A1, B1),
        #     x0,
        #     method='trust-constr',
        #     constraints=constraint,
        #     options={'xtol': 1e-15, 'gtol': 1e-15, 'maxiter': 1000}
        # )
        # ========================================

        # COBYLA 参数说明:
        # ========================================
        # rhobeg: 初始信赖域半径（初始搜索步长）
        #   - 控制第一步搜索的范围
        #   - 对于 lambda ∈ [0.9, 1.1]，推荐 0.05-0.1
        #   - 值太大会导致初始步骤过激
        #   - 值太小会导致搜索过于局部化
        #
        # tol: 收敛容差（最小信赖域半径）
        #   - 当搜索半径 < tol 时停止
        #   - 推荐 1e-6 到 1e-8
        #   - 决定最终解的精度
        #
        # catol: 约束容差（可选）
        #   - 约束违反量 < catol 认为满足约束
        #   - 默认约为 1e-4
        #   - 对于 tolerance=1e-5 的问题，建议设置 1e-6
        #
        # 当前配置适用于:
        #   - lambda 初值约 1.0，变化范围 ±10%
        #   - 约束容差 1e-5 级别
        #   - 需要稳健的无梯度优化
        # ========================================
        
        # result = opt.minimize(
        #     lambda x: self.saddle_point_eq_number(x, A1, B1),
        #     x0,
        #     method='COBYLA',
        #     constraints=constraint,
        #     options={
        #         'rhobeg': 0.05,     # 初始步长（约为 lambda 的 5%）
        #         'tol': 1e-15,        # 收敛容差（高精度）
        #         'catol': 0,      # 约束容差（比 tolerance 严格）
        #         'maxiter': 1000,    # 最大迭代次数
        #         'disp': False       # 不显示详细信息
        #     }
        # )
        
        # lambda_opt = result.x

        # # 目标函数
        # def objective(lambda_param):
        #     return self.saddle_point_eq_number(lambda_param[0], A1, B1)
        
        # # 约束函数
        # def constraint_func(lambda_param, grad):
        #     if grad.size > 0:
        #         # 数值梯度 (nlopt需要)
        #         eps = 1e-8
        #         grad[0] = (self.saddle_point_constraint(lambda_param[0] + eps, A1, B1)[0] - 
        #                   self.saddle_point_constraint(lambda_param[0] - eps, A1, B1)[0]) / (2 * eps)
        #     constraints = self.saddle_point_constraint(lambda_param[0], A1, B1)
        #     return constraints[0]
        
        # opt_nlopt = nlopt.opt(nlopt.LN_COBYLA, 1)
        # opt_nlopt.set_min_objective(lambda x, grad: objective(x))
        # opt_nlopt.add_inequality_constraint(constraint_func, 0)
        # opt_nlopt.set_xtol_rel(1e-20)  # 放宽收敛条件，避免过早停止
        # opt_nlopt.set_ftol_rel(1e-20)
        # opt_nlopt.set_maxeval(1000)  # 增加最大迭代次数
            
        # lambda_opt = opt_nlopt.optimize([x0])[0]
        
        # 优化后再次验证约束
        final_constraint = self.saddle_point_constraint(lambda_opt, A1, B1)[0]
        if final_constraint > 0:
            print(f"  [WARNING] Final lambda violates constraint: c={final_constraint:.6e}")
            print(f"  Attempting to use gapless condition lambda instead...")
            # 如果违反约束，使用gapless条件的lambda
            x0_gapless = self.saddle_point_gapless_condition(A1, B1)
            lambda_opt = x0_gapless
        
        # 使用nlopt优化
        # try:
        #    opt_nlopt = nlopt.opt(nlopt.LN_COBYLA, 1)
        #    opt_nlopt.set_min_objective(lambda x, grad: objective(x))
        #    opt_nlopt.add_inequality_constraint(constraint_func, 1e-8)
        #    opt_nlopt.set_xtol_rel(1e-15)
        #    opt_nlopt.set_ftol_rel(1e-15)
        #    opt_nlopt.set_maxeval(1000)
           
        #    lambda_opt = opt_nlopt.optimize([x0])[0]
        # except:
        #     # 备选: 使用scipy优化
        #     print("NLopt failed, using scipy optimization")
            
        #     def scipy_constraint(lambda_param):
        #         return -np.array(self.saddle_point_constraint(lambda_param, A1, B1))
            
        #     constraint = {'type': 'ineq', 'fun': scipy_constraint}
            
        #     result = opt.minimize(
        #         lambda x: self.saddle_point_eq_number(x, A1, B1),
        #         x0,
        #         method='SLSQP',
        #         constraints=constraint,
        #         options={'ftol': 1e-15, 'xtol': 1e-15, 'maxiter': 1000}
        #     )

        
        # print("+++++++++++++++++++++++++++++++++++")
        # def scipy_constraint(lambda_param):
        #     return -np.array(self.saddle_point_constraint(lambda_param, A1, B1))
            
        # constraint = {'type': 'ineq', 'fun': scipy_constraint}
            
        # result = opt.minimize(
        #     lambda x: self.saddle_point_eq_number(x, A1, B1),
        #     [x0],  # 确保传入列表/数组
        #     method='SLSQP',
        #     constraints=constraint,
        #     options={'ftol': 1e-15, 'maxiter': 1000}  # 移除 xtol（SLSQP 不支持）
        # )
            
        # lambda_opt = result.x
        # 确保返回标量
        if isinstance(lambda_opt, np.ndarray):
            lambda_opt = float(lambda_opt[0] if len(lambda_opt) > 0 else lambda_opt)
        else:
            lambda_opt = float(lambda_opt)
        
        # print(f"  [DEBUG] Optimized lambda: {lambda_opt:.6f}, type: {type(lambda_opt)}")
        return lambda_opt
    
    def saddle_point_eq_auxiliary_fields(self, x: np.ndarray) -> float:
        """
        辅助场的自洽方程
        
        Parameters:
        -----------
        x : ndarray
            [A1_imag, B1] 参数向量
            
        Returns:
        --------
        float
            自洽方程的值
        """
        A1 = 1j * x[0]
        A2 = A1
        A3 = A1
        B1 = x[1]
        B2 = -B1
        B3 = B1
        # print(f"aux: Current A1: {A1}, B1: {B1}")
        
        # 找到无能隙条件
        x0 = self.saddle_point_gapless_condition(A1, B1)
        # print(f"x0 for lambda: {x0}, gapless lambda: {x0}")
        
        # 优化lambda
        lambda_param = self.saddle_point_optimization_number(x0 + 0.01, A1, B1)
        # print(f"x0 for lambda: {x0}, optimized lambda: {lambda_param}")
        
        # res = Bogoliubov_constraint(0, self.k1, self.k2, self.Q1, self.Q2, A1, A2, A3, B1, B2, B3, lambda_param, self.h)
        
        # # 检查约束是否满足（最小本征值必须为正）
        # print(f"Checking constraint: min eigenvalue = {res:.6e} for A1={A1}, B1={B1}, lambda={lambda_param}")
        # if res < 0:
        #     print(f"[ERROR] Negative eigenvalue detected: res={res:.6e}")
        #     print(f"Parameters: A1={A1}, B1={B1}, lambda={lambda_param}")
        #     print("++++++++++++++++++++++++++++++++++++++++++++++++++")
        #     raise ValueError(f"Constraint violation: minimum eigenvalue is negative (res={res:.6e}). "
        #                    f"Hamiltonian is not positive-definite. Optimization terminated.")
        
        # 使用改进的Bogoliubov变换
        Ubov = Bogoliubov_transform_2_jax(
             0, self.k1, self.k2, self.Q1, self.Q2, A1, A2, A3, B1, B2, B3, lambda_param, self.h,
             self.J1plus, self.J2plus, self.J3plus)[0]
            
        AA, BB = saddle_point_sum_jax(Ubov, self.k1, self.k2, self.Q1, self.Q2)[1:3]  # 修复：[1:3]返回2个元素
        # print(f"Current A1: {A1}, B1: {B1}, lambda: {lambda_param}, AA: {AA}, BB: {BB}")
        
        # 转换JAX数组为Python标量
        AA = complex(AA)
        BB = float(np.real(BB))  # BB应该是实数
            
        y = np.zeros(2)
        y[0] = np.imag(AA - A1)
        y[1] = BB - B1
        
        z = np.dot(y, y)
        # print(f"+++++++++++++++++++++++++++++++++++++++++++")
        return z
    
    def saddle_point_optimization(self, x0: np.ndarray) -> Tuple[complex, complex, complex, complex, complex, complex, float]:
        """
        鞍点优化主函数
        
        Parameters:
        -----------
        x0 : ndarray
            初始猜测 [A1_imag, B1]
            
        Returns:
        --------
        tuple
            (A1, A2, A3, B1, B2, B3, lambda) - 优化的鞍点参数
        """
        # 用于跟踪优化进度
        self.iteration_count = 0
        self.best_value = float('inf')
        
        def objective_with_progress(x):
            """带进度显示的目标函数"""
            self.iteration_count += 1
            obj_value = self.saddle_point_eq_auxiliary_fields(x)
            
            # 更新最佳值
            if obj_value < self.best_value:
                self.best_value = obj_value
                print(f"  [Iteration {self.iteration_count}] New best: A1={1j*x[0]:.6f}, B1={x[1]:.6f}, obj={obj_value:.6e}")
            elif self.iteration_count % 10 == 0:
                print(f"  [Iteration {self.iteration_count}] Current: A1={1j*x[0]:.6f}, B1={x[1]:.6f}, obj={obj_value:.6e}")
            
            return obj_value
        # 目标函数
        # def objective(x, grad):
        #     if grad.size > 0:
        #         # 数值梯度
        #         eps = 1e-8
        #         f0 = self.saddle_point_eq_auxiliary_fields(x)
        #         for i in range(len(x)):
        #             x_plus = x.copy()
        #             x_plus[i] += eps
        #             grad[i] = (self.saddle_point_eq_auxiliary_fields(x_plus) - f0) / eps
        #     return self.saddle_point_eq_auxiliary_fields(x)
        
        # # 使用nlopt优化
        # try:
        #     opt_nlopt = nlopt.opt(nlopt.LN_NELDERMEAD, 2)
        #     opt_nlopt.set_min_objective(objective)
        #     opt_nlopt.set_xtol_rel(1e-20)
        #     opt_nlopt.set_ftol_rel(1e-20)
        #     opt_nlopt.set_maxeval(10000)
            
        #     x_opt = opt_nlopt.optimize(x0)
        # except:
        #     # 备选: 使用scipy优化
        #     print("NLopt failed, using scipy optimization")
        #     result = opt.minimize(
        #         self.saddle_point_eq_auxiliary_fields,
        #         x0,
        #         # method='Nelder-Mead',
        #         method='SLSQP',
        #         options={'xatol': 1e-20, 'fatol': 1e-20, 'maxiter': 10000}
        #     )
        #     x_opt = result.x
        
        # 使用 scipy 优化（无约束，因为约束已在内部 saddle_point_eq_auxiliary_fields 中处理）
        print("\n" + "="*60)
        print("开始鞍点参数优化 (A1, B1)")
        print("="*60)
        print(f"初始猜测: A1={1j*x0[0]:.6f}, B1={x0[1]:.6f}")
        print("-"*60)
        
        result = opt.minimize(
                objective_with_progress,
                x0,
                method='Nelder-Mead',  # 使用 Nelder-Mead，不需要梯度
                options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 10000, 'disp': True}
            )
        
        print("-"*60)
        print(f"优化完成:")
        print(f"  总迭代次数: {self.iteration_count}")
        print(f"  函数评估次数: {result.nfev}")
        print(f"  收敛状态: {'成功' if result.success else '失败'}")
        print(f"  最终目标函数值: {result.fun:.6e}")
        if not result.success:
            print(f"  失败原因: {result.message}")
        print("="*60 + "\n")
        
        # 提取优化结果
        x_opt = result.x
        A1 = 1j * x_opt[0]
        A2 = A1
        A3 = A1
        B1 = x_opt[1]
        B2 = -B1
        B3 = B1
        print("Optimized A1:", A1)
        print("Optimized B1:", B1)
        
        # 计算最优lambda
        x0_lambda = self.saddle_point_gapless_condition(A1, B1)
        lambda_opt = self.saddle_point_optimization_number(x0_lambda + 0.01, A1, B1)
        
        return A1, A2, A3, B1, B2, B3, lambda_opt

def optimize_saddle_point(k1: np.ndarray, k2: np.ndarray, h: float, Q1: float, Q2: float, 
                         x0: np.ndarray, S: float = 0.5, 
                         J1plus: float = 0.5, J2plus: float = 0.5, J3plus: float = 0.5) -> Tuple[complex, complex, complex, complex, complex, complex, float]:
    """
    便利函数：执行鞍点优化
    
    Parameters:
    -----------
    k1, k2 : ndarray
        动量网格
    h : float
        对称破缺场
    Q1, Q2 : float
        磁序波矢
    x0 : ndarray
        初始猜测 [A1_imag, B1]
    S : float
        自旋量子数
    J1plus, J2plus, J3plus : float
        交换耦合参数
        
    Returns:
    --------
    tuple
        优化的鞍点参数
    """
    # 设置全局参数
    print("Setting global parameters for optimization...")
    set_global_params(J1plus=J1plus, J2plus=J2plus, J3plus=J3plus, Q1=Q1, Q2=Q2)
    
    # 创建优化器
    optimizer = SaddlePointOptimizer(S=S, J1plus=J1plus, J2plus=J2plus, J3plus=J3plus)
    optimizer.set_lattice(k1, k2, h, Q1, Q2)

    # A1 = 0.567 * 1j
    # A2 = 0.567 * 1j
    # A3 = 0.567 * 1j
    # B1 = 0.3
    # B2 = -0.3
    # B3 = 0.3
    # lambda_val=1.02808437
    
    # print("Constraint check:", optimizer.saddle_point_constraint(lambda_val, A1, B1))
    # print("=============================================")
    # lambda_val = optimizer.saddle_point_gapless_condition(A1, B1)
    # print("x0_lambda:", lambda_val)
    # print("=============================================")
    # lambda_opt = optimizer.saddle_point_optimization_number(lambda_val + 0.01, A1, B1)

    # res=Bogoliubov_constraint(0, k1, k2, Q1, Q2, A1, A2, A3, B1, B2, B3, lambda_val, 1/100)
    # print("=============================================")
    # de_lambda = optimizer.saddle_point_optimization_number(lambda_val+0.01, A1, B1)
    # print("de_lambda:", de_lambda)
    # res = optimizer.saddle_point_eq_auxiliary_fields(np.array([0.54, 0.31]))
    # print("res:", res)
    # print("=============================================")

    print("strarting saddle point optimization...")
    x_opt=optimizer.saddle_point_optimization(x0)
    # print("=============================================")
    print("Optimization complete: ", x_opt)
    
    return x_opt