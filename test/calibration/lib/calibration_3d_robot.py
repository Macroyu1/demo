import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize

# 读取并提取3D点和机器人坐标系点
def read_robot_points(xls_file):
    data = pd.read_excel(xls_file)
    camera_3d = data[['3DX', '3DY', '3DZ']].values
    robot_coords = data[['RobotX', 'RobotY', 'RobotZ']].values
    valid_indices = np.all(np.isfinite(camera_3d), axis=1) & np.all(np.isfinite(robot_coords), axis=1)
    return camera_3d[valid_indices], robot_coords[valid_indices]

# 使用最小二乘法计算 3D 坐标到机器人坐标系的转换关系
def least_squares_transform_3d_to_robot(camera_3d, robot_coords):
    camera_3d_homogeneous = np.hstack([camera_3d, np.ones((camera_3d.shape[0], 1))])  # (N, 4)
    A = camera_3d_homogeneous  # (N, 4)
    B = robot_coords  # (N, 3)
    try:
        transform_params, _, _, _ = np.linalg.lstsq(A, B, rcond=None)  # (4, 3)
        return transform_params
    except np.linalg.LinAlgError:
        print("最小二乘法计算失败，检查数据是否符合要求。")
        return None

# 计算误差（目标函数）
def compute_errors(transform_params, camera_3d, robot_coords):
    transform_matrix = transform_params.reshape(4, 3)  # 将参数恢复为4x3矩阵
    camera_3d_homogeneous = np.hstack([camera_3d, np.ones((camera_3d.shape[0], 1))])  # (N, 4)
    
    # 计算转换后的3D坐标
    transformed_3d = np.dot(camera_3d_homogeneous, transform_matrix)  # 注意这里是 camera_3d_homogeneous 与 transform_matrix 相乘

    # 计算误差（转换后的坐标与真实机器人坐标的差值）
    errors = transformed_3d - robot_coords  # (N, 3)

    # 返回误差的平方和（为了满足 least_squares 的要求）
    return np.sum(errors**2, axis=1)  # 返回每个点的误差平方和，形状为 (N,)

# 第一种优化方法：最小二乘法优化
def optimize_with_least_squares(camera_3d, robot_coords, initial_transform_params):
    result = least_squares(compute_errors, initial_transform_params.flatten(), args=(camera_3d, robot_coords))
    optimized_transform_params = result.x.reshape(4, 3)  # 得到优化后的变换矩阵
    return optimized_transform_params, result.fun  # 返回优化后的参数和误差

# 第二种优化方法：使用 `trust-constr` 方法进行优化
def optimize_with_trust_constr(camera_3d, robot_coords, initial_transform_params):
    def objective_function(params):
        return np.sum(compute_errors(params, camera_3d, robot_coords)**2)
    
    result = minimize(objective_function, initial_transform_params.flatten(), method='trust-constr')
    optimized_transform_params = result.x.reshape(4, 3)  # 得到优化后的变换矩阵
    return optimized_transform_params, result.fun  # 返回优化后的参数和误差

# 计算优化后的每个坐标轴的平均误差和最大误差
def compute_axis_errors(optimized_transform_params, camera_3d, robot_coords):
    # 计算优化后的误差
    errors = compute_errors(optimized_transform_params.flatten(), camera_3d, robot_coords)
    
    # 计算误差的平方和，计算每个点的误差
    transformed_3d = np.dot(np.hstack([camera_3d, np.ones((camera_3d.shape[0], 1))]), optimized_transform_params.reshape(4, 3))
    diff = transformed_3d - robot_coords

    # 计算每个坐标轴的平均误差和最大误差
    mean_error_x = np.mean(np.abs(diff[:, 0]))  # X轴平均误差
    mean_error_y = np.mean(np.abs(diff[:, 1]))  # Y轴平均误差
    mean_error_z = np.mean(np.abs(diff[:, 2]))  # Z轴平均误差
    
    max_error_x = np.max(np.abs(diff[:, 0]))  # X轴最大误差
    max_error_y = np.max(np.abs(diff[:, 1]))  # Y轴最大误差
    max_error_z = np.max(np.abs(diff[:, 2]))  # Z轴最大误差
    
    # 以数组的形式返回误差
    return np.array([mean_error_x, mean_error_y, mean_error_z, max_error_x, max_error_y, max_error_z])

# 主函数：计算并选择误差最小的优化方法
def robot_calibration_process(config_path, xls_path=r"./calibration/point.xls"):
    print("开始手眼标定")
    # 读取3D点和机器人坐标系的点
    camera_3d, robot_coords = read_robot_points(xls_path)

    # 初始的变换矩阵（可以是最小二乘法的初始值）
    initial_transform_params = least_squares_transform_3d_to_robot(camera_3d, robot_coords)

    if initial_transform_params is None:
        print("初始变换矩阵计算失败，请检查数据。")
        return None

    # 使用两种优化方法计算优化结果
    optimized_transform_params_ls, error_ls = optimize_with_least_squares(camera_3d, robot_coords, initial_transform_params)
    optimized_transform_params_tc, error_tc = optimize_with_trust_constr(camera_3d, robot_coords, initial_transform_params)

    # 计算两种方法的每个轴的平均误差和最大误差
    errors_ls = compute_axis_errors(optimized_transform_params_ls, camera_3d, robot_coords)
    errors_tc = compute_axis_errors(optimized_transform_params_tc, camera_3d, robot_coords)

    # 打印两种优化方法的误差信息
    print("最小二乘法优化结果:")
    print(errors_ls)
    
    print("\ntrust-constr优化结果:")
    print(errors_tc)

    # 选择误差最小的优化结果
    if np.mean(errors_ls[:3]) < np.mean(errors_tc[:3]):
        print("选择最小二乘法优化结果")
        return optimized_transform_params_ls
    else:
        print("选择 trust-constr 优化结果")
        return optimized_transform_params_tc

