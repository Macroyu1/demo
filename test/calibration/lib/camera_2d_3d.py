import numpy as np
import pandas as pd
import yaml
import cv2



# 读取并提取标定点
def read_calibration_points(xls_file):
    data = pd.read_excel(xls_file)
    camera_2d = data[['2DX', '2DY']].values
    camera_3d = data[['3DX', '3DY']].values
    valid_indices = np.all(np.isfinite(camera_2d), axis=1) & np.all(np.isfinite(camera_3d), axis=1)
    return camera_2d[valid_indices], camera_3d[valid_indices]

# 去畸变和归一化2D像素坐标
def undistort_and_normalize(camera_2d, K, dist_coeffs):
    camera_2d_undistorted = cv2.undistortPoints(camera_2d, K, dist_coeffs)
    return np.array([camera_2d_undistorted[i, 0, :] for i in range(camera_2d_undistorted.shape[0])])

# 使用最小二乘法计算 2D 到 3DX 和 3DY 的转换关系
def least_squares_transform(camera_2d, camera_3d):
    camera_2d_homogeneous = np.hstack([camera_2d, np.ones((camera_2d.shape[0], 1))])  # (N, 3)
    A = camera_2d_homogeneous  # (N, 3)
    B = camera_3d  # (N, 2)
    try:
        transform_params, _, _, _ = np.linalg.lstsq(A, B, rcond=None)  # (3, 2)
        return transform_params
    except np.linalg.LinAlgError:
        print("最小二乘法计算失败，检查数据是否符合要求。")
        return None

# 计算并打印误差信息，输出最后三个点的平均误差
def compute_and_print_errors(camera_3d, transformed_3d, camera_2d, camera_2d_homogeneous, transform_params):
    errors = camera_3d - transformed_3d  # 计算每个点在X和Y轴上的误差
    
    # 计算最后三个点的X轴和Y轴的平均误差
    last_three_indices = [-3, -2, -1]  # 最后3个点的索引
    last_three_errors = errors[last_three_indices]
    
    # X轴和Y轴的平均误差
    mean_error_x = np.mean(np.abs(last_three_errors[:, 0]))
    mean_error_y = np.mean(np.abs(last_three_errors[:, 1]))

    # 计算最大误差
    max_error_x = np.max(np.abs(errors[:, 0]))
    max_error_y = np.max(np.abs(errors[:, 1]))

    print(f"\nX轴平均误差: {mean_error_x:.4f}")
    print(f"Y轴平均误差: {mean_error_y:.4f}")
    print(f"X轴最大误差: {max_error_x:.4f}")
    print(f"Y轴最大误差: {max_error_y:.4f}")

    return mean_error_x, mean_error_y, max_error_x, max_error_y


# 主函数：计算 2D 相机到 3DX 和 3DY 的转换关系
def camera_calibration_process(config_path = r".\\calibration\\config.yaml",xls_path = r"./calibration/point.xls",K=None,dist_coeffs=None):
    print(K,dist_coeffs)

    # 读取标定点
    camera_2d, camera_3d = read_calibration_points(xls_path)

    # 去畸变和归一化 2D 像素坐标
    camera_2d_normalized = undistort_and_normalize(camera_2d, K, dist_coeffs)

    # 使用最小二乘法计算转换关系
    transform_params = least_squares_transform(camera_2d_normalized, camera_3d)
    if transform_params is None:
        print("标定过程失败，请检查输入数据和计算步骤。")
        return None

    # 打印转换矩阵
    print("\n计算得到的转换关系：")
    print(transform_params)

    # 计算预测的 3DX 和 3DY
    camera_2d_homogeneous = np.hstack([camera_2d_normalized, np.ones((camera_2d_normalized.shape[0], 1))])
    transformed_3d = camera_2d_homogeneous @ transform_params

    # 打印误差信息并返回
    mean_error_x, mean_error_y, max_error_x, max_error_y = compute_and_print_errors(camera_3d, transformed_3d, camera_2d, camera_2d_homogeneous, transform_params)

    # 更新配置文件
    # update_config(config_path, transform_params, mean_error_x, mean_error_y, max_error_x, max_error_y)

    return transform_params, (mean_error_x, mean_error_y, max_error_x, max_error_y)

# # # 主脚本执行部分
# config_path = r'.\\calibration\\config.yaml'  # 可以直接使用路径，或在yaml中读取
# transform_params = camera_calibration_process(config_path)
