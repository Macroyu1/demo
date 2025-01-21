import numpy as np
import pandas as pd
import cv2

# 模块1: 数据读取
def read_data(file_path):
    """读取 2D 像素坐标和相机坐标系下的 3D 点"""
    df = pd.read_excel(file_path)
    points_2d = df[['2DX', '2DY']].values  # 像素坐标
    points_3d = df[['3DX', '3DY', '3DZ']].values  # 相机坐标系下的 3D 点
    return points_2d, points_3d

# 模块2: 像素坐标 -> 图像坐标
def pixel_to_image_coords(points_2d, camera_matrix, dist_coeffs):
    """
    利用内参和畸变系数，将像素坐标转换为去畸变图像坐标。
    points_2d: 原始像素坐标 (N, 2)
    camera_matrix: 相机内参矩阵
    dist_coeffs: 畸变系数
    """
    points_2d = np.asarray(points_2d, dtype=np.float32).reshape(-1, 1, 2)  # (N, 1, 2)
    undistorted = cv2.undistortPoints(points_2d, camera_matrix, dist_coeffs, P=camera_matrix)
    return undistorted.reshape(-1, 2)  # 返回去畸变的图像坐标 (N, 2)

# 模块3: 计算外参矩阵
def compute_extrinsics(points_2d, points_3d, camera_matrix):
    """使用 solvePnP 计算外参 (旋转和平移)"""
    points_2d = np.asarray(points_2d, dtype=np.float32)  # 转换为 float32 类型
    points_3d = np.asarray(points_3d, dtype=np.float32)
    
    success, rvec, tvec = cv2.solvePnP(
        points_3d, points_2d, camera_matrix, None, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        raise ValueError("Failed to compute extrinsics using solvePnP.")
    rotation_matrix, _ = cv2.Rodrigues(rvec)  # 将旋转向量转换为旋转矩阵
    return rotation_matrix, tvec

# 模块4: 验证外参矩阵
def validate_projection(points_3d, points_2d_actual, camera_matrix, rotation_matrix, tvec):
    """
    验证通过外参矩阵将 3D 点投影到 2D 像素坐标系后的误差。
    """
    # 构造外参矩阵 [R|t]
    extrinsics = np.hstack((rotation_matrix, tvec))
    print(extrinsics,camera_matrix)
    # 将 3D 点投影到 2D 像素坐标系
    points_2d_pred, _ = cv2.projectPoints(points_3d, extrinsics[:, :3], extrinsics[:, 3], camera_matrix, None)
    points_2d_pred = points_2d_pred.squeeze()
    
    # 计算误差
    errors = np.linalg.norm(points_2d_actual - points_2d_pred, axis=1)
    # for i, (actual, predicted, error) in enumerate(zip(points_2d_actual, points_2d_pred, errors)):
    #     print(f"Point {i + 1}:")
    #     print(f"  Actual 2D: {actual}")
    #     print(f"  Predicted 2D: {predicted}")
    #     print(f"  Error: {error:.2f} pixels")
    # print(f"Average Projection Error: {np.mean(errors):.2f} pixels")

# 主函数
def main():
    file_path = "calibration/point.xls"

    # 相机内参矩阵和畸变系数
    camera_matrix = np.array([[3934.15, 0, 1219.89],
                               [0, 3697.26, 927.352],
                               [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([-0.0582, -0.6964, -0.003, 0.0175, 1.1209], dtype=np.float32)

    # 1. 读取数据
    points_2d, points_3d = read_data(file_path)

    # 2. 转换到图像坐标系
    points_2d_normalized = pixel_to_image_coords(points_2d, camera_matrix, dist_coeffs)

    # 3. 计算外参矩阵
    rotation_matrix, tvec = compute_extrinsics(points_2d_normalized, points_3d, camera_matrix)

    # 4. 验证外参矩阵的准确性
    validate_projection(points_3d, points_2d, camera_matrix, rotation_matrix, tvec)

if __name__ == "__main__":
    main()
