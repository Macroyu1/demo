import numpy as np
import pandas as pd
import cv2
import glob
import yaml

def calibrate_camera_optimized(chessboard_size, square_size, image_folder):
    # 获取图片路径
    images = glob.glob(f"{image_folder}/*.jpg")
    if not images:
        raise FileNotFoundError(f"No images found in folder: {image_folder}")

    obj_points = []  # 3D 点
    img_points = []  # 2D 点

    # 准备棋盘格的 3D 点
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    gray_shape = None

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 检测棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            # 亚像素角点优化
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners)
            gray_shape = gray.shape[::-1]  # 记录图像尺寸

    # 检查是否检测到任何角点
    if not obj_points or gray_shape is None:
        raise ValueError("No valid chessboard corners were detected in the images.")

    # 相机标定
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray_shape, None, None
    )

    # 返回相机内参和畸变系数
    return camera_matrix, dist_coeffs

def read_data(file_path):
    """读取 2D 像素坐标和相机坐标系下的 3D 点"""
    df = pd.read_excel(file_path)
    points_2d = df[['2DX', '2DY']].values  # 像素坐标
    points_3d_camera = df[['3DX', '3DY', '3DZ']].values  # 相机坐标系下的 3D 点
    points_3d_robot = df[['RobotX', 'RobotY', 'RobotZ']].values  # 机器人坐标系下的 3D 点
    return points_2d, points_3d_camera, points_3d_robot

def pixel_to_image_coords(points_2d, camera_matrix, dist_coeffs):
    points_2d = np.asarray(points_2d, dtype=np.float32).reshape(-1, 1, 2)  # (N, 1, 2)
    undistorted = cv2.undistortPoints(points_2d, camera_matrix, dist_coeffs, P=camera_matrix)
    return undistorted.reshape(-1, 2)  # 返回去畸变的图像坐标 (N, 2)

def compute_extrinsics(points_2d, points_3d, camera_matrix, dist_coeffs):
    """使用 solvePnP 计算外参 (旋转和平移)"""
    points_2d = np.asarray(points_2d, dtype=np.float32)  # 转换为 float32 类型
    points_3d = np.asarray(points_3d, dtype=np.float32)
    
    success, rvec, tvec = cv2.solvePnP(
        points_3d, points_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        raise ValueError("Failed to compute extrinsics using solvePnP.")
    rotation_matrix, _ = cv2.Rodrigues(rvec)  # 将旋转向量转换为旋转矩阵

    RT_2D_3D = np.hstack((rotation_matrix, tvec))  # 构造外参矩阵 [R|t]
    return RT_2D_3D

def validate_projection(points_3d, points_2d_actual, camera_matrix, extrinsics, dist_coeffs):
    """验证投影误差，返回平均误差"""
    points_2d_pred, _ = cv2.projectPoints(points_3d, extrinsics[:, :3], extrinsics[:, 3], camera_matrix, dist_coeffs, None)
    points_2d_pred = points_2d_pred.squeeze()

    # 计算误差
    errors = np.linalg.norm(points_2d_actual - points_2d_pred, axis=1)
    average_error = np.mean(errors)
    print(f"Average Projection Error: {average_error:.2f} pixels")
    return average_error

def compute_camera_to_robot_transform(points_3d_camera, points_3d_robot):
    """使用最小二乘法计算从相机坐标系到机器人坐标系的转换矩阵 (4x4)"""
    points_3d_camera = np.hstack((points_3d_camera, np.ones((points_3d_camera.shape[0], 1))))  # 转齐次坐标
    points_3d_robot = np.asarray(points_3d_robot, dtype=np.float64)

    # 使用最小二乘法拟合
    transform_matrix, _, _, _ = np.linalg.lstsq(points_3d_camera, points_3d_robot, rcond=None)

    # 构造 4x4 齐次变换矩阵
    transform_matrix = np.vstack((transform_matrix.T, [0, 0, 0, 1]))
    return transform_matrix

def write_config(file_path, camera_matrix, dist_coeffs, RT_2D_3D, RT_3D_Robot):
    """更新配置文件，添加文件头和尾注释"""
    with open(file_path, 'w', encoding='utf-8') as file:
        # 输出文件开始
        file.write("################################\n")  # 使用 file.write() 输出文本
        file.write("# 输出文件开始\n")  # 直接写文本注释
        file.write("################################\n")  # 输出分隔线
        
        # 写入文件的主要内容
        file.write("# 标定程序配置文件\n")  # 标题注释
        config = {}
        config["camera_matrix"] = camera_matrix.tolist()
        config["dist_coeffs"]   = dist_coeffs.tolist()
        config["RT_2D_3D"]      = RT_2D_3D.tolist()
        config["RT_3D_Robot"]   = RT_3D_Robot.tolist()


        yaml.dump(config, file, default_flow_style=False, allow_unicode=True)  # 输出实际配置内容
        print("结果已保存至config.yaml")
        # 输出文件结束
        file.write("################################\n")  # 分隔线
        file.write("# 输出文件结束\n")  # 结束注释
        file.write("################################\n\n\n")  # 结束分隔线

def main():
    file_path = "calibration/point.xls"
    config_path = r".\\calibration\\config.yaml"
    chessboard_size = (6, 9)  # 棋盘格内角点数
    square_size = 0.03  # 单个格子的边长（单位：米）
    image_folder = "calibration/chessboard_images"  # 棋盘格图片文件夹路径

    camera_matrix, dist_coeffs = calibrate_camera_optimized(chessboard_size, square_size, image_folder)

    # camera_matrix = np.array([[8.05626858e+03 ,0.00000000e+00 ,1.22427677e+03],
    #                         [0.00000000e+00, 9.11662061e+03, 1.02881568e+03],
    #                         [0.00000000e+00 ,0.00000000e+00 ,1.00000000e+00]], dtype=np.float32)
    # dist_coeffs = np.array([ 2.67832446e+00, -3.94011761e+02,  1.77625701e-01,  2.53417358e-02,  1.96795897e+04], dtype=np.float32)


    points_2d, points_3d_camera, points_3d_robot = read_data(file_path)
    points_2d_normalized = pixel_to_image_coords(points_2d, camera_matrix, dist_coeffs)

    RT_2D_3D = compute_extrinsics(points_2d_normalized, points_3d_camera, camera_matrix, dist_coeffs)
    validate_projection(points_3d_camera, points_2d, camera_matrix, RT_2D_3D, dist_coeffs)
    
    RT_3D_Robot = compute_camera_to_robot_transform(points_3d_camera, points_3d_robot)

    print("camera_matrix = ",camera_matrix)
    print("dist_coeffs = " , dist_coeffs)
    print("RT_Camera = " , RT_2D_3D)
    print("RT_Robot = ",RT_3D_Robot)
    
    write_config(config_path, camera_matrix, dist_coeffs, RT_2D_3D, RT_3D_Robot)
    
if __name__ == "__main__":
    main()




 # # 相机内参矩阵和畸变系数
    # camera_matrix = np.array([[8.05626858e+03 ,0.00000000e+00 ,1.22427677e+03],
    #                         [0.00000000e+00, 9.11662061e+03, 1.02881568e+03],
    #                         [0.00000000e+00 ,0.00000000e+00 ,1.00000000e+00]], dtype=np.float32)
    # dist_coeffs = np.array([ 2.67832446e+00, -3.94011761e+02,  1.77625701e-01,  2.53417358e-02,  1.96795897e+04], dtype=np.float32)
