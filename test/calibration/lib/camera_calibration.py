import cv2
import numpy as np
import glob
import yaml
import os

def read_config(file_path):
    """读取配置文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"配置文件 {file_path} 不存在。")
    with open(file_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
        if config is None:
            config = {}  # 如果配置文件为空或格式错误，初始化为空字典
    return config

def calibrate_camera(image_folder, chessboard_size=(9, 6)):
    """
    使用棋盘格标定2D相机，返回相机的内参和畸变系数。
    """
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    objpoints = []  # 3D点（假定棋盘格在z=0的平面上）
    imgpoints = []  # 2D点（图像坐标系）

    # 获取图像文件夹中的所有图像
    images = glob.glob(f"{image_folder}\\*.jpg") + glob.glob(f"{image_folder}/*.png") + glob.glob(f"{image_folder}/*.jpeg") + glob.glob(f"{image_folder}/*.bmp")
    print(f"加载图像文件夹路径：{image_folder}")
    print(f"加载到的图像文件：{images}")  # 打印加载到的所有图像路径

    if not images:
        print(f"未找到棋盘格图片，检查路径 {image_folder}")
        return {"success": False, "message": "未找到棋盘格图片"}

    # 遍历每一张图片并检测角点
    for fname in images:
        print(f"正在加载图像: {fname}")
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 检测棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    if not objpoints or not imgpoints:
        return {"success": False, "message": "未能检测到有效的棋盘格角点"}

    # 标定相机
    ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    if ret:
        return {
            "success": True,
            "camera_matrix": camera_matrix.tolist(),
            "dist_coeffs": dist_coeffs.tolist(),
        }
    else:
        return {"success": False, "message": "相机标定失败"}

def get_or_calibrate_camera(config_path=r".\\calibration\\config.yaml", image_folder=r".\\calibration\\chessboard_images", chessboard_size=(9, 6)):
    """检查配置文件是否已标定，如果未标定则进行标定"""
    try:
        config = read_config(config_path)
    except FileNotFoundError:
        # 如果配置文件不存在，初始化默认配置
        config = {
            "calibration_status": {"calibrated": False},
            "camera_parameters": {"camera_matrix": None, "dist_coeffs": None},  # 初始化 camera_parameters
        }

    # 确保 calibration_status 存在
    if "calibration_status" not in config:
        config["calibration_status"] = {"calibrated": False}
    
    # 确保 camera_parameters 存在
    if "camera_parameters" not in config:
        config["camera_parameters"] = {"camera_matrix": None, "dist_coeffs": None}

    if not config.get("calibration_status", {}).get("calibrated", False):
        print("未标定，开始2D相机标定...")
        result = calibrate_camera(image_folder, chessboard_size)
        if result["success"]:
            config["calibration_status"]["calibrated"] = True
            config["camera_parameters"]["camera_matrix"] = result["camera_matrix"]
            config["camera_parameters"]["dist_coeffs"] = result["dist_coeffs"]
            # write_config(config_path, config)
            print("标定成功，参数已保存到配置文件。")
        else:
            print(f"标定失败：{result.get('message', '未知错误')}")  # 在标定失败时输出详细错误信息
            return None
    else:
        print("相机已标定，加载内参。")

    return {
        "camera_matrix": np.array(config["camera_parameters"]["camera_matrix"]),
        "dist_coeffs": np.array(config["camera_parameters"]["dist_coeffs"]),
    }

def undistort_image(image_path, camera_matrix, dist_coeffs):
    """对图像进行畸变校正"""
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
    
    # 裁剪图像
    x, y, w, h = roi
    undistorted_img = undistorted_img[y:y+h, x:x+w]
    return img, undistorted_img

def compare_images(image_path, camera_matrix, dist_coeffs):
    """比较原图和校正后的图像"""
    original_img, corrected_img = undistort_image(image_path, camera_matrix, dist_coeffs)

    # 使用 cv2.resize() 调整图像大小
    original_img_resized = cv2.resize(original_img, (600, 400))  # 设置为 600x400
    corrected_img_resized = cv2.resize(corrected_img, (600, 400))  # 设置为 600x400

    # 显示对比图像
    cv2.imshow("Before and After Undistortion", original_img_resized)
    cv2.imshow("Corrected Image", corrected_img_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # 获取标定参数
    calibration_data = get_or_calibrate_camera(r".\\calibration\\config.yaml", r".\\calibration\\chessboard_images")
    if calibration_data is None:
        print("标定失败，无法继续")
        return

    camera_matrix = calibration_data["camera_matrix"]
    dist_coeffs = calibration_data["dist_coeffs"]

    # 指定图像路径，进行校正前后对比
    image_path = "image\\test.jpg"
    compare_images(image_path, camera_matrix, dist_coeffs)

if __name__ == "__main__":
    main()
