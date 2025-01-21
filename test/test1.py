import pandas as pd
import numpy as np

def main():
    # 第1步：从Excel文件加载2D和3D点
    file_path = "calibration/point.xls"
    df = pd.read_excel(file_path)
    
    # 第2步：提取2D和3D点
    points_2d = df[['2DX', '2DY']].values  # 2D像素坐标
    points_3d = df[['3DX', '3DY', '3DZ']].values  # 3D坐标（相机坐标系下）

    # 将2D和3D点转换为NumPy数组（类型为float32）
    points_2d = np.asarray(points_2d, dtype=np.float32)
    points_3d = np.asarray(points_3d, dtype=np.float32)

    # 第3步：使用最小二乘法计算变换矩阵
    # 在2D点中添加一个1的列，以表示仿射变换中的平移
    ones = np.ones((points_2d.shape[0], 1))
    points_2d_homogeneous = np.hstack((points_2d, ones))  # 同次坐标
    points_3d_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))  # 转齐次坐标
    print(points_2d_homogeneous)
    print(points_3d_homogeneous)
    # 使用最小二乘法计算变换矩阵
    transform_matrix, _, _, _ = np.linalg.lstsq(points_3d_homogeneous, points_2d_homogeneous, rcond=None)
    print("变换矩阵:")
    print(transform_matrix)
    # 第5步：可选步骤，您可以将变换应用到一些2D点，查看结果
    # 例如，应用矩阵到2D点：
    transformed_points_2d = points_3d_homogeneous @ transform_matrix
    for i ,point,in enumerate(transformed_points_2d):
        print(points_2d[i],transformed_points_2d[i])



if __name__ == "__main__":
    main()
