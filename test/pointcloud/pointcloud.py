import open3d as o3d
import numpy as np
import cv2

class Point_Cloud:
    def __init__(self,pcd_file) -> None:
        self.points = self.read(pcd_file)
    
    def read(self,pcd_file) -> np.asarray:
        # 读取点云文件
        self.pcd = o3d.io.read_point_cloud(pcd_file)
        print(np.shape(self.pcd.points))
        # 均匀下采样
        self.pcd = self.pcd.uniform_down_sample(every_k_points=5)
        print(np.shape(self.pcd.points))
        if not self.pcd.has_points():
            raise ValueError(f"点云文件 {pcd_file} 加载失败或为空！")
        return np.asarray(np.asarray(self.pcd.points))

    def show(self,points):
        # 创建新的点云
        pcd = o3d.geometry.PointCloud()
        
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.visualization.draw_geometries([pcd])

    def ROI(self,roi:list):
        mask = (
            (self.points[:, 0] >= roi[0]) & (self.points[:, 0] <= roi[1]) &
            (self.points[:, 1] >= roi[2]) & (self.points[:, 1] <= roi[3]) &
            (self.points[:, 2] >= 5000 - roi[5]) & (self.points[:, 2] <= 5000-roi[4])
        )
        self.points = self.points[mask]
        return(self.points)
    
    def top_mask(self, min_points=1000, z_step=10.0):
        # 按 z 值降序排序
        sorted_points = self.points[np.argsort(self.points[:, 2])[::-1]]

        # 获取最大和最小 z 值
        max_z = sorted_points[0, 2]
        min_z = sorted_points[-1, 2]

        # 分层筛选
        current_z = z_step+1
        while current_z < max_z:
            # 筛选当前层的点
            mask = (sorted_points[:, 2] >= current_z) & (sorted_points[:, 2] < current_z + z_step)
            layer_points = sorted_points[mask]

            # 如果当前层点数满足要求，返回该层点云
            if len(layer_points) >= min_points:
                mask = mask = (sorted_points[:, 2] >= current_z - 5*z_step) & (sorted_points[:, 2] < current_z + 5*z_step)
                layer_points = sorted_points[mask]
                return layer_points

            # 否则移动到下一层
            current_z += z_step

        # 如果没有找到满足条件的层，返回空数组
        return np.array([])
    
    def cluster(self,points, eps=50, min_points=10):
        # 数据检查与预处理
        points = np.array(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points 必须是形状为 (N, 3) 的二维数组！")
        points = points[np.isfinite(points).all(axis=1)]

        # 构建点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # DBSCAN 聚类
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
        max_label = labels.max()

        if max_label < 0:
            print("所有点均为噪点，无有效簇！")
            return None, []

        print(f"检测到 {max_label + 1} 个簇")
        
        # 为每个簇分配随机颜色
        # 随机构建n+1种颜色，这里需要归一化
        colors = np.random.randint(1, 255, size=(max_label + 1, 3)) / 255.
        colors = colors[labels]             # 每个点云根据label确定颜色
        colors[np.array(labels) < 0] = 0    # 噪点配置为黑色
        pcd.colors = o3d.utility.Vector3dVector(colors)  # 格式转换(由于pcd.colors需要设置为Vector3dVector格式)

        # 计算每个簇的最大最小 XY 坐标
        bounds = []
        for label in range(max_label + 1):
            cluster_points = points[labels == label]
            min_x, min_y ,min_z = cluster_points[:, :].min(axis=0)
            max_x, max_y ,max_z = cluster_points[:, :].max(axis=0)
            bounds.append({
                "label": label,
                "min_xyz":[min_x, min_y, min_z],
                "max_xyz":[max_x, max_y, max_z]
            })

        return points, bounds
    
    def point_cloud_mask(self,bounds):
        mask = []  # 初始化掩膜为全零

        for bound in bounds:
            label = bound["label"]
            min_x, min_y, min_z = np.array(bound["min_xyz"])
            max_x, max_y, max_z = np.array(bound["max_xyz"])

            # 填充掩膜
            mask[min_y:max_y, min_x:max_x, min_z:max_z] = label + 1  # 避免背景为 0，与簇区分

        return mask

def erode_point_cloud(points, radius=1.0, min_neighbors=10):
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 构建 KDTree
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    # 初始化结果列表
    eroded_points = []

    # 遍历点云，计算每个点的邻居数
    for i, point in enumerate(points):

        num, idx, _ = kdtree.search_radius_vector_3d(point, radius)

        if len(idx) >= min_neighbors:
            eroded_points.append(point)

    return np.array(eroded_points)

def dilate_point_cloud(points, radius=1.0):
    # 膨胀操作：在每个点周围随机生成若干个新点
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 构建 KDTree
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    dilated_points = list(points)
    for point in points:
        _, idx, _ = kdtree.search_radius_vector_3d(point, radius)
        for neighbor_idx in idx:
            neighbor = points[neighbor_idx]
            dilated_points.append(neighbor + np.random.uniform(-radius, radius, size=3))
    
    return np.unique(np.array(dilated_points), axis=0)

def point_cloud_to_image_mask(points, image_shape, intrinsic_matrix, extrinsic_matrix):
    # 转换点云坐标到齐次坐标
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))

    # 将点云从世界坐标系转换到相机坐标系
    points_camera = (extrinsic_matrix @ points_homogeneous.T).T  # 乘以外参矩阵进行坐标转换


    # 将相机坐标系的点投影到图像平面
    points_image = (intrinsic_matrix @ points_camera[:, :3].T).T  # 投影到图像平面
    points_image[:, :2] /= points_image[:, 2][:, None]  # 归一化，转换到像素坐标

    # 转换为像素坐标
    pixel_coords = points_image[:, :2].astype(int)

    # 创建空的图像掩膜
    mask = np.zeros(image_shape, dtype=np.uint8)

    # 过滤掉超出图像范围的点
    valid_mask = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < image_shape[1]) & (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < image_shape[0])
    valid_pixel_coords = pixel_coords[valid_mask]

    # 在掩膜中标记点
    mask[valid_pixel_coords[:, 1], valid_pixel_coords[:, 0]] = 255

    return mask

def apply_mask_to_image(image, mask, display_size=(640, 480)):
    # 确保掩膜是二值的
    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

    # 使用膨胀操作填充孤立点，连接相邻区域
    kernel = np.ones((15, 15), np.uint8)  # 定义膨胀的内核
    dilated_mask = cv2.dilate(mask, kernel, iterations=2)  # 膨胀操作

    # 确保掩膜和图像大小一致
    dilated_mask_resized = cv2.resize(dilated_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 创建一个与图像大小相同的黑色背景
    result_image = np.zeros_like(image)

    # 将掩膜区域从原图像复制到结果图像中
    result_image[dilated_mask_resized == 255] = image[dilated_mask_resized == 255]

    # 调整显示图像大小
    result_image_resized = cv2.resize(result_image, display_size)

    return result_image_resized
    

def main():
    PC = Point_Cloud("image\\test.pcd")
    points = PC.ROI(roi = [-900, 900, -900, 900, 0, 5000])

    # 膨胀和腐蚀操作
    # eroded_points = erode_point_cloud(roi_points, radius=2, min_neighbors=1)
    # dilated_points = dilate_point_cloud(eroded_points, radius=5.0)
    top_points = PC.top_mask(min_points=50, z_step=10.0)
    # print(np.shape(top_points))

    
    # 显示处理后的点云
    # result_points, bound = PC.cluster(top_points,50,10)
    # PC.show(result_points[PC.point_cloud_mask(bound)])
    PC.show(top_points)

    # 相机内参矩阵和畸变系数
    intrinsic_matrix = np.array([[3444.26, 0, 1161.62],
                               [0, 3447.16, 1053.98],
                               [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([-0.0582, -0.6964, -0.003, 0.0175, 1.1209], dtype=np.float32)

    extrinsic_matrix = [
    [0.997125, -0.0534843, 0.0536749, -95.5026],
    [0.0546529, 0.998294, -0.0205439, 118.367],
    [-0.0524846, 0.0234183, 0.998347, -8.90425],
    [0,0,0,1]
    ]
    # extrinsic_matrix = [
    # [-0.9987,0.04416948,-0.02414783,1290.80541992],
    # [0.04423093,0.99901927,-0.00201421,886.10870361],
    # [0.02403515,-0.0030797,-0.99970639,2210.33666992],
    # [0,0,0,1]
    # ]

    # 图像形状 (height, width)
    image_shape = (2448, 2048)

    # 读取图像
    image = cv2.imread("image\\test.jpg")  # 替换为你的图像文件路径

    # 生成图像掩膜
    image_mask = point_cloud_to_image_mask(top_points, image_shape, intrinsic_matrix, extrinsic_matrix)

    # 将掩膜应用到图像上，并调整显示大小
    result_image = apply_mask_to_image(image, image_mask, display_size=(2448,2048))  # 设置你想显示的图像大小
    # result_image = cv2.resize(image_mask,(800,600))

    # 显示最终图像
    cv2.imshow("Masked Image", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
