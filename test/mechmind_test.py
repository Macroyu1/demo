# With this sample, you can connect to a camera and obtain the 2D image, depth map, and point cloud data.
import time
 
from mecheye.shared import *
from mecheye.area_scan_3d_camera import *
from mecheye.area_scan_3d_camera_utils import *
import cv2
import numpy as np
import open3d as o3d

import matplotlib.pyplot as plt

reference_direction = np.array([0, 0, 1])  # 示例：z轴方向
angle_threshold = 50.0  # 可调节的阈值

def pass_through(points, limit_min, limit_max, filter_value_name):        #点云直通滤波
 
	if (filter_value_name == "x")or(filter_value_name == "X"):
		target=0
	elif (filter_value_name == "y") or (filter_value_name == "Y"):
		target=1
	elif (filter_value_name == "z") or (filter_value_name == "Z"):
		target=2
	else:
		print("请输入正确的参数")
		exit()
	index = np.where((points[:, target] >= limit_min) & (points[:, target] <= limit_max))[0]
	sampled_points = points[index]

	return sampled_points

def filter_point_cloud_by_normal_direction(points, reference_direction, angle_threshold):    #计算点云法向量并根据法向量进行滤波
    point_cloud = o3d.geometry.PointCloud()  # 创建Open3D点云对象
    point_cloud.points = o3d.utility.Vector3dVector(points)  # 将顶点坐标赋值给点云对象
    # o3d.visualization.draw_geometries([pcd])  # 可视化点云
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))    #open3d计算点云法向量

   
    normals = np.asarray(point_cloud.normals)   #将法向量转换为array

    reference_direction = reference_direction / np.linalg.norm(reference_direction)    # 将参考方向归一化

    angle_threshold_rad = np.radians(angle_threshold)   # 设置角度阈值（以弧度为单位）

    # 过滤点云
    filtered_points = []
    filtered_normals = []

    for i in range(len(normals)):
        # 计算法向量与参考方向的夹角
        cos_angle = np.dot(normals[i], reference_direction)
        
        # 如果夹角小于阈值，则保留该点
        if cos_angle > np.cos(angle_threshold_rad):
            filtered_points.append(point_cloud.points[i])
            filtered_normals.append(normals[i])

    # 创建新的点云对象
    filtered_point_cloud = o3d.geometry.PointCloud()
    filtered_point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_point_cloud.normals = o3d.utility.Vector3dVector(filtered_normals)

    return filtered_points, filtered_point_cloud
          
            
# camera_info_list = Camera.discover_cameras()           #枚举当前可连接的全部相机，并获取各个相机的信息
# camera = Camera()                        #实例化Camera类，调用该类中的connect()方法，通过IP地址获取的设备信息连接对应的相机
# # camera.connect("192.168.23.203")


# frame_2d = Frame2D()                           #采集并获取用于生成2D图的数据
# camera.capture_2d(frame_2d)
# color = frame_2d.get_color_image()
# gray = frame_2d.get_gray_scale_image()

# color_img = color.data()
# # cv2.imshow('img', color_img)
# # cv2.waitKey(0)

# frame_3d = Frame3D()                     #采集并获取用于生成深度图和含法向量的无纹理点云的数据
# camera.capture_3d_with_normal(frame_3d)
# depth = frame_3d.get_depth_map()
# point_cloud = frame_3d.get_untextured_point_cloud_with_normals()
# point_xyz = frame_3d.get_untextured_point_cloud

# depth_img = depth.data()
# # cv2.imshow('img2', depth_img)
# # cv2.waitKey(0)

# # point_xyzrgbn = point_cloud.data()
# # point_xyz = point_xyzrgbn[:, :, :3].reshape(-1, 3) 
# # print(point_xyz)

# # # point_xyz_cloud = o3d.geometry.PointCloud()  # 创建Open3D点云对象
# # # point_xyz_cloud.points = o3d.utility.Vector3dVector(point_xyz)  # 将顶点坐标赋值给点云对象
# # o3d.visualization.draw_geometries([point_xyz_cloud])

# # a = ConvertDepthMapToPointCloud()
# # a.main()
pcd = o3d.io.read_point_cloud("test\\UntexturedPointCloud.ply")

# 检查点云数据
print(pcd)

# 可视化点云
#o3d.visualization.draw_geometries([pcd])

point_xyz = np.array(pcd.points)
#print(point_xyz)
x_min, x_max = [-500, 500]              #集装笼边界作为后续点云处理的边界
y_min, y_max = [-500, 500]
z_max = np.min(point_xyz[:, 2])          #找到点云中的最高点，也就是距离相机最近的点
sampled_points = pass_through(point_xyz, x_min, x_max,"x")        #直通滤波提取感兴趣区域点云
sampled_points = pass_through(sampled_points, y_min, y_max,"y")
sampled_points = pass_through(sampled_points, 0 , z_max+500,"z")          #提取最高点指定距离内的点云
#print(sampled_points.shape)

_, filtered_point_cloud = filter_point_cloud_by_normal_direction(sampled_points, reference_direction, angle_threshold)       #根据法向量滤波
filtered_point_cloud, ind = filtered_point_cloud.remove_radius_outlier(nb_points=50,radius=10)     #半径滤波去除离群点
 


labels = np.array(filtered_point_cloud.cluster_dbscan(eps=20, min_points=1000))

# 获取所有唯一的标签
unique_labels = set(labels)
# 可视化聚类结果
colors = np.array(plt.cm.jet(np.linspace(0, 1, len(set(unique_labels)))))

filtered_points = np.array(filtered_point_cloud.points)             #将点云转换为np数组用于后续计算

vis = o3d.visualization.Visualizer()           #创建可视化点云窗口
vis.create_window()
vis.add_geometry(filtered_point_cloud)
# cluster_points = filtered_points[labels == 2]
# print(labels.shape)
# print(cluster_points.shape)
# 给每个点分配颜色
color_map = np.zeros((np.array(labels).shape[0], 3))
for i, label in enumerate(labels):
    if label == -1:  # 噪声点
        color_map[i] = [0, 0, 0]
    else:
        color_map[i] = colors[label][0:3]

# 设置点云颜色
filtered_point_cloud.colors = o3d.utility.Vector3dVector(color_map)

# 可视化点云
#o3d.visualization.draw_geometries([filtered_point_cloud])

# 计算AABB（轴对齐包围盒）
def compute_aabb(points):
    # 计算点云的最小值和最大值
    min_point = np.min(points, axis=0)  # 最小值，形状 (3,)
    max_point = np.max(points, axis=0)  # 最大值，形状 (3,)
    
    # 将 min_point 和 max_point 转换为 (3, 1) 的列向量
    min_point = min_point.reshape(3, 1)
    max_point = max_point.reshape(3, 1)
    
    # 创建 AxisAlignedBoundingBox
    aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_point, max_bound=max_point)
    
    return aabb

bboxs = []          #存储包围盒信息,包含各个包围盒的最大最小角点
for label in unique_labels:             #计算各类点云包围盒
        if label == -1:
            continue  # -1 标签表示噪声点

        # 获取该标签的点
        cluster_points = np.array(filtered_points[labels == label])
        #print(cluster_points)
        # 计算包围盒
        # min_point, max_point = compute_aabb(cluster_points)
        # print(max_point, min_point)
        
        # 创建包围盒几何体
        cluster_point_cloud = o3d.geometry.PointCloud()
        cluster_point_cloud.points = o3d.utility.Vector3dVector(cluster_points)
        bbox = cluster_point_cloud.get_axis_aligned_bounding_box()
        box_min_point = np.array(bbox.min_bound)
        box_max_point = np.array(bbox.max_bound)
        bboxs.append(np.concatenate((box_min_point, box_max_point)))
        #bboxs.append(bbox.max_bound)
        # bbox = o3d.geometry.AxisAlignedBoundingBox(min_point, max_point)
        bbox.color = (1, 0, 0)  # 设置包围盒颜色为红色
        vis.add_geometry(bbox)
        # 输出包围盒信息
        print(f"Cluster {label}:")
        print(cluster_points.shape)
        print(f"  Bounding Box Min Point: {bbox.min_bound}")
        print(f"  Bounding Box Max Point: {bbox.max_bound}")
        # print(f"  Dimensions: {max_point - min_point}")
vis.run()
bboxs = np.array(bboxs)              #货物坐标信息，6×n，存储包围盒最大最小角点的三维坐标
print(bboxs)

def stack_box(boundary, bboxs):                                             #码垛规划,boundary为集装笼边界也就是码垛的边界,bboxs为包围盒信息
    x_min, x_max, y_min, y_max, z_min, z_max= boundary           #集装笼边界作为后续点云处理的边界
    
    #tmp_position = np.zeros(3)
    if bboxs.shape[0] == 0:                                      #集装笼中没有箱子是第一个箱子放在笼子左下角，并添加指定的偏移量
        stack_position = [x_min+50, y_min+50, z_max+50]
        
    else:                                                                 
        box_x_max = np.max(bboxs[:, 3])              #寻找X坐标最大值
        idx = (box_x_max - bboxs[:, 3]) < 600         
        box_max_line = bboxs[idx]                     #寻找X最大值那一列箱子
        idy = np.argmax(box_max_line[:, 4])                 #寻找X最大值那一列箱子中Y值最大的箱子，也就是最右上角的箱子
        print(box_max_line[idy])
        tmp_position = [box_max_line[idy][0], box_max_line[idy][4], box_max_line[idy][5]]
       
        if y_max - box_max_line[idy][4] <600 and x_max - box_max_line[idy][3]>600:         #如果到达y轴边界而没到达x轴边界，将下一个货物放到下一行堆放
            tmp_position[0] = np.max(bboxs[:, 3])+600
            tmp_position[1] = y_min
        if y_max - box_max_line[idy][4] <600 and x_max - box_max_line[idy][3]<=600:   #如果x、y都到达边界，则往上一层堆放
            tmp_position[0] = x_min
            tmp_position[1] = y_min
            tmp_position[2] = np.max(bboxs[:, 5])
        stack_position = [tmp_position[0]+50, tmp_position[1]+50, tmp_position[2]+50]
    
    return stack_position

boundary = [-500, 500, -500, 500, 0, 1600]
stack_position = stack_box(boundary, bboxs)
print(stack_position)
