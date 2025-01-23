import cv2
import open3d as o3d
import numpy as np

#-366.39 -493.07 4320.52
#23.197 -504.593 4316.237
#-341.86 -197.41 4305.61
#35.198 -222.699 4337.799
#1307 650


intrinsic_matrix = np.array([[2111.84955910501, 0, 1290.97644079674],
                               [0, 2104.38342695474, 909.415077675728],
                               [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([-0.0582, -0.6964, -0.003, 0.0175, 1.1209], dtype=np.float32)
        
extrinsic_matrix = np.array([ [ 0.992949194608488 ,    -0.000183697262421475    ,  -0.00752634335280559   ,  -8.37491630316323e-05],
     [ -2.57858190291198e-05  ,       0.992861890735398 ,     -0.00529848127847806   ,   0.000101933510194447],
      [-0.00139121994487156  ,   -0.000959970207155758     ,     0.99292337549335     ,  -0.0124224916904693],
       [                  0    ,                     0   ,                      0   ,                      1]]) 

point_3d = np.array([[-366.39 ,-493.07 ,4320.52,1],[23.197, -504.593 ,4316.237,1],[-341.86 ,-197.41 ,4305.61,1],[35.198 ,-222.699 ,4337.799,1]])
point_2d = np.array([1307 ,650])
# print(np.shape(point_3d))
points_image = point_3d @ extrinsic_matrix.T  # 乘以外参矩阵进行坐标转换

pixel_coords = (intrinsic_matrix @ points_image[:, :3].T).T  # 内参矩阵变换
points_image = pixel_coords[:,:2]/pixel_coords[:,2][:,None]
points = np.round(points_image[:, :2]).astype(int)  # 转换为整数像素坐标
print("实际2D坐标:[1307 650]")
print("计算2D坐标:",points)

left_top = (np.min(points[:, 0]), np.min(points[:, 1]))  # 最小的 x 和 y
right_bottom = (np.max(points[:, 0]), np.max(points[:, 1]))  # 最大的 x 和 y

img = cv2.imread("test\image\\test.jpg")
cv2.rectangle(img, left_top,right_bottom, (0, 255, 0), 2)
img = cv2.resize(img, (1224, 1024))


cv2.imshow('Rectangle', img)
cv2.waitKey(0)
cv2.destroyAllWindows()