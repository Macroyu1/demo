% 读取表格数据
filename = 'E:\Work Data\demo\test\calibration\point.xls';
data = xlsread(filename);

% 提取各列的数据
X1 = data(:, 2);  % 2DX
Y1 = data(:, 3);  % 2DY
X2 = data(:, 5);  % 3DX
Y2 = data(:, 6);  % 3DY
Z2 = data(:, 7);  % 3DZ

% 提取图像像素坐标
image_points = [X1, Y1];

% 提取世界坐标
world_points = [X2,Y2,Z2];

% 假设相机内参矩阵 K 已知

% 使用 PnP 算法估算外参
[rotation_matrix, translation_vector] = estimateWorldCameraPose(image_points, world_points, cameraParams.Intrinsics);

% 组合旋转矩阵和平移向量为4x4矩阵
transformation_matrix = [rotation_matrix, translation_vector'; 
                         0, 0, 0, 1];
% 设置显示格式为常规浮动显示
format long g;
% 打印旋转平移矩阵
disp('Transformation Matrix (4x4):');
disp(transformation_matrix);

%% 

    "RT":   [[0.997125,-0.0534843,0.0536749,-95.5026],
            [0.0546529,0.998294,-0.0205439,118.367],
            [-0.0524846,0.0234183,0.998347,-8.90425],
            [0,0,0,1]],