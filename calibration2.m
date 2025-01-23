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
world_points = [X2, Y2, Z2];

% 获取相机内参矩阵 K
K = cameraParams.Intrinsics.IntrinsicMatrix;

% 初始旋转矩阵和平移向量（假设单位矩阵和平移为零）
R_init = eye(3);
t_init = zeros(3,1);

% 定义优化目标函数：最小化重投影误差
fun = @(x) reprojection_error(x, image_points, world_points, K);

% 进行最小二乘优化，估计旋转矩阵和平移向量
options = optimset('Display','iter','TolFun',1e-6,'TolX',1e-6);
x_opt = lsqnonlin(fun, [R_init(:); t_init], [], [], options);

% 将优化结果分离为旋转矩阵和平移向量
R_opt = reshape(x_opt(1:9), 3, 3);  % 旋转矩阵 (3x3)
t_opt = x_opt(10:12);  % 平移向量 (3x1)

% 打印旋转矩阵和平移向量
disp('Estimated Rotation Matrix:');
disp(R_opt);

disp('Estimated Translation Vector:');
disp(t_opt);

% 组合旋转矩阵和平移向量为4x4矩阵
transformation_matrix = [R_opt, t_opt; 
                         0, 0, 0, 1];

% 打印旋转平移矩阵
disp('Transformation Matrix (4x4):');
disp(transformation_matrix);

% 计算重投影误差函数
function err = reprojection_error(x, image_points, world_points, K)
    % 提取旋转矩阵和平移向量
    R = reshape(x(1:9), 3, 3);  % 旋转矩阵 (3x3)
    t = x(10:12);  % 平移向量 (3x1)
    
    % 检查 R 和 t 是否有效
    if any(isnan(R(:))) || any(isnan(t))
        disp('NaN detected in R or t');
        err = NaN;
        return;
    end
    
    % 计算重投影误差
    num_points = size(image_points, 1);
    projected_points = zeros(num_points, 2);
    
    for i = 1:num_points
        % 世界坐标点
        P = world_points(i, :)';
        
        % 通过旋转矩阵和平移向量变换到相机坐标系
        p_cam = R * P + t;
        
        % 归一化，使得 Z = 1
        if p_cam(3) == 0
            disp('Z coordinate is zero for point ');
            err = NaN;
            return;
        end
        
        % 将3D相机坐标转换为2D图像坐标
        p_img = K * p_cam;
        p_img = p_img(1:2) / p_img(3);  % 归一化得到2D图像坐标
        
        % 存储重投影点
        projected_points(i, :) = p_img';
    end
    
    % 计算重投影误差
    err = image_points - projected_points;
    err = err(:);  % 将误差向量化
end