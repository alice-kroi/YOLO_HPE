import cv2
import numpy as np

def visualize_pose(image, keypoints, connections=None, point_size=5, line_thickness=2):
    """
    可视化人体姿态关键点
    :param image: 图片路径或numpy数组
    :param keypoints: 关键点坐标数组，形状为[N, 2]
    :param connections: 关键点连接关系，默认为COCO格式
    :param point_size: 关键点大小
    :param line_thickness: 连接线粗细
    :return: 绘制后的图像
    """
    # COCO关键点连接定义（17个关键点）
    if connections is None:
        connections = [
            (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
            (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
            (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
            (0, 3), (0, 4), (3, 5), (4, 6)
        ]
    
    # 读取图像
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"无法读取图像: {image}")
    else:
        img = image.copy()
    
    # 转换坐标格式
    keypoints = np.array(keypoints).reshape(-1, 2)
    
    # 绘制骨骼连接
    for (i, j) in connections:
        if i < len(keypoints) and j < len(keypoints):
            start = tuple(map(int, keypoints[i]))
            end = tuple(map(int, keypoints[j]))
            cv2.line(img, start, end, (0, 255, 255), line_thickness)
    
    # 绘制关键点
    for (x, y) in keypoints:
        cv2.circle(img, (int(x), int(y)), point_size, (0, 0, 255), -1)
    
    # 显示结果
    cv2.imshow('Human Pose Visualization', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img

if __name__ == "__main__":
    # 示例测试
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_keypoints = [
        [320, 100],  # 鼻子
        [300, 120], [340, 120],  # 眼睛
        [290, 150], [350, 150],  # 耳朵
        [320, 200], [250, 250], [390, 250],  # 肩膀/肘部
        [200, 350], [440, 350],  # 手腕
        [320, 300], [280, 400], [360, 400],  # 髋部/膝盖
        [280, 450], [360, 450]  # 脚踝
    ]
    
    visualize_pose(test_image, test_keypoints)