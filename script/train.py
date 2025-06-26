import torch
import os
from ultralytics import YOLO
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from img_pose_show import visualize_pose

# 训练主函数
def run_prediction():
    # 精简配置
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'model_weights': 'yolov8n-pose.pt',  # 确保该权重文件存在
        'source': 'img/R-C.jpg',  # 预测源文件路径
        'conf': 0.5,  # 置信度阈值
        'save': True  # 保存预测结果
    }
    
    # 加载模型
    model = YOLO(config['model_weights']).to(config['device'])
    
    # 执行预测
    results = model.predict(
        source=config['source'],
        conf=config['conf'],
        save=config['save']
    )
    
    # 打印结果
    for result in results:
        print(f"检测到 {len(result.boxes)} 个人体")
        if result.keypoints is not None:
            # 转换坐标为numpy数组格式
            keypoints = result.keypoints.xy.cpu().numpy()
            print(f"关键点坐标：\n{keypoints}")
            
            # 可视化第一个检测到的人体关键点
            if len(keypoints) > 0:
                visualize_pose(
                    config['source'],
                    keypoints=keypoints[0],  # 取第一个检测结果
                    point_size=6,
                    line_thickness=2
                )

if __name__ == "__main__":
    run_prediction()  # 重命名主函数