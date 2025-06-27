import torch
from ultralytics import YOLO

def load_and_modify_model(model_path='yolov8n-pose.pt'):
    """
    加载YOLOv8n-pose模型并展示结构，支持层替换
    :param model_path: 模型文件路径
    :return: 修改后的模型
    """
    # 加载原始模型
    model = YOLO(model_path)
    print("原始模型结构：")
    print(model.model)
 
    # 遍历并打印模型结构
    for name, layer in model.named_modules():
        print(f"Layer: {name} - Type: {type(layer).__name__}")
  
    # 示例：替换第一个Conv层的激活函数（可根据需要修改替换逻辑）
    def replace_layers(model):
        for name, layer in model.named_modules():
            #print(layer.type)
            # 替换检测头中的Conv层（示例）
            if isinstance(layer, torch.nn.BatchNorm2d) and "model.model.8.m.0" in name:


                new_conv = torch.nn.MaxPool2d(
                    kernel_size=2
                )
                setattr(model.model, name, new_conv)
                print(f"已替换层: {name}")
                #print(model.model)


    replace_layers(model)
    return model

if __name__ == "__main__":
    modified_model = load_and_modify_model()
    print("\n修改后的模型结构验证：")
    print(modified_model.model)