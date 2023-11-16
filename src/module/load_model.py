import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class Load_Model:
    def __init__(self, num_classes, load_weights_dir=""):
        self.load_weights_dir = load_weights_dir
        self.num_classes = num_classes

    # 모델 정의
    def __get_object_detection_model(self, num_classes, weights):
        # ResNet-50을 기반으로 하는 Faster R-CNN 모델 정의
        backbone = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        in_features = backbone.roi_heads.box_predictor.cls_score.in_features
        backbone.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


        return backbone

    def load(self):
        if self.load_weights_dir != "":
            weights = torch.load(self.load_weights_dir)
        else:
            weights = "DEFAULT"
        model = self.__get_object_detection_model(self.num_classes, weights)

        return model

