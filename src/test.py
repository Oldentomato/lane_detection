import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from module import DataLoad,train_print_progress, print_progress
import torchvision.transforms as transforms
import glob


json_folder_path = '../data/labels'
image_folder_path = '../data/images'

num_classes = 6  

image_paths = glob.glob(f"{image_folder_path}/*.jpg") + glob.glob(f"{image_folder_path}/*.png")

data_load = DataLoad("test_label.pkl", "test_except.pkl", "test_image.pkl", json_folder_path, image_paths)


y_train, except_data = data_load.run_label()  # bbox와 category 데이터 그리고 제외된 데이터
x_train = data_load.run_image()



print(len(y_train))
print(len(x_train))

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to (224, 224)
    transforms.ToTensor(),           # Convert PIL Image to PyTorch tensor
    
])
# 데이터 변환 함수
def transform_data(image, targets):
    image = transform(image)

    # Convert labels to one-hot encoding
    # labels_onehot = torch.zeros((len(targets['labels']), num_classes), dtype=torch.float32)
    # labels_onehot.scatter_(1, torch.tensor(targets['labels']).view(-1, 1), 1)

    boxes = torch.as_tensor(targets['boxes'], dtype=torch.float32)
    labels = torch.as_tensor(targets['labels'], dtype=torch.int64)

    targets = {
        'boxes': boxes,
        'labels': labels
    }
    return image, targets

# 변환된 데이터 예시
transformed_data = transform_data(x_train[0], y_train[0])
print(transformed_data)

# 데이터셋 클래스
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images, targets, transform=None):
        self.images = images
        self.targets = targets
        self.transform = transform

    def __getitem__(self, idx):
        image, target = self.images[idx], self.targets[idx]
        if self.transform:
            image, target = self.transform(image, target)

        return image, target

    def __len__(self):
        return len(self.images)
    
def collate_fn(batch):
    return tuple(zip(*batch))

# 모델 정의
def get_object_detection_model(num_classes):
    # ResNet-50을 기반으로 하는 Faster R-CNN 모델 정의
    backbone = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = backbone.roi_heads.box_predictor.cls_score.in_features
    backbone.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


    return backbone

# 모델 초기화

model = get_object_detection_model(num_classes)

# 데이터 로더
dataset = CustomDataset(images=x_train, targets=y_train, transform=transform_data)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# 모델 학습
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

num_epochs = 5
for epoch in range(num_epochs):
    for i,(images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        loss.backward()
        optimizer.step()

        train_print_progress(i, len(data_loader), loss.item())
    print_progress(epoch, num_epochs)
        # print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# 모델 저장
torch.save(model.state_dict(), '../out/models/object_detection_model.pth')
