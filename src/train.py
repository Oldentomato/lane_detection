import torch
from module import DataLoad,train_print_progress, On_Train, Load_Model
import torchvision.transforms as transforms
import glob


json_folder_path = '../data/labels'
image_folder_path = '../data/images'
load_weights_dir = ""

num_classes = 6  

num_epochs = 5

image_paths = glob.glob(f"{image_folder_path}/*.jpg") + glob.glob(f"{image_folder_path}/*.png")

data_load = DataLoad("test_label.pkl", "test_except.pkl", "test_image.pkl", json_folder_path, image_paths)


y_train, except_data = data_load.run_label()  # bbox와 category 데이터 그리고 제외된 데이터
x_train = data_load.run_image()


print(f"y_count: {len(y_train)}")
print(f"x_count: {len(x_train)}")

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


# 데이터 로더
dataset = CustomDataset(images=x_train, targets=y_train, transform=transform_data)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

model_loader = Load_Model(num_classes, load_weights_dir)

model = model_loader.load()


# 모델 학습
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

one_epoch_train = On_Train(model,optimizer,data_loader,device, model_save_dir="../out/weights")


for epoch in range(num_epochs):
    one_epoch_train.run(epoch, is_eval=True)

    print(f"Epoch {epoch} / {num_epochs}")

# 모델 저장
# torch.save(model.state_dict(), '../out/models/object_detection_model.pth')
