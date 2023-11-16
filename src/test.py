import torch
from PIL import Image
import torchvision.transforms as transforms
from module import Load_Model

def make_prediction(model, img, threshold):
    model.eval()
    preds = model(img)
    for id in range(len(preds)):
        idx_list = []

        for idx, score in enumerate(preds[id]['scores']):
            if score > threshold:
                idx_list.append(idx)


        preds[id]['boxes'] = preds[id]['boxes'][idx_list]
        preds[id]['boxes'] = preds[id]['boxes'][idx_list]
        preds[id]['boxes'] = preds[id]['boxes'][idx_list]

    return preds

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to (224, 224)
    transforms.ToTensor(),           # Convert PIL Image to PyTorch tensor
    
])

num_classes = 6  
load_weights_dir = "../out/weights/model_num_5.pt"

model_loader = Load_Model(num_classes,load_weights_dir)

model = model_loader.load()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)

with torch.no_grad():
    img = Image.open("")
    img = transform(img)

    pred = make_prediction(model, img, 0.5)
    print(pred)
