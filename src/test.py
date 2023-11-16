import torch
from PIL import Image
import torchvision.transforms as transforms
from module import Load_Model
import torchvision
import matplotlib.pyplot as plt

def make_prediction(model, img, threshold):
    model.eval()
    preds = model(img)
    for id in range(len(preds)):
        idx_list = []

        for idx, score in enumerate(preds[id]['scores']):
            if score > threshold:
                idx_list.append(idx)


        preds[id]['boxes'] = preds[id]['boxes'][idx_list]
        preds[id]['labels'] = preds[id]['labels'][idx_list]
        preds[id]['scores'] = preds[id]['scores'][idx_list]

    return preds

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to (224, 224)
    transforms.ToTensor(),           # Convert PIL Image to PyTorch tensor
    
])

num_classes = 6  
load_weights_dir = "../out/weights/model_num_2.pt"

model_loader = Load_Model(num_classes,load_weights_dir)

model = model_loader.load()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model.to(device)

with torch.no_grad():
    img = Image.open("../data/test/2215112.jpg").convert("RGB")
    img = transform(img)
    img_unsqeeze = img.unsqueeze_(0)

    
    pred = make_prediction(model, img_unsqeeze, 0.2)
    print(pred)

    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(img.numpy().transpose((1, 2, 0)))

    for box, label, score in zip(prediction[0]['boxes'], prediction[0]['labels'], prediction[0]['scores']):
        box = [round(coord.item(), 2) for coord in box.tolist()]
        label = label.item()
        score = round(score.item(), 3)

        # 박스 시각화
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r',
                                facecolor='none', label=f'{classes[label]}: {score}')
        ax.add_patch(rect)

        # 레이블과 점수 표시
        plt.text(box[0], box[1] - 5, f'{classes[label]}: {score}', color='r')

    # pred_scores = pred[0]['scores'].tolist()
    # pred_labels = pred[0]['labels'].tolist()

    # print(pred_scores)
    # print(pred_labels)
    
    # print(pred_labels.index(max(pred_scores)))
