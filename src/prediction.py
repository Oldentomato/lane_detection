import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from module import Load_Model, print_progress
import glob
import os

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

def visualize_images_with_bbox(image, bboxes, pred_labels, pred_scores, count):
    draw = ImageDraw.Draw(image)

    # bbox 그리기
    for bbox,label,score in zip(bboxes,pred_labels,pred_scores):
        bbox_data = bbox
        draw_bbox(draw, bbox_data, label, score)
    

    image.save(f'../out/test/{count}.jpg')

def draw_bbox(draw, bbox_data, labels, scores):
    # bbox를 이미지에 그리는 함수
    label_map = ['white_dotted', 'white_solid', 'yellow_dotted', "yellow_solid", "blue_dotted", "blue_solid"]
    font = ImageFont.truetype("arial.ttf", 28)
    for bbox,label,score in zip(bbox_data, labels, scores):
        xmin, ymin, xmax, ymax = bbox
        draw.rectangle([(xmin, ymin), (xmax, ymax )],
                       outline="green", width=3)
        draw.text((xmin,ymin), f"{label_map[label]} : {scores}", (255,0,0),font) # x=0, y=10, (0,0,0) : 검은색(RGB값)



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
    image_folder_path = '../data/images'     
    image_paths = glob.glob(f"{image_folder_path}/*.jpg") + glob.glob(f"{image_folder_path}/*.png")
    label_map = ['white_dotted', 'white_solid', 'yellow_dotted', "yellow_solid", "blue_dotted", "blue_solid"]
    for i, image_path in enumerate(image_paths):
        image = Image.open(image_path).convert("RGB")
        origin_image = image

        image = transform(image)
        image = image.unsqueeze_(0)
        pred = make_prediction(model, image, 0.2)
        pred_scores = pred[0]['scores'].tolist()
        pred_labels = pred[0]['labels'].tolist()
        pred_boxes = pred[0]['boxes'].tolist()

        visualize_images_with_bbox(image, pred_boxes, pred_labels, pred_scores, i)
    
