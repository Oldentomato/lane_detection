from PIL import ImageDraw, ImageFont, Image
from module import print_progress
import glob
import os
import json
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='True'



def calculate_bbox_size(bbox):
    # 바운딩 박스의 크기 계산
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width,height


def detect_small_bbox(bbox, width_thres, height_thres):
    # threshold보다 작은 크기의 바운딩 박스 감지
    w, h = calculate_bbox_size(bbox)
    if w < width_thres or h < height_thres:
        return True
    else:
        return False

def parse_json_files(json_folder):
    labels_data = []
    excepted_data = []



    def convert_to_bbox(data):
        if not data or not isinstance(data, list):
            return None

        if len(data) < 2:
            return None

        x_values = [item[0] for item in data]
        y_values = [item[1] for item in data]

        x_min = min(x_values)
        y_min = min(y_values)
        x_max = max(x_values)
        y_max = max(y_values)

        if x_max > 1920 or y_max > 1200 or x_min < 0 or y_min < 0:
            return None

        if x_max == x_min or y_max == y_min:
            return None

        bbox = [x_min, y_min, x_max, y_max]

        w_thres = 50
        h_thres = 50
        if detect_small_bbox(bbox, w_thres, h_thres):
            return None
        else:
            return bbox

    # Iterate over each JSON file in the folder
    for i,json_file in enumerate(os.listdir(json_folder)):
        print_progress(i,len(os.listdir(json_folder)))
        if json_file.endswith(".json"):
            json_path = os.path.join(json_folder, json_file)

            with open(json_path, 'r') as file:
                data = json.load(file)


            boxes = []
            label = []
            for annotation in data['annotations']:
                box_data = annotation['data']
                attributes = annotation['attributes']

                try:
                    # Extracting values from attributes
                    lane_color = next(attr['value'] for attr in attributes if attr['code'] == 'lane_color')
                    lane_type = next(attr['value'] for attr in attributes if attr['code'] == 'lane_type')

                    # Constructing label in the format 'color_type'
                    label_str = f'{lane_color.lower()}_{lane_type.lower()}'

                    # Mapping labels to integers
                    label_map = {'white_dotted': 0, 'white_solid': 1, 'yellow_dotted': 2, "yellow_solid": 3, "blue_dotted": 4, "blue_solid": 5}
                    label.append(label_map[label_str])

                    # Constructing the box data
                    point_data = [[point['x'], point['y']] for point in box_data]
                    convert = convert_to_bbox(point_data)
                    if convert == None:
                        continue
                    else:
                        boxes.append(convert)
                except:
                    continue
            else:
                if len(boxes) == 0:
                    excepted_data.append(data['image']['file_name'])

            # Adding to the labels_data list
            if len(boxes) >= 1:
                labels_data.append({'boxes': boxes, 'labels': label})


    return labels_data, excepted_data

def visualize_images_with_bbox(images, bbox_data_list):
    # images: 이미지 파일 경로의 리스트
    # bbox_data_list: bbox 데이터 리스트 [[xmin, ymin, xmax, ymax], ...]
    # output_path: 결과 이미지의 저장 경로

    num_images = len(images)
    if num_images != len(bbox_data_list):
        raise ValueError("이미지와 bbox 데이터의 수가 일치하지 않습니다.")

    
    for i,bbox in enumerate(bbox_data_list):
        print_progress(i,len(bbox_data_list))
        img = images[i]
        draw = ImageDraw.Draw(img)

        # bbox 그리기
        bbox_data = bbox['boxes']
        label = bbox['labels']
        draw_bbox(draw, bbox_data, label)
        

        img.save(f'../out/test/{i}.jpg')
    # result_image.show()

def draw_bbox(draw, bbox_data, label):
    # bbox를 이미지에 그리는 함수
    label_map = ['white_dotted', 'white_solid', 'yellow_dotted', "yellow_solid", "blue_dotted", "blue_solid"]
    font = ImageFont.truetype("arial.ttf", 28)
    for bbox,labelnum in zip(bbox_data, label):
        xmin, ymin, xmax, ymax = bbox
        draw.rectangle([(xmin, ymin), (xmax, ymax )],
                       outline="green", width=3)
        draw.text((xmin,ymin), f"{label_map[labelnum]}", (255,0,0),font) # x=0, y=10, (0,0,0) : 검은색(RGB값)

json_folder_path = '../data/labels'
image_folder_path = '../data/images'     
image_paths = glob.glob(f"{image_folder_path}/*.jpg") + glob.glob(f"{image_folder_path}/*.png")



y_train, except_data = parse_json_files(json_folder_path)  # bbox와 category 데이터 그리고 제외된 데이터
x_train = []
for image_path in image_paths:
    if os.path.basename(image_path) not in except_data:
        image = Image.open(image_path).convert("RGB")
        x_train.append(image)

print(f"y_len: {len(y_train)}")
print(f"x_len: {len(x_train)}")

visualize_images_with_bbox(x_train, y_train)