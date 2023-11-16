import json
import os
from module.progress_bar import print_progress

def parse_json_files(json_folder):
    labels_data = []
    excepted_data = []

    def convert_to_bbox(data, origin_wid, origin_hei):
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

        x_min = x_min / origin_wid
        y_min = y_min / origin_hei
        x_max = x_max / origin_wid
        y_max =  y_max / origin_hei

        bbox = [x_min, y_min, x_max, y_max]
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
                    origin_wid = data['image']['image_size'][1]
                    origin_hei = data['image']['image_size'][0]
                    convert = convert_to_bbox(point_data, origin_wid, origin_hei)
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

#여기서 annotations를 loop하는 내내 label_map 내 파일이 없다면 image_list에서 제거하는 작업이 필요함
# Example usage
# json_folder_path = '../../data'
# labels_data = parse_json_files(json_folder_path)

# Printing the result
# print(labels_data)