from os import path
from module.data_preprocessing import parse_json_files
from module.save2pickle import save_to_pickle, load_from_pickle
import os
from PIL import Image

class DataLoad():
    def __init__(self, save_label_name, save_except_name, save_image_name, json_folder_path, image_datas, is_saving=True):
        self.save_label_name = save_label_name
        self.save_except_name = save_except_name
        self.json_folder_path = json_folder_path
        self.save_image_name = save_image_name
        self.image_datas = image_datas
        self.is_saving = is_saving

    def __get_images(self):
        temp_images = []
        for image_path in self.image_datas:
            if os.path.basename(image_path) not in self.saved_except:
                image = Image.open(image_path).convert("RGB")
                temp_images.append(image)

        return temp_images


    def __labeldata_check(self):
        if path.exists(f"../out/labels/{self.save_label_name}") and path.exists(f"../out/labels/{self.save_except_name}"):
            print("cache label file found! loading...")
            self.saved_label= load_from_pickle(f"../out/labels/{self.save_label_name}")
            self.saved_except = load_from_pickle(f"../out/labels/{self.save_except_name}")
            return True
        else:
            print("cache label file not found generating...")
            self.saved_label, self.saved_except = parse_json_files(self.json_folder_path)
            return False
        
    def __imagedata_check(self):
        if path.exists(f"../out/images/{self.save_image_name}"):
            print("cache image file found! loading...")
            self.saved_image = load_from_pickle(f"../out/images/{self.save_image_name}")
            return True
        else:
            print("cache image file not found generating...")
            self.saved_image = self.__get_images()
            return False




    def run_label(self):
        if not self.__labeldata_check():
            if self.is_saving:
                save_to_pickle(self.saved_label, f"../out/labels/{self.save_label_name}")
                save_to_pickle(self.saved_except, f"../out/labels/{self.save_except_name}")


        return self.saved_label, self.saved_except
    
    def run_image(self):
        if not self.__imagedata_check():
            if self.is_saving:
                save_to_pickle(self.saved_image, f"../out/images/{self.save_image_name}")

        return self.saved_image

