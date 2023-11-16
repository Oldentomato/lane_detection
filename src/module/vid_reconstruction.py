import cv2
import time

def Get_Video_Frame(dir, save_dir):
    vidcap = cv2.VideoCapture(dir)
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    out = cv2.VideoWriter(save_dir+"output.avi", fourcc, fps, (1280,720))