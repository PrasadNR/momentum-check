from pathlib import Path
from tqdm import tqdm
from time import time
from glob import glob
import pandas as pd
import numpy as np
import cv2
import os

# input_path = "C:\\Raghavendra\\PRASAD\\imagenet-object-localization-challenge"
input_path = "C:\\Raghavendra\\PRASAD\\imagenet-object-localization-challenge\\ILSVRC\\Data\\CLS-LOC\\"
output_path = "C:\\Raghavendra\\PRASAD\\XGBoost_classification\\data\\"
LOC_val_solution = "C:\\Raghavendra\\PRASAD\\imagenet-object-localization-challenge\\LOC_val_solution.csv"

def process_image(source_path, destination_path):
    target_size = 256

    input_image = cv2.imread(source_path)
    height, width = input_image.shape[0], input_image.shape[1]

    cv2_resize = 0; resize_height = target_size; resize_width = target_size
    if height > width:
        #cv2 resize has width x height
        resize_width = int((width / height) * target_size)
        cv2_resize = cv2.resize(input_image, (resize_width, target_size))
    if height < width:
        #cv2 resize has width x height
        resize_height = int((height / width) * target_size)
        cv2_resize = cv2.resize(input_image, (target_size, resize_height))

    output_image = np.zeros((target_size, target_size, 3))
    output_image[:resize_height, :resize_width, :] = cv2_resize
    cv2.imwrite(destination_path, output_image)

def os_mkdir_train():
    os.mkdir(output_path + "train")
    for each_folder in os.listdir(input_path + "train"):
        os.mkdir(output_path + "train\\" + each_folder)

    glob_input_path = glob(input_path + "*\\*\\*")
    for each_glob_input in tqdm(glob_input_path):
        parent_name = Path(each_glob_input).parent.name
        file_name = Path(each_glob_input).name.split(".")[0]
        process_image(each_glob_input, output_path + "train\\" + parent_name + "\\" + file_name + ".jpg")

def os_mkdir_test():
    os.mkdir(output_path + "test")
    df = pd.read_csv(LOC_val_solution)
    df["test_class"] = df["PredictionString"].str.strip().str.split(" ", expand=True)[0]
    for each_folder in set(df["test_class"]):
        os.mkdir(output_path + "test\\" + each_folder)

    dict_df = dict(zip(df["ImageId"], df["test_class"]))
    for each_image in tqdm(os.listdir(input_path + "val\\")):
        each_image_split = each_image.split(".")[0]
        test_class = dict_df[each_image_split]
        destination_path = output_path + "test\\" + test_class + "\\" + each_image_split + ".jpg"
        source_path = input_path + "val\\" + each_image
        process_image(source_path, destination_path)

t0 = time()
os_mkdir_train()
os_mkdir_test()
print(time() - t0, "seconds")
