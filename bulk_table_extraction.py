# # Installing Dependencies
# !pip install torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
# # %cd /content/drive/MyDrive/mahindra/test_certificates/MS-DiT
# # !git clone https://github.com/microsoft/unilm.git
# %cd /content/drive/MyDrive/mahindra/test_certificates/MS-DiT/unilm/dit
# !pip install -r requirements.txt
# !python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
# !pip install shapely
# !pip install timm
# !pip install easyocr
# !pip install opencv-python
# !pip install numpy
# !pip install pandas

# Import Dependencies 
import os
import cv2
import sys
import json
import torch
import numpy as np
from scipy.ndimage import interpolation as inter
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import setup_logger
import easyocr

setup_logger()

sys.path.append('/home/hi-born4/Bristlecone/AI-powered document processing and extraction system/unilm/dit/object_detection')
from unilm.dit.object_detection.ditod import add_vit_config
from tata_tsr import main as tata_tsr_main
from jcap_tsr import main as jcap_tsr_main
from jsw_tsr import main as jsw_tsr_main

def correct_skew(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return best_angle, corrected

def table_detection(image_path, config_file):
    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.merge_from_file(config_file)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.DEVICE = device
    predictor = DefaultPredictor(cfg)
    if image_path.endswith(('.jpg', '.png', '.jpeg')):
        print('Read Image ', image_path)
        img = cv2.imread(image_path)
        imgname = os.path.basename(image_path)
        img_name, img_ext = os.path.splitext(imgname)
        # Deskew Image
        print('Correct Skew')
        angle, image = correct_skew(img)
        # Table Detection
        print('Table Detection')
        md = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        md.set(thing_classes=["table"])
        output = predictor(image)["instances"]
        scores = output.scores.to("cpu").numpy() if output.has("scores") else None
        bbox_list = output.pred_boxes.tensor.detach().to('cpu').numpy()
        max_index = np.argmax(scores)
        table_bbox = list(bbox_list[max_index])
        x1, y1, x2, y2 = table_bbox
        img_table = image[int(y1):int(y2), int(x1):int(x2)]
        
        return img_table

def process_images_in_directory(directory_path, config_file):
    output_dir = os.path.join(directory_path, 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(directory_path):
        if file_name.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(directory_path, file_name)
            print(f'Processing {image_path}')
            img_table = table_detection(image_path, config_file)

            # Save the table image
            table_image_path = os.path.join(output_dir, f'{os.path.splitext(file_name)[0]}_table.jpg')
            cv2.imwrite(table_image_path, img_table)

# Set directory and model paths
directory_path = 'data/jsw/new_imgs'
config_file = 'table-detection-weights/maskrcnn/maskrcnn_dit_base.yaml'

process_images_in_directory(directory_path, config_file)
