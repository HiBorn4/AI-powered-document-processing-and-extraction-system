''' 
Installing Dependencies
!pip install torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
# %cd /content/drive/MyDrive/mahindra/test_certificates/MS-DiT
# !git clone https://github.com/microsoft/unilm.git
%cd /content/drive/MyDrive/mahindra/test_certificates/MS-DiT/unilm/dit
!pip install -r requirements.txt
!python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
!pip install shapely
!pip install timm
!pip install easyocr
!pip install opencv-python
!pip install numpy
!pip install pandas
'''
# Import Dependencies 
import os
import cv2
import sys
sys.path.append('/home/hi-born4/Bristlecone/AI-powered document processing and extraction system/unilm/dit/object_detection')
from ditod import add_vit_config
import torch
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
import numpy as np
from scipy.ndimage import interpolation as inter

# Set image and model path
image_path = '/home/hi-born4/Bristlecone/AI-powered document processing and extraction system/data/JSW_Steel/2360100-01-02.jpg'
config_file = '/home/hi-born4/Bristlecone/AI-powered document processing and extraction system/table-detection-weights/maskrcnn/maskrcnn_dit_base.yaml'
output_dir = 'table_extracted'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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
    corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
            borderMode=cv2.BORDER_REPLICATE)

    return best_angle, corrected

def table_detection(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} does not exist.")
        return None
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Failed to read image {image_path}. Check file path or file integrity.")
        return None
    
    print('Read Image ', image_path)
    imgname = image_path.split('/')[-1]
    img_name, img_ext = imgname.split('.')
    # Deskew Image
    print('Correct Skew')
    angle, image = correct_skew(img)
    # Save the deskewed image
    deskewed_image_path = os.path.join(output_dir, f"{img_name}_deskewed.{img_ext}")
    cv2.imwrite(deskewed_image_path, image)
    print(f"Deskewed image saved at: {deskewed_image_path}")
    
    # Table Detection
    print('Table Detection')
    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.merge_from_file(config_file)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.DEVICE = device
    predictor = DefaultPredictor(cfg)
    
    md = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    md.set(thing_classes=["table"])
    output = predictor(image)["instances"]
    scores = output.scores.to("cpu").numpy() if output.has("scores") else None
    bbox_list = output.pred_boxes.tensor.detach().to('cpu').numpy()
    max_index = np.argmax(scores)
    table_bbox = list(bbox_list[max_index])
    x1, y1, x2, y2 = table_bbox
    img_table = image[int(y1):int(y2) , int(x1):int(x2)] 
    # Save the detected table image
    table_image_path = os.path.join(output_dir, f"{img_name}_table.{img_ext}")
    cv2.imwrite(table_image_path, img_table)
    print(f"Table detected image saved at: {table_image_path}")
    return img_table

# Run table detection and save images
table_img = table_detection(image_path)
