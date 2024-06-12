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
from tata_tsr import main as tata_tsr_main
from jcap_tsr import main as jcap_tsr_main
from jsw_tsr import main as jsw_tsr_main
import easyocr

# Set image and model path
image_path = 'data/JCAPL/2584577.jpg'
config_file = 'table-detection-weights/maskrcnn/maskrcnn_dit_base.yaml'

ocr_model = easyocr.Reader(['en'], gpu=False)

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

def table_detection():
    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.merge_from_file(config_file)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.DEVICE = device
    predictor = DefaultPredictor(cfg)
    if image_path.endswith('.jpg') or image_path.endswith('.png') or image_path.endswith('.jpeg'):
        print('Read Image ', image_path)
        img = cv2.imread(image_path)
        imgname = image_path.split('/')[-1]
        img_name, img_ext = imgname.split('.')
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
        if 'jcap':
            img_table = image[int(y1):int(y2) , int(x1):int(x2)] 
            img_table = img_table[:, 5:-10]
            main_arr = jcap_tsr_main(image, table_bbox, ocr_model)
        elif 'jsw':
            img_table = image[int(y1):int(y2) , int(x1):int(x2)]
            main_arr = jsw_tsr_main(image, table_bbox, ocr_model)
        elif 'tata':
            img_table = image[int(y1):int(y2) , int(x1):int(x2)]
            main_arr = tata_tsr_main(image, table_bbox, ocr_model)
    return main_arr


main_arr = table_detection()
print('main array')
print(main_arr)
