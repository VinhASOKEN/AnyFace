from pathlib import Path
import sys
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

import cv2
import numpy as np
import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from PIL import Image
from detect_face.tools.detector import FaceDetector
from tqdm import tqdm

from model_age.model import init_model_resnet18 as init_model_age
import model_age.config as config_age
from model_emotional.model import init_model_resnet18 as init_model_emotional
import model_emotional.config as config_emotional
from model_gender_mask.model import init_model_resnet18 as init_model_gender_mask
import model_gender_mask.config as config_gender_mask
from model_race.model import init_model_resnet18 as init_model_race
import model_race.config as config_race
from model_skintone.model import init_model_resnet18 as init_model_skintone
import model_skintone.config as config_skintone


if __name__ == "__main__":
    img_folder = "/data/disk2/vinhnguyen/AnyFace/test_images"
    output_folder = "/data/disk2/vinhnguyen/AnyFace/result_images"
    
    detector = FaceDetector("/data/disk2/vinhnguyen/AnyFace/detect_face/weights/mobilev3.onnx")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model_age = init_model_age(config_age.Testing_Config).to(device)
    model_age.eval()
    model_emotional = init_model_emotional(config_emotional.Testing_Config).to(device)
    model_emotional.eval()
    model_gender_mask = init_model_gender_mask(config_gender_mask.Testing_Config).to(device)
    model_gender_mask.eval()
    model_race = init_model_race(config_race.Testing_Config).to(device)
    model_race.eval()
    model_skintone = init_model_skintone(config_skintone.Testing_Config).to(device)
    model_skintone.eval()

    transform = Compose([
            Resize(config_age.Testing_Config['image_size']),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
        ])
    

    for img_name in tqdm(os.listdir(img_folder)):
        img_path = os.path.join(img_folder, img_name)
        image_cv2 = cv2.imread(img_path)

        faceobjects = detector.DetectFace(image_cv2)
        
        if(len(faceobjects) > 0):
            for i in range(len(faceobjects)):
                x, y, w, h = faceobjects[i].rect.x, faceobjects[i].rect.y, faceobjects[i].rect.w, faceobjects[i].rect.h
                cv2.rectangle(image_cv2, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                
                image_face= image_cv2[int(y):int(y+h), int(x):int(x+w)]
                numpy_image_face = np.array(image_face)
                pil_image_face = Image.fromarray(cv2.cvtColor(numpy_image_face, cv2.COLOR_BGR2RGB))
                pil_image_face = transform(pil_image_face).unsqueeze(0).to(device)
                
                pred_age = model_age(pil_image_face)
                pred_emotional = model_emotional(pil_image_face)
                pred_gender, pred_mask = model_gender_mask(pil_image_face)
                pred_race = model_race(pil_image_face)
                pred_skintone = model_skintone(pil_image_face)
                
                class_age_pred = config_age.Class_Info['name'][pred_age.argmax(dim=1)]
                class_emotional_pred = config_emotional.Class_Info['name'][pred_emotional.argmax(dim=1)]
                class_gender_pred = config_gender_mask.Class_Info['name_gender'][pred_gender.argmax(dim=1)]
                class_mask_pred = config_gender_mask.Class_Info['name_mask'][pred_mask.argmax(dim=1)]
                class_race_pred = config_race.Class_Info['name'][pred_race.argmax(dim=1)]
                class_skintone_pred = config_skintone.Class_Info['name'][pred_skintone.argmax(dim=1)]
                
                info_str = (
                            f"BBox            : {x}, {y}, {w}, {h}\n"
                            f"Class Age       : {class_age_pred}\n"
                            f"Class Emotional : {class_emotional_pred}\n"
                            f"Class Gender    : {class_gender_pred}\n"
                            f"Class Mask      : {class_mask_pred}\n"
                            f"Class Race      : {class_race_pred}\n"
                            f"Class Skintone  : {class_skintone_pred}\n"
                            )

                txt_filename = os.path.splitext(img_name)[0] + ".txt"
                txt_path = os.path.join(output_folder, txt_filename)
                with open(txt_path, 'w') as txt_file:
                    txt_file.write(info_str)
                    
            output_img_path = os.path.join(output_folder, img_name)
            cv2.imwrite(output_img_path, image_cv2)
