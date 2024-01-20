import torch
import os
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from PIL import Image
from torchvision.models import *
import json
import numpy as np
import torch.nn as nn

def is_image(filename):
    return any(filename.endswith(end) for end in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def one_hot_label(num_class, index_class):
    label = [0 for i in range(num_class)]
    label[index_class] = 1
    return torch.Tensor(label)


"""
Xây dựng hàm DataLoader
"""
class LoadDataset(Dataset):
    def __init__(self, config, data_json):
        super(LoadDataset, self).__init__()
        self.config = config
        self.data_json = data_json
        self.num_class = config['class']['num']
        self.images = []

        for img_name in os.listdir(self.config['path']):
            img_path = os.path.join(self.config['path'], img_name)
            self.images.append(img_path)

        self.transform = Compose([
            Resize(self.config['image_size']),
            transforms.RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        img_name = self.images[index].split('/')[-1]
        original_image = Image.open(self.images[index])
        if original_image.mode != 'RGB':
            original_image = original_image.convert('RGB')

        x, y, w, h = map(int, json.loads(self.data_json[img_name][0]))
        image_array = np.array(original_image)
        cropped_image = image_array[y:y+h, x:x+w]
        cropped_image = Image.fromarray(cropped_image)

        cropped_image = self.transform(cropped_image)

        class_gender = self.data_json[img_name][6]
        class_mask = self.data_json[img_name][3]
        label_gennder = one_hot_label(self.num_class, self.config['class']['name_gender'].index(class_gender))
        label_mask = one_hot_label(self.num_class, self.config['class']['name_mask'].index(class_mask))
        
        return cropped_image, label_gennder, label_mask

    def __len__(self):
        return len(self.images)


"""
Xây dựng hàm loss
"""
def intit_loss():
    loss = torch.nn.CrossEntropyLoss()
    return loss


"""
Xây dựng hàm Optimizer
"""
def init_optimizeer(model, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    return optimizer


"""
Xây dựng model Resnet18
"""
backbone = torch.nn.Sequential(*list(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).children())[:-1])

class ClsfModel(nn.Module):
    def __init__(self, config):
        super(ClsfModel, self).__init__()

        self.config = config
        self.backbone = backbone
        self.head_gender = nn.Sequential(
            torch.nn.Flatten(),
            nn.Linear(512, config['class']['num']),
            nn.Softmax(dim=1)
          )

        self.head_mask = nn.Sequential(
            torch.nn.Flatten(),
            nn.Linear(512, config['class']['num']),
            nn.Softmax(dim=1)
          )

    def forward(self, x):
        features = self.backbone(x)
        output_gender = self.head_gender(features)
        output_mask = self.head_mask(features)
        return output_gender, output_mask


def init_model_resnet18(config):
    model = ClsfModel(config)
    if config['load_checkpoint'] is not None:
        print("Loading Resnet18 model checkpoint from " + config['load_checkpoint'] + ' ...')
        model.load_state_dict(torch.load(config['load_checkpoint']))
    
    return model