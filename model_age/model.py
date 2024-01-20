import torch
import os
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from PIL import Image
from torchvision.models import *
import json
import numpy as np

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

        class_name = self.data_json[img_name][1]
        label = one_hot_label(self.num_class, self.config['class']['name'].index(class_name))

        return cropped_image, label

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
def init_model_resnet18(config):
    backbone = torch.nn.Sequential(*list(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).children())[:-1])

    model = torch.nn.Sequential(
        backbone,
        torch.nn.Flatten(),
        torch.nn.Linear(512, config['class']['num']),
        torch.nn.Softmax(dim=1)
    )

    if config['load_checkpoint'] is not None:
        print("Loading Resnet18 model checkpoint from " + config['load_checkpoint'] + ' ...')
        model.load_state_dict(torch.load(config['load_checkpoint']))

    return model