from torch.utils.data import DataLoader
from model import *
import config
import json

"""
Xây dựng hàm train
"""
def training_time(train_config, valid_config, data_json_train, data_json_valid):
    train_data = LoadDataset(train_config, data_json_train)
    valid_data = LoadDataset(valid_config, data_json_valid)
    train_loader = DataLoader(train_data, batch_size=train_config['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=valid_config['batch_size'])

    model = init_model_resnet18(train_config)
    loss_fn = intit_loss()
    optimizer = init_optimizeer(model, train_config)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)

    highest_acc = 0
    if not os.path.exists(train_config['model_save_path']):
        os.mkdir(train_config['model_save_path'])

    for epoch in range(train_config['epoch']):
        print()
        model.train()
        for batch_idx, (data, target_gender, target_mask) in enumerate(train_loader):
            data, target_gender, target_mask = data.to(device), target_gender.to(device), target_mask.to(device)
            optimizer.zero_grad()
            output_gender, output_mask = model(data)
            loss_gender = loss_fn(output_gender, target_gender)
            loss_mask = loss_fn(output_mask, target_mask)
            loss = loss_gender * 0.4 + loss_mask * 0.6
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, loss.item()))
                print("Loss gender:", loss_gender.item())
                print("Loss mask:", loss_mask.item())

        model.eval()
        correct_gender = 0
        correct_mask = 0
        for (data, target_gender, target_mask) in valid_loader:
            data, target_gender, target_mask = data.to(device), target_gender.to(device), target_mask.to(device)
            output_gender, output_mask = model(data)
            
            pred_gender = output_gender.argmax(dim=1, keepdim=True)
            target_gender = target_gender.argmax(dim=1, keepdim=True)
            correct_gender += pred_gender.eq(target_gender.view_as(pred_gender)).sum().item()
            
            pred_mask = output_mask.argmax(dim=1, keepdim=True)
            target_mask = target_mask.argmax(dim=1, keepdim=True)
            correct_mask += pred_mask.eq(target_mask.view_as(pred_mask)).sum().item()

        print(str(correct_gender) + '/' + str(len(valid_loader.dataset)))
        accuracy_gender = 100. * correct_gender / len(valid_loader.dataset)
        print('\nValid set: Accuracy_gender: {}/{} ({:.0f}%)\n'.format(correct_gender, len(valid_loader.dataset), accuracy_gender))
        
        print(str(correct_mask) + '/' + str(len(valid_loader.dataset)))
        accuracy_mask = 100. * correct_mask / len(valid_loader.dataset)
        print('\nValid set: Accuracy_mask: {}/{} ({:.0f}%)\n'.format(correct_mask, len(valid_loader.dataset), accuracy_mask))

        accuracy = (accuracy_gender + accuracy_mask) / 2
        if accuracy >= highest_acc:
            highest_acc = accuracy
            torch.save(model.state_dict(), os.path.join(train_config['model_save_path'], f'training_epoch_{epoch}.pth'))
            print(f"Saving best model to {os.path.join(train_config['model_save_path'], f'training_epoch_{epoch}.pth')}")


if __name__ == '__main__':
    with open("/data/disk2/vinhnguyen/AnyFace/data/labels_train.json", 'r') as file:
        data_json_train = json.load(file)
        
    with open("/data/disk2/vinhnguyen/AnyFace/data/labels_valid.json", 'r') as file:
        data_json_valid = json.load(file)
        
    training_time(config.Train_Config, config.Valid_Config, data_json_train=data_json_train, data_json_valid=data_json_valid)
