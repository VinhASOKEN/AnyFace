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
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, loss.item()))

        model.eval()
        correct = 0
        for (data, target) in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            target = target.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = 100. * correct / len(valid_loader.dataset)
        print('Valid set: Accuracy: {}/{} ({:.0f}%)'.format(correct, len(valid_loader.dataset), accuracy))

        if accuracy >= highest_acc:
            highest_acc = accuracy
            torch.save(model.state_dict(), os.path.join(train_config['model_save_path'], f'training_epoch_{epoch}.pth'))
            print(f"Saving best model to {os.path.join(train_config['model_save_path'], f'training_epoch_{epoch}.pth')}")
            print()


if __name__ == '__main__':
    with open("/data/disk2/vinhnguyen/AnyFace/data/labels_train.json", 'r') as file:
        data_json_train = json.load(file)
        
    with open("/data/disk2/vinhnguyen/AnyFace/data/labels_valid.json", 'r') as file:
        data_json_valid = json.load(file)
        
    training_time(config.Train_Config, config.Valid_Config, data_json_train=data_json_train, data_json_valid=data_json_valid)