import os

import torch
from sklearn.metrics import precision_score, recall_score
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm

mean = [0.5924, 0.5461, 0.5246]
std = [0.3553, 0.3625, 0.3698]

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomRotation(10),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean, std)]),
    "val": transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize(mean, std)])}


def getDataLoader(data_path, batch_size, nw):
    train_dataset = datasets.ImageFolder(root=os.path.join(data_path, "train"), transform=data_transform["train"])
    train_num = len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=nw)
    validate_dataset = datasets.ImageFolder(root=os.path.join(data_path, "val"), transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)
    print('Using {} dataloader workers every process'.format(nw))
    print("using {} images for training, {} images for validation.".format(train_num, val_num))
    return train_loader, train_num, validate_loader, val_num


def init_model(model, weights, freeze_layers):
    if os.path.exists(weights):
        model.load_state_dict(torch.load(weights))

    # 是否冻结权重
    if freeze_layers:
        for name, para in model.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "classifier" not in name:
                para.requires_grad_(False)


def add2tensorboard(writer, epoch, running_loss, val_loss, acc, precision, recall, optimizer):
    tags = ["train_loss", "val_loss", "accuracy", "precision", "recall", "learning_rate"]
    writer.add_scalar(tags[0], running_loss, epoch)
    writer.add_scalar(tags[1], val_loss, epoch)
    writer.add_scalar(tags[2], acc, epoch)
    writer.add_scalar(tags[3], precision, epoch)
    writer.add_scalar(tags[4], recall, epoch)
    writer.add_scalar(tags[5], optimizer.param_groups[0]["lr"], epoch)


def load_record(path):
    best_acc = 0
    total_epoch = 0
    best_precision = 0
    best_recall = 0
    if os.path.exists(path):
        with open(path, "r") as f:
            best_acc = float(f.readline())
            best_precision = float(f.readline())
            best_recall = float(f.readline())
            total_epoch = int(f.readline())
    return best_acc, best_precision, best_recall, total_epoch


def save_record(path, acc, precision, recall, epoch):
    with open(path, "w") as f:
        f.write(f"{acc}\n")
        f.write(f"{precision}\n")
        f.write(f"{recall}\n")
        f.write(f"{epoch}")


def validate(model, device, validate_loader, loss_function, val_num):
    val_steps = len(validate_loader)
    model.eval()
    acc = 0
    precision = 0
    recall = 0
    val_loss = 0
    with torch.no_grad():
        val_bar = tqdm(validate_loader)
        for val_data in val_bar:
            val_images, val_labels = val_data
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            outputs = model(val_images)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, val_labels).sum().item()
            # 验证损失
            loss = loss_function(outputs, predict_y)
            val_loss += loss.item()
            preds = predict_y.cpu().numpy()
            labels = val_labels.cpu().numpy()
            precision += precision_score(labels, preds, average='macro', zero_division=1)
            recall += recall_score(labels, preds, average='macro', zero_division=1)

    val_accurate = acc / val_num
    precision /= val_steps
    recall /= val_steps

    return val_accurate, precision, recall, val_loss
