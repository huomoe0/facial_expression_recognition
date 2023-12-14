import argparse
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils


def main(args):
    print(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    print('Start Tensorboard with "tensorboard --logdir ' + args.logs)
    writer = SummaryWriter(args.logs)

    train_loader, train_num, validate_loader, val_num = utils.getDataLoader(args.data_path, args.batch_size, args.nw)

    model = torchvision.models.densenet121(pretrained=True)
    print(model)
    in_channel = model.classifier.in_features
    model.classifier = nn.Linear(in_channel, args.num_classes)
    model.to(device)

    loss_function = nn.CrossEntropyLoss()
    pg = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4, nesterov=True)

    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best_acc, best_precision, best_recall, total_epoch = utils.load_record(args.record_path)

    for epoch in range(args.epochs):
        # train
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = loss_function(output, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, args.epochs, loss)
        scheduler.step()

        # validate
        val_accurate, precision, recall, val_loss = utils.validate(model, device, validate_loader, loss_function,
                                                                   val_num)

        print('[epoch %d] train_loss: %.3f  val_loss: %.3f  val_accuracy: %.3f  val_precision: %.3f  val_recall: %.3f' %
              (epoch + 1, running_loss, val_loss, val_accurate, precision, recall))

        if val_accurate > best_acc:
            best_acc = val_accurate
            best_precision = precision
            best_recall = recall
            torch.save(model.state_dict(), args.weights)

        utils.add2tensorboard(writer, total_epoch, running_loss, val_loss, val_accurate, precision, recall, optimizer)
        total_epoch += 1
        utils.save_record(args.record_path, best_acc, best_precision, best_recall, total_epoch)

    print(f'Finished Training  best_acc: {best_acc} precision: {best_precision} recall: {best_recall}')
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", required=False)
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--nw', type=int, default=1)
    parser.add_argument('--data-path', type=str, default="data/RAF-DB")
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--lrf', type=float, default=0.1)
    # 常用超参数
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weights', type=str, default='densenet121.pth', help='initial weights path')
    parser.add_argument('--logs', type=str, default='logs', help='log dir')
    parser.add_argument('--record_path', type=str, default='densenet121.txt')
    opt = parser.parse_args()
    main(opt)
