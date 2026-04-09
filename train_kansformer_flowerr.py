############################################################################################################
# 1. 使用argparse类实现可以在训练的启动命令中指定超参数
# 2. 可以通过在启动命令中指定 --seed 来固定网络的初始化方式，以达到结果可复现的效果
# 3. 使用了更高级的学习策略 cosine warm up
# 4. 可以通过在启动命令中指定 --model 来选择使用的模型 
# 5. 使用amp包实现半精度训练
# 6. 实现了数据加载类的自定义实现 (Đã nâng cấp sang DatasetMarr 5-Fold)
############################################################################################################

import os
import argparse
import math
import shutil
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

import classic_models
from utils.lr_methods import warmup
from utils.train_engin import train_one_epoch, evaluate

# NHÚNG DATASET MỚI VÀO ĐÂY
from dataset_wbc import DatasetMarr, labels_map

parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=15, help='the number of classes')  # 15 loại tế bào
parser.add_argument('--epochs', type=int, default=50, help='the number of training epoch')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size for training')
parser.add_argument('--lr', type=float, default=0.0002, help='star learning rate')
parser.add_argument('--lrf', type=float, default=0.0001, help='end learning rate')
parser.add_argument('--seed', default=False, action='store_true', help='fix the initialization of parameters')
parser.add_argument('--tensorboard', default=False, action='store_true', help=' use tensorboard for visualization')
parser.add_argument('--use_amp', default=False, action='store_true', help=' training with mixed precision') 
parser.add_argument('--data_path', type=str, default="./data/AML_LMU")
parser.add_argument('--model', type=str, default="kansformer1", help=' select a model for training')
parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
# THÊM THAM SỐ CHẠY FOLD
parser.add_argument('--fold', type=int, default=0, help='Chạy Fold số mấy (0-4)')

opt = parser.parse_args()

if opt.seed:
    def seed_torch(seed=7):
        random.seed(seed)  
        os.environ['PYTHONHASHSEED'] = str(seed)  
        np.random.seed(seed)  
        torch.manual_seed(seed)  
        torch.cuda.manual_seed(seed)  
        torch.cuda.manual_seed_all(seed)  
        print('random seed has been fixed')
    seed_torch()


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(args)

    if opt.tensorboard:
        log_path = os.path.join('./results/tensorboard', args.model)
        print('Start Tensorboard with "tensorboard --logdir={}"'.format(log_path))
        if os.path.exists(log_path) is False:
            os.makedirs(log_path)
            print("tensorboard log save in {}".format(log_path))
        else:
            shutil.rmtree(log_path) 
        tb_writer = SummaryWriter(log_path)

    # KHUÔN TRANSFORM CHUẨN CỦA TÁC GIẢ
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # ====================================================================================
    # GỌI DATASET MỚI (DATASET_MARR TỪ FILE CSV)
    # ====================================================================================
    print("\n" + "="*50)
    print(f"[*] ĐANG HUẤN LUYỆN TRÊN FOLD SỐ: {args.fold}")
    print("="*50 + "\n")

    train_dataset = DatasetMarr(dataroot=args.data_path,
                                dataset_selection="AML_LMU",
                                labels_map=labels_map,
                                fold=args.fold,
                                transform=data_transform["train"],
                                state='train',
                                is_hsv=True,  # Bật tăng cường màu
                                is_hed=True)  # Bật tăng cường màu thuốc nhuộm H&E
                                
    val_dataset = DatasetMarr(dataroot=args.data_path,
                              dataset_selection="AML_LMU",
                              labels_map=labels_map,
                              fold=args.fold,
                              transform=data_transform["val"],
                              state='validation')

    #nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  
    #print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                               pin_memory=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                                             pin_memory=True, num_workers=4)

    # create model
    model = classic_models.find_model_using_name(opt.model, num_classes=opt.num_classes).to(device)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(pg, lr=args.lr)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    best_acc = 0.

    # save parameters path
    save_path = os.path.join(os.getcwd(), 'results/weights', args.model)
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for epoch in range(args.epochs):
        # train
        mean_loss, train_acc = train_one_epoch(model=model, optimizer=optimizer, data_loader=train_loader,
                                               device=device, epoch=epoch, use_amp=args.use_amp, lr_method=warmup)
        scheduler.step()
        # validate
        val_acc = evaluate(model=model, data_loader=val_loader, device=device)

        print('[epoch %d] train_loss: %.3f  train_acc: %.3f  val_accuracy: %.3f' % (
        epoch + 1, mean_loss, train_acc, val_acc))
        
        # Cập nhật log file cho từng Fold
        log_file = os.path.join(save_path, f"kansformer_AML_fold{args.fold}_log.txt")
        with open(log_file, 'a') as f:
            f.writelines('[epoch %d] train_loss: %.3f  train_acc: %.3f  val_accuracy: %.3f' % (
            epoch + 1, mean_loss, train_acc, val_acc) + '\n')

        if opt.tensorboard:
            tags = ["train_loss", "train_acc", "val_accuracy", "learning_rate"]
            tb_writer.add_scalar(tags[0], mean_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], val_acc, epoch)
            tb_writer.add_scalar(tags[3], optimizer.param_groups[0]["lr"], epoch)

        # CẬP NHẬT TÊN FILE .PTH THEO TỪNG FOLD ĐỂ KHÔNG BỊ ĐÈ
        if val_acc > best_acc:
            best_acc = val_acc
            save_name = f"kansformer_AML_fold{args.fold}.pth"
            torch.save(model.state_dict(), os.path.join(save_path, save_name))

if __name__ == '__main__':
    main(opt)