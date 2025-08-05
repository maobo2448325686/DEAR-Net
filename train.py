import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from data_loader_new import Dataset_self
from data_utils import calMetric_iou
import numpy as np
from models.dearnet import DEARNet
from utils.eval import Statistics, Calculation
import itertools
from tqdm import tqdm
from models import diceLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(input_train, input_val, model_dir, load_model, network_name, batch_size, arg_lab=False,
          EPOCH=100, LR=0.0001):

    data_loader_train_img = torch.utils.data.DataLoader(dataset=input_train,
                                                        batch_size=batch_size, shuffle=True, num_workers=4)
    data_loader_test_img = torch.utils.data.DataLoader(dataset=input_val,
                                                           batch_size=1, shuffle=True, num_workers=4)
    model = network_name.to(device, dtype=torch.float)

    # set optimization
    optimizer = optim.Adam(itertools.chain(model.parameters()), lr=LR, betas=(0.9, 0.999))

    if load_model:
        checkpoint = torch.load(load_model)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        print('加载 epoch {} 成功！'.format(start_epoch))
    else:
        start_epoch = 0

    if torch.cuda.is_available():
        model = model.cuda()
    alpha = 1
    loss_func2 = nn.BCELoss().cuda()
    dice_loss = diceLoss.DiceLoss().cuda()
    if arg_lab:
        model_name = '_lab_model.pkl'
    else:
        model_name = '_model.pkl'
    f1_best = 0.
    # training
    for epoch in range(start_epoch + 1, EPOCH + 1):
        train_bar = tqdm(data_loader_train_img)
        running_results = {'batch_sizes': 0, 'SR_loss': 0, 'CD_loss': 0, 'loss': 0}
        model.train()
        for data1, data2, labels, filename in train_bar:
            running_results['batch_sizes'] += batch_size

            data1 = data1.to(device, dtype=torch.float).cuda()
            data2 = data2.to(device, dtype=torch.float).cuda()
            labels = labels.to(device, dtype=torch.float)
            labels = torch.argmax(labels, 1).unsqueeze(1).float().cuda()

            dist = model(data1, data2)
            # calculate IoU
            CD_loss = alpha * loss_func2(dist, labels) + (1 - alpha) * dice_loss(dist, labels)

            model.zero_grad()
            CD_loss.backward()
            optimizer.step()

            # loss for current batch before optimization
            running_results['CD_loss'] += CD_loss.item() * batch_size

            train_bar.set_description(
                desc='[%d/%d] loss: %.4f' % (
                    epoch, EPOCH,
                    running_results['CD_loss'] / running_results['batch_sizes']))

        # model saving and val set evaluation
        if epoch % 1 == 0:
            output_name = model_dir + '/data_name_' + str(epoch) + model_name
            best_output_name = model_dir + "BEST_" + str(epoch) + model_name
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, output_name)

            model.eval()
            test_bar = tqdm(data_loader_test_img)
            inter, unin = 0, 0
            matrix_all = np.array([0, 0, 0, 0])
            valing_results = {'loss': 0, 'SR_loss': 0, 'CD_loss': 0, 'batch_sizes': 0, 'IoU': 0}
            for img1, img2, labels_test, filename in test_bar:
                valing_results['batch_sizes'] += 1

                img1 = img1.to(device, dtype=torch.float)
                img2 = img2.to(device, dtype=torch.float)
                labels_test = labels_test.to(device, dtype=torch.float)
                labels_test = torch.argmax(labels_test, 1).unsqueeze(1).float()

                dist = model(img1, img2)
                # calculate IoU
                gt_value = (labels_test > 0).float()
                prob = (dist > 0.5).float()
                prob = prob.cpu().detach().numpy()

                gt_value = gt_value.cpu().detach().numpy()
                gt_value = np.squeeze(gt_value)
                result = np.squeeze(prob)
                intr, unn = calMetric_iou(gt_value, result)
                inter = inter + intr
                unin = unin + unn

                # loss for current batch before optimization
                valing_results['IoU'] = (inter * 1.0 / unin)

                test_bar.set_description(
                    desc='IoU: %.4f' % (valing_results['IoU'],
                                        ))

                # evaluation
                matrix = Statistics(gt_value, result)
                matrix_all = matrix + matrix_all

            acc, recall, precision, f1, K = Calculation(matrix_all)

            if f1 > f1_best:
                f1_best = f1
                torch.save(state, best_output_name)

            print("OA:", acc, "P值:", precision, "R值:", recall, "F1值:", f1, "kappa值:", K)
            del img1, img2, labels_test
        del data1, data2, labels, filename


if __name__ == '__main__':
    ############### training data path and model dir
    data_A_dir = r'Your data path'
    data_train_dir = data_A_dir + r'/train/images'
    data_val_dir = data_A_dir + r'/val/images'
    data_test_dir = data_A_dir + r'/test/images'
    network_name = DEARNet(backbone_name="pvtv2")
    batch_size = 4

    ### train&val
    pth_path = 'Your pth path'
    # 检查路径是否存在
    if not os.path.exists(pth_path):
        # 如果路径不存在，则创建路径
        os.makedirs(pth_path)
        print(f"路径 {pth_path} 已创建。")
    else:
        print(f"路径 {pth_path} 已存在。")

    modelfile = False
    input_train = Dataset_self(data_train_dir)
    input_val = Dataset_self(data_val_dir)
    train(input_train, input_val, pth_path, modelfile, network_name, batch_size,
          EPOCH=100, LR=0.0001)
