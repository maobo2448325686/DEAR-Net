import torch
import torch.nn as nn

from ptflops import get_model_complexity_info

from data_loader_new import Dataset_self
from data_utils import calMetric_iou
import numpy as np
from mode.DEARNet import DEARNet
from tools.eval import Statistics, Calculation
import time
from tqdm import tqdm
from PIL import Image

import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def img(pr, filename, gt):
    save_dir = r""
    # 将 pr 和 gt 转换为二维 numpy 数组
    pr = pr[0, 0]
    gt = gt[0, 0].cpu().numpy()

    index_tp = np.where(np.logical_and(pr == 1, gt == 1))
    index_fp = np.where(np.logical_and(pr == 1, gt == 0))
    index_tn = np.where(np.logical_and(pr == 0, gt == 0))
    index_fn = np.where(np.logical_and(pr == 0, gt == 1))

    # 创建一个三通道的空白图像
    map = np.zeros([gt.shape[0], gt.shape[1], 3], dtype=np.uint8)

    # 使用 zip 将行索引和列索引组合成二维索引
    for i, j in zip(index_tp[0], index_tp[1]):
        map[i, j] = [255, 255, 255]  # white
    for i, j in zip(index_fp[0], index_fp[1]):
        map[i, j] = [255, 0, 0]  # red
    for i, j in zip(index_tn[0], index_tn[1]):
        map[i, j] = [0, 0, 0]  # black
    for i, j in zip(index_fn[0], index_fn[1]):
        map[i, j] = [0, 255, 0]  # Cyan

    # 将 numpy 数组转换为 PIL 图像并保存
    change_map = Image.fromarray(map)
    save_path = os.path.join(save_dir, filename[0])  # 使用 os.path.join 拼接路径
    change_map.save(save_path)


def test(input_a, modelfile, network_name, BATCH_SIZE=1):
    data_loader_test_img = torch.utils.data.DataLoader(dataset=input_a,
                                                       batch_size=1, shuffle=True)

    model = network_name()
    model.eval()

    model.load_state_dict(torch.load(modelfile)['model'])
    if torch.cuda.is_available():
        model = model.cuda()

    # 计算参数量和flops
    # 自定义输入构造器：返回一个元组 (img1, img2)
    # wrapped = WrappedDEARNet(model).eval()
    #
    # macs, params = get_model_complexity_info(
    #     wrapped,
    #     input_res=(6, 256, 256),  # 两个 3 通道图像拼接在一起
    #     as_strings=False,
    #     print_per_layer_stat=False
    # )
    # # 转换为 G 和 M
    # flops_G = macs * 0.5 / 1e9
    # params_M = params / 1e6
    # print(f'Params: {params_M:.2f} M, FLOPs: {flops_G:.2f} G')

    test_bar = tqdm(data_loader_test_img)
    inter, unin = 0, 0
    matrix_all = np.array([0, 0, 0, 0])
    res = []
    valing_results = {'loss': 0, 'SR_loss': 0, 'CD_loss': 0, 'batch_sizes': 0, 'IoU': 0}
    for img1, img2, labels, filename in test_bar:
        valing_results['batch_sizes'] += 1

        data1 = img1.to(device, dtype=torch.float)
        data2 = img2.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)
        labels = torch.argmax(labels, 1).unsqueeze(1).float()

        gt_value = labels.float()

        torch.cuda.synchronize()
        start_time = time.time()
        dist = model(data1, data2)
        torch.cuda.synchronize()
        end_time = time.time()
        res.append(end_time - start_time)

        prob = (dist > 0.5).float()
        prob = prob.cpu().detach().numpy()
        # gt_value = gt_value.cpu().detach().numpy()
        img(prob, filename, gt_value)

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

        ### evaluation
        matrix = Statistics(gt_value, result)
        matrix_all = matrix + matrix_all

    acc, recall, precision, f1, K = Calculation(matrix_all)

    time_sum = 0
    for i in res:
        time_sum += i
    fps = 1.0 / (time_sum / len(res))

    # param_count = count_parameters(model)
    print("OA:", acc, "P值:", precision, "R值:", recall, "F1值:", f1, "kappa值:", K)


# class WrappedDEARNet(nn.Module):
#     def __init__(self, original_model):
#         super().__init__()
#         self.model = original_model
#
#     def forward(self, x):
#         # 把单输入拆成两个输入
#         img1, img2 = x.chunk(2, dim=1)  # 假设 x 的通道数是 6
#         return self.model(img1, img2)

if __name__ == '__main__':

    data_A_dir = 'Data/levir'
    data_test_dir = data_A_dir + r'\test\images'
    network_name = DEARNet
    batch_size = 1

    # ### test levir_CD
    modelfile = None
    input_test = Dataset_self(data_test_dir)
    test(input_test, modelfile, network_name, BATCH_SIZE=1)
