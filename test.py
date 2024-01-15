import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import pathlib
import matplotlib.pyplot as plt
import os
import pathlib
import pprint
import copy
import SimpleITK as sitk
import numpy as np
import pandas as pd
import torch
import skimage
from PIL import Image

import yaml
from matplotlib import pyplot as plt
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from torch.autograd import Variable
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureType
)
import pandas as pd
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import yaml
from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
# torch.cuda.set_device(1)
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM
from dataprocess.tumor import get_datasets, get_test_datasets
from models.unet3d import UNet
# from models.szgv1 import SZG_model
from utils import reload_ckpt_bis, \
    count_parameters, save_args_1, generate_segmentations_monai, inference, post_trans

parser = argparse.ArgumentParser(description='szg')
parser.add_argument('--devices', default='0', type=str, help='Set the CUDA_VISIBLE_DEVICES env var from this string')
parser.add_argument('--text_folder', default="../Rectal_Tumor/train", type=str)
parser.add_argument('--model_folder',default="/Result/unet",type=str)
parser.add_argument('--model_best_name',default="unet_end.pth.tar",type=str)
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2).')
parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='mini-batch size (default: 1)')
parser.add_argument('--warm', default=3, type=int, help="number of warming up epochs")
parser.add_argument('--val', default=1, type=int, help="how often to perform validation step")
parser.add_argument('--fold', default=0, type=int, help="Split number (0 to 4)")
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None, nargs='+',)
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')


class SemanticSegmentationTarget:
    def __init__(self, mask):
        # self.mask = torch.from_numpy(mask)
        self.mask = mask
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        # print("model_output.shape, self.mask.shape",model_output.shape, self.mask.shape)
        # model_output.shape, self.mask.shape torch.Size([1, 16, 320, 320]) torch.Size([1, 320, 320])
        return (model_output[:,:,:,:] * self.mask).sum()
def main(args):
    post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold_values=True)])
    # setup
    ngpus = torch.cuda.device_count()
    print(f"Working with {ngpus} GPUs")
    args.save_folder_1 = pathlib.Path(f"./{args.model_folder}")
    args.save_folder_1.mkdir(parents=True, exist_ok=True)
    args.checkpoint = args.save_folder_1 / args.model_best_name

    # Create model
    model_1 = UNet().cuda()
    # 如果有预训练，就加这个函数
    # model_1.load_from(args)

    bench_dataset = get_test_datasets(args.text_folder,args.seed, fold_number=args.fold)
    bench_loader = torch.utils.data.DataLoader(bench_dataset, batch_size=1, num_workers=args.workers)
    print("Bench Test dataset number of batch:", len(bench_loader))
    model_1 = model_1.eval()
    # with torch.no_grad():
    reload_ckpt_bis(f'{os.path.join(args.checkpoint)}', model_1, device)
    for idx, val_data in enumerate(bench_loader):
        patient_id = val_data["patient_id"][0]
        ref_path = val_data["seg_path"][0]
        ref_seg_img = sitk.ReadImage(ref_path)
        ref_seg = sitk.GetArrayFromImage(ref_seg_img)
        # print('val_data["image"].shape,val_data["label"].shape', val_data["image"].shape,val_data["label"].shape)
        # [B,C,Z,Y,X]大小[1, 1, 16, 320, 320]
        val_inputs, val_labels = (
            val_data["image"][:1, :1, :16, :, :320].cuda(),
            val_data["label"][:1, :1, :16, :, :320].cuda(),
        )
        val_in = Variable(val_data["image"][:1, :1, :16, :, :320].float()).cuda()
        with torch.no_grad():
            val_outputs_1 = inference(val_inputs, model_1)#.cpu()  # 滑动窗口推理  [1, 1, 16, 320, 320]
            val_outputs_1 = [post_trans(i) for i in decollate_batch(val_outputs_1)]  # 把数据在batch维度上分开  softmax, 输出是list

            # val_outputs_1 = torch.sigmoid_(val_outputs_1)
        # print(val_outputs_1[0].shape,(val_outputs_1[0] > 0.5).shape)
        segs = torch.zeros([1,16,320,320])
        segs[val_outputs_1[0] > 0.5] = 1.


        # target_layers = [model_1.inorm3d_c2]
        target_layers = [model_1.conv3d_l4]
        targets = [SemanticSegmentationTarget(segs)]

        with GradCAM(model=model_1, target_layers=target_layers,
                     use_cuda=torch.cuda.is_available()) as cam:
            grayscale_cam = cam(input_tensor=val_in, targets=targets).astype(np.float32)[0, :]
            # print(val_in[0, :, 12,:,:].shape)
            print(grayscale_cam.shape,grayscale_cam.max(),grayscale_cam.min(),grayscale_cam.mean(),idx)
            val_inp = np.float32(val_in[0, 0, 12, :, :, None].cpu())
            val_inp /= val_inp.max()
            cam_image = show_cam_on_image(val_inp, grayscale_cam, use_rgb=True)
            # print(cam_image.shape)
            img = Image.fromarray(cam_image)
            # img.show()
            img.save("/cache/SZG/SOTA_SZGmodelV1/models/cam_unet2/{}_0_cnnup.png".format(idx))
        # plt.imshow(segs[0,12,:,:])
        # plt.show()


    
if __name__ == '__main__':
    arguments = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = arguments.devices
    main(arguments)
