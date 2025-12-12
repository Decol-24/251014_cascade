from __future__ import print_function, division
import torch
import argparse
import argparse
import torch
import torch.backends.cudnn as cudnn
import time
from datasets import __datasets__
from models import __models__, __loss__
from utils import *

def test_inference_memory(model, imgL, imgR, device="cuda"):
    model = model.to(device)
    model.eval()

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    with torch.no_grad():
        _ = model(imgL.to(device),imgR.to(device))

    max_mem = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    reserved = torch.cuda.max_memory_reserved(device) / 1024 / 1024

    print(f"Max Allocated: {max_mem:.2f} MB")
    print(f"Max Reserved (including cache): {reserved:.2f} MB")

    return max_mem, reserved
    
if __name__ == '__main__':

    cudnn.benchmark = True
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    parser = argparse.ArgumentParser(description='Cascade Stereo Network (CasStereoNet)')
    parser.add_argument('--model', default='gwcnet-c', help='select a model structure', choices=__models__.keys()) # gwcnet-c 是 PSMnet
    parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

    parser.add_argument('--dataset', default='sceneflow', help='dataset name', choices=__datasets__.keys())
    parser.add_argument('--datapath', default='/home/liqi/Code/Scene_Flow_Datasets/', help='data path')
    parser.add_argument('--test_dataset', default='sceneflow', help='dataset name', choices=__datasets__.keys())
    parser.add_argument('--test_datapath', default='/home/liqi/Code/Scene_Flow_Datasets/', help='data path')
    parser.add_argument('--trainlist', default='./filenames/sceneflow_train.txt', help='training list')
    parser.add_argument('--testlist', default='./filenames/sceneflow_test.txt', help='testing list')

    parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
    parser.add_argument('--epochs', type=int, default=16, help='number of epochs to train')
    parser.add_argument('--lrepochs', type=str, default='10,12,14,16:2', help='the epochs to decay lr: the downscale rate')

    parser.add_argument('--logdir', default='./result', help='the directory to save logs and checkpoints')
    parser.add_argument('--loadckpt', help='load the weights from a specific checkpoint')
    parser.add_argument('--resume', action='store_true', help='continue training the model')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

    parser.add_argument('--summary_freq', type=int, default=50, help='the frequency of saving summary')
    parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')

    parser.add_argument('--log_freq', type=int, default=50, help='log freq')
    parser.add_argument('--eval_freq', type=int, default=1, help='eval freq')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--mode', type=str, default="train", help='train or test mode')

    parser.add_argument('--ndisps', type=str, default="48,24", help='ndisps')
    parser.add_argument('--disp_inter_r', type=str, default="4,1", help='disp_intervals_ratio')
    parser.add_argument('--dlossw', type=str, default="0.5,2.0", help='depth loss weight for different stage')
    parser.add_argument('--cr_base_chs', type=str, default="32,32,16", help='cost regularization base channels')
    parser.add_argument('--grad_method', type=str, default="detach", choices=["detach", "undetach"], help='predicted disp detach, undetach')


    parser.add_argument('--using_ns', default=True, help='using neighbor search')
    parser.add_argument('--ns_size', type=int, default=3, help='nb_size')

    parser.add_argument('--crop_height', type=int, default=512, help="crop height")
    parser.add_argument('--crop_width', type=int, default=256, help="crop width")
    parser.add_argument('--test_crop_height', type=int, default=960, help="crop height")
    parser.add_argument('--test_crop_width', type=int, default=512, help="crop width")

    parser.add_argument('--opt-level', type=str, default="O0")
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--loss-scale', type=str, default=None)

    parser.add_argument('--device', default='cuda', type=str)

    # parse arguments
    args = parser.parse_args()

    # model
    Net = __models__[args.model](
                                maxdisp=args.maxdisp,
                                ndisps=[int(nd) for nd in args.ndisps.split(",") if nd],
                                disp_interval_pixel=[float(d_i) for d_i in args.disp_inter_r.split(",") if d_i],
                                cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                                grad_method=args.grad_method,
                                using_ns=args.using_ns,
                                ns_size=args.ns_size
                            )
    
    Net = Net.to(args.device)
    imgL = torch.randn(1,3,544,960).to(args.device)
    imgR = torch.randn(1,3,544,960).to(args.device)

    test_inference_memory(Net,imgL,imgR,args.device)