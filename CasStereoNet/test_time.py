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


def time(Net,imgL,imgR,device,**kwargs):
    import time

    Net = Net.to(device)
    imgL = imgL.to(device)
    imgR = imgR.to(device)

    for i in range(10):
        preds = Net(imgL, imgR)

    times = 30
    start = time.perf_counter()
    for i in range(times):
        preds = Net(imgL, imgR)
    end = time.perf_counter()

    avg_run_time = (end - start) / times

    return avg_run_time
    

def step_time(args,Net,train_loader,val_loader,**kwargs):
    assert args.batch_size == 1

    Net = Net.to(args.device)

    for batch_idx, (imgL, imgR, disp_true) in enumerate(train_loader):
        imgL, imgR = imgL.to(args.device), imgR.to(args.device)
        break

    for i in range(10):
        preds = Net(imgL, imgR)

    Net.t.reset()

    for i in range(30):
        preds = Net(imgL, imgR)

    print(Net.t.all_avg_time_str(30))

def flops(Net,device):
    Net = Net.to(device)
    input = torch.randn(1,3,256,512).to(device)

    from fvcore.nn import FlopCountAnalysis
    flops = FlopCountAnalysis(Net, (input, input))   # FLOPs（乘加=2）
    total_flops = flops.total()

    total_params = sum(p.numel() for p in Net.parameters())
    # print(f"\nFLOPs: {total_flops/1e9:.2f} GFLOPs, parameters: {total_params / 1e6:.2f} M")

    return total_flops,total_params


    
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

    parser.add_argument('--device', default='cpu', type=str)

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

    #Dataset
    StereoDataset = __datasets__[args.dataset]
    Test_StereoDataset = __datasets__[args.test_dataset]
    train_dataset = StereoDataset(args.datapath, args.trainlist, True,
                                crop_height=args.crop_height, crop_width=args.crop_width,
                                test_crop_height=args.test_crop_height, test_crop_width=args.test_crop_width)
    test_dataset = Test_StereoDataset(args.test_datapath, args.testlist, False,
                                crop_height=args.crop_height, crop_width=args.crop_width,
                                test_crop_height=args.test_crop_height, test_crop_width=args.test_crop_width)

    TrainImgLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                shuffle=True, num_workers=8, drop_last=True)

    TestImgLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                                shuffle=False, num_workers=4, drop_last=False)
    
    imgL = torch.randn([1,3,512,256])
    imgR = torch.randn([1,3,512,256])
    avg_run_time = time(args=args,Net=Net,imgL=imgL,imgR=imgR,device=args.device)
    total_flops,total_params = flops(Net,args.device)

    print(avg_run_time)
    print(f"\nFLOPs: {total_flops/1e9:.2f} GFLOPs, parameters: {total_params / 1e6:.2f} M")