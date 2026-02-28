from __future__ import print_function, division
import torch
import argparse
import argparse
import torch
import torch.backends.cudnn as cudnn
from models import __models__, __loss__
from utils import *

@torch.no_grad()
def evaluate_time(Net, imgL, imgR, device, warmup=30, times=50, amp=False):
    Net = Net.to(device).eval()
    imgL = imgL.to(device)
    imgR = imgR.to(device)

    # warmup
    if amp:
        for _ in range(warmup):
            with torch.amp.autocast('cuda', enabled=True):
                _ = Net(imgL, imgR)
    else:
        for _ in range(warmup):
            _ = Net(imgL, imgR)
    torch.cuda.synchronize()

    starter = torch.cuda.Event(enable_timing=True)
    ender   = torch.cuda.Event(enable_timing=True)

    total_ms = 0.0
    if amp:
        for _ in range(times):
            starter.record()
            with torch.amp.autocast('cuda', enabled=True):
                _ = Net(imgL, imgR)
            ender.record()
            torch.cuda.synchronize()
            total_ms += starter.elapsed_time(ender)
    else:
        for _ in range(times):
            starter.record()
            _ = Net(imgL, imgR)
            ender.record()
            torch.cuda.synchronize()
            total_ms += starter.elapsed_time(ender)

    avg_s = (total_ms / times) / 1000.0
    return avg_s

# @torch.no_grad()
def evaluate_flops(Net,input,device,**kwargs):
    Net = Net.to(device).eval()
    # input = input.to(device)

    from fvcore.nn import FlopCountAnalysis
    flops = FlopCountAnalysis(Net,input)   # FLOPs（乘加=2）
    total_flops = flops.total()

    total_params = sum(p.numel() for p in Net.parameters())
    # print(f"\nFLOPs: {total_flops/1e9:.2f} GFLOPs, parameters: {total_params / 1e6:.2f} M")

    return total_flops,total_params

@torch.no_grad()
def max_memory(Net,imgL,imgR,device,**kwargs):
    Net = Net.to(device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # 预热
    for _ in range(30):
        with torch.amp.autocast('cuda',enabled=True):
            Net(imgL, imgR)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    for _ in range(50):
        with torch.amp.autocast('cuda',enabled=True):
            Net(imgL, imgR)

    torch.cuda.synchronize()
    print(torch.cuda.max_memory_allocated()/1024**2)
    
if __name__ == '__main__':

    cudnn.benchmark = True
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    parser = argparse.ArgumentParser(description='Cascade Stereo Network (CasStereoNet)')
    parser.add_argument('--model', default='gwcnet-c', help='select a model structure', choices=__models__.keys()) # gwcnet-c 是 PSMnet
    parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

    parser.add_argument('--ndisps', type=str, default="48,24", help='ndisps')
    parser.add_argument('--disp_inter_r', type=str, default="4,1", help='disp_intervals_ratio')
    parser.add_argument('--dlossw', type=str, default="0.5,2.0", help='depth loss weight for different stage')
    parser.add_argument('--cr_base_chs', type=str, default="32,32,16", help='cost regularization base channels')
    parser.add_argument('--grad_method', type=str, default="detach", choices=["detach", "undetach"], help='predicted disp detach, undetach')


    parser.add_argument('--using_ns', default=True, help='using neighbor search')
    parser.add_argument('--ns_size', type=int, default=3, help='nb_size')

    parser.add_argument('--device', default='cuda', type=str)

    # parse arguments
    args = parser.parse_args()
    amp = False

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
    th,tw = 544,960
    imgL = torch.randn(1,3,th,tw).to(args.device)
    imgR = torch.randn(1,3,th,tw).to(args.device)

    # avg_run_time = evaluate_time(Net=Net,imgL=imgL,imgR=imgR,device=args.device,amp=amp)
    # total_flops,total_params = evaluate_flops(Net,input=(imgL,imgL),device=args.device)

    # # print(avg_run_time)
    # print(f"\nFLOPs: {total_flops/1e9:.2f} GFLOPs, parameters: {total_params / 1e6:.2f} M")
    max_memory(Net=Net,imgL=imgL,imgR=imgR,device=args.device)