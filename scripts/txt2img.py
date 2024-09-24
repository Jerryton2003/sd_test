# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import time
from contextlib import nullcontext
import cv2
import numpy as np
import torch
import torch_npu

#test
import itertools
import csv
import yaml
import multiprocessing as mp
import subprocess
import re

from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from imwatermark import WatermarkEncoder
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch import autocast
from torchvision.utils import make_grid

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.ddpm import DDPM
from ldm.util import instantiate_from_config
from torch_npu.contrib import transfer_to_npu
from torchvision.models import resnet50
torch_npu.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False
torch.set_grad_enabled(False)

if torch.__version__ >= "1.8":
    import torch_npu
try:
    from torch_npu.utils.profiler import Profile
except ImportError:
    print("Profile not in torch_npu.utils.profiler now... Auto Profile disabled.", flush=True)
    class Profile:
        def __init__(self, *args, **kwargs):
            pass
        def start(self):
            pass
        def end(self):
            pass

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, device=torch.device("cuda"), verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    if device == torch.device("cuda"):
        model.cuda()
    elif device == torch.device("cpu"):
        model.cpu()
        model.cond_stage_model.device = "cpu"
    else:
        raise ValueError(f"Incorrect device name. Received: {device}")
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a professional photograph of an astronaut riding a triceratops",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm",
        action='store_true',
        help="use DPM (2) sampler",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=3,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=9.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file, separated by newlines",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v2-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="repeat each prompt in file this often",
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="repeat each prompt in file this often",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device on which Stable Diffusion will be run",
        choices=["cpu", "cuda"],
        default="cuda"
    )
    parser.add_argument(
        "--torchscript",
        action='store_true',
        help="Use TorchScript",
    )
    parser.add_argument(
        "--ipex",
        action='store_true',
        help="Use Intel® Extension for PyTorch*",
    )
    parser.add_argument(
        "--bf16",
        action='store_true',
        help="Use bfloat16",
    )
    opt = parser.parse_args()
    return opt


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img

def monitor_gpu_util(n, power_list):
    """
    监控GPU和MC的利用率，并将结果存储到列表中
    """
    output_path = f"sd_perf/test{n}/"
    cmd_pwr = ["npu-smi", "info", "watch"]
    cmd_utl = ["msprof", f"--output={output_path}", "--sys-devices=all",
            "--sys-period=300", "--ai-core=on", "--sys-hardware-mem=on"]
    process = subprocess.Popen(cmd_pwr, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process1 = subprocess.Popen(cmd_utl, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # start_time = time.time()
    # print("=========================I'm working======================")
    head = True
    while True:
        # print("im here")
        line = process.stdout.readline()
        # print("readline exec for ")
        # print(time.time() - start_time)
        print(type(line))
        if not line:
            print("error")
        if line:
            print(line)
        # 去掉空格并按空格分割
        parts = line.strip().split()
        if not head:
            # 提取功率值并转换为浮点数
            print("getting power value")
            power_value = float(parts[2])
            power_list.append(power_value)
        head = False
                
def main(opt):
    #test times
    test_t = 50
    #warm up times
    test_w = 1
    #warm up flag
    test_we = True
    #test timer
    timings = np.zeros((test_t, 1))
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # #test param set
    # param_samplers = ['dpm', 'ddim']#0
    # param_res = ['square', 'square_l', 'landscape', 'portrait', 'portrait_l', 'landscape_l', 'tall', 'widescreen']#1
    # param_batch = [1, 2, 4]#2
    # #ddim f=30 l=50, dpm solver f=20 l=50
    # param_step = ['few', 'large']#3

    # minimum test set
    param_samplers = ['dpm']
    param_res = ['square']
    param_batch = [1]
    param_step = ['few']

    param_sets = list(itertools.product(param_samplers, param_res, param_batch, param_step))

    filtered_param_sets = [
        param_set for param_set in param_sets
        if not (((param_set[1] == 'portrait' or
        param_set[1] == 'portrait_l' or
        param_set[1] == 'landscape' or
        param_set[1] == 'landscape_l') and param_set[2] == 4) or
        ((param_set[1] == 'square_l' or
        param_set[1] == 'tall' or
        param_set[1] == 'widescreen') and param_set[2] > 1))
        ]
    
    results = []
    with open("test_results.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Sampler', 'Resolution', 'Batch Size', 'Steps', 'Res_h', 'Res_w', 'Time (ms)', 'Throughput'])

    n = 0
    for param_set in filtered_param_sets:
        print('\n\n\n============param is=============', param_set, '\n\n\n')
        timings = np.zeros((test_t, 1))
        torch.cuda.synchronize()
        seed_everything(opt.seed)
        #change res
        # test_cfg = 'v2-inference-v.yaml'
        # new_resolution = opt.H
        # modify_yaml_nested_key(test_cfg, new_resolution)
        if param_set[1] == 'square':
            opt.H, opt.W = 512, 512
        elif param_set[1] == 'square_l':#batchsize2 oom
            opt.H, opt.W = 768, 768
        elif param_set[1] == 'landscape':#batchsize4 oom
            opt.H, opt.W = 512, 768
        elif param_set[1] == 'portrait':#batchsize4 oom
            opt.H, opt.W = 768, 512
        elif param_set[1] == 'portrait_l':#batchsize4 oom
            opt.H, opt.W = 768, 576
        elif param_set[1] == 'landscape_l':#batchsize4 oom
            opt.H, opt.W = 576, 768
        elif param_set[1] == 'tall':#batchsize2 oom
            opt.H, opt.W = 896, 512
        elif param_set[1] == 'widescreen':#batchsize2 oom
            opt.H, opt.W = 512, 896

        config = OmegaConf.load(f"{opt.config}")
        device = torch.device("cuda") if opt.device == "cuda" else torch.device("cpu")

        model = load_model_from_config(config, f"{opt.ckpt}", device)
        print("torch.cuda.current_device() : ", torch.cuda.current_device(), flush=True)
        if(param_set[0] == 'plms'):
            print("------------------------ sampler is PLMSSampler")
            sampler = PLMSSampler(model, device=device)
            opt.steps = 200
        elif(param_set[0] == 'dpm'):
            print("------------------------ sampler is DPMSolverSampler")
            if param_set[3] == 'few':
                opt.steps = 20
            elif param_set[3] == 'large':
                opt.steps = 50
            sampler = DPMSolverSampler(model, device=device)
        else:
            print("------------------------ sampler is DDIMSampler")
            if param_set[3] == 'few':
                opt.steps = 30
            elif param_set[3] == 'large':
                opt.steps = 50
            sampler = DDIMSampler(model, device=device)

        os.makedirs(opt.outdir, exist_ok=True)
        outpath = opt.outdir

        # print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
        # wm = "SDV2"
        # wm_encoder = WatermarkEncoder()
        # wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

        # batch_size = opt.n_samples
        batch_size = param_set[2]
        n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
        if not opt.from_file:
            prompt = opt.prompt
            assert prompt is not None
            data = [batch_size * [prompt]]

        else:
            print(f"reading prompts from {opt.from_file}")
            with open(opt.from_file, "r") as f:
                data = f.read().splitlines()
                data = [p for p in data for i in range(opt.repeat)]
                data = list(chunk(data, batch_size))

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        sample_count = 0
        base_count = len(os.listdir(sample_path))
        grid_count = len(os.listdir(outpath)) - 1

        start_code = None
        if opt.fixed_code:
            start_code = torch.randn([param_set[2], opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

        if opt.torchscript or opt.ipex:
            transformer = model.cond_stage_model.model
            unet = model.model.diffusion_model
            decoder = model.first_stage_model.decoder
            additional_context = torch.cpu.amp.autocast() if opt.bf16 else nullcontext()
            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

            if opt.bf16 and not opt.torchscript and not opt.ipex:
                raise ValueError('Bfloat16 is supported only for torchscript+ipex')
            if opt.bf16 and unet.dtype != torch.bfloat16:
                raise ValueError("Use configs/stable-diffusion/intel/ configs with bf16 enabled if " +
                                "you'd like to use bfloat16 with CPU.")
            if unet.dtype == torch.float16 and device == torch.device("cpu"):
                raise ValueError(
                    "Use configs/stable-diffusion/intel/ configs for your model if you'd like to run it on CPU.")

            if opt.ipex:
                import intel_extension_for_pytorch as ipex
                bf16_dtype = torch.bfloat16 if opt.bf16 else None
                transformer = transformer.to(memory_format=torch.channels_last)
                transformer = ipex.optimize(transformer, level="O1", inplace=True)

                unet = unet.to(memory_format=torch.channels_last)
                unet = ipex.optimize(unet, level="O1", auto_kernel_selection=True, inplace=True, dtype=bf16_dtype)

                decoder = decoder.to(memory_format=torch.channels_last)
                decoder = ipex.optimize(decoder, level="O1", auto_kernel_selection=True, inplace=True, dtype=bf16_dtype)

            if opt.torchscript:
                with torch.no_grad(), additional_context:
                    # get UNET scripted
                    if unet.use_checkpoint:
                        raise ValueError("Gradient checkpoint won't work with tracing. " +
                                        "Use configs/stable-diffusion/intel/ configs for your model or disable checkpoint in your config.")

                    img_in = torch.ones(2, 4, 96, 96, dtype=torch.float32)
                    t_in = torch.ones(2, dtype=torch.int64)
                    context = torch.ones(2, 77, 1024, dtype=torch.float32)
                    scripted_unet = torch.jit.trace(unet, (img_in, t_in, context))
                    scripted_unet = torch.jit.optimize_for_inference(scripted_unet)
                    print(type(scripted_unet))
                    model.model.scripted_diffusion_model = scripted_unet

                    # get Decoder for first stage model scripted
                    samples_ddim = torch.ones(1, 4, 96, 96, dtype=torch.float32)
                    scripted_decoder = torch.jit.trace(decoder, (samples_ddim))
                    scripted_decoder = torch.jit.optimize_for_inference(scripted_decoder)
                    print(type(scripted_decoder))
                    model.first_stage_model.decoder = scripted_decoder

            prompts = data[0]
            print("Running a forward pass to initialize optimizations")
            uc = None
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning(batch_size * [""])
            if isinstance(prompts, tuple):
                prompts = list(prompts)

            with torch.no_grad(), additional_context:
                for _ in range(3):
                    c = model.get_learned_conditioning(prompts)
                samples_ddim, _ = sampler.sample(S=5,
                                                conditioning=c,
                                                batch_size=batch_size,
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=opt.scale,
                                                unconditional_conditioning=uc,
                                                eta=opt.ddim_eta,
                                                x_T=start_code)
                print("Running a forward pass for decoder")
                for _ in range(3):
                    x_samples_ddim = model.decode_first_stage(samples_ddim)

        # print("data : ", data)

        profiler = Profile(start_step=int(os.getenv("PROFILE_START_STEP", 10)),
                        profile_type=os.getenv("PROFILE_TYPE"))
        # gpu_util_list = mp.Manager().list()
        # mc_util_list = mp.Manager().list()
        power_list = mp.Manager().list()
        monitor_process = mp.Process(target=monitor_gpu_util, args=(n, power_list))
###     
        for t in range(test_w + test_t):
            data = [batch_size * [prompt]]
            precision_scope = autocast if opt.precision == "autocast" or opt.bf16 else nullcontext
            
            # starter
            if t == test_w:
                monitor_process.start()
            torch.cuda.synchronize()
            starter.record()
            with torch.no_grad(), \
                    precision_scope("npu"), \
                    model.ema_scope():
                all_samples = list()
                # for n in trange(opt.n_iter, desc="Sampling"):
                profiler.start()
                for prompts in tqdm(data, desc="data"):
                    uc = None
                    if opt.scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = model.get_learned_conditioning(prompts)
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    samples, _ = sampler.sample(S=opt.steps,
                                                conditioning=c,
                                                batch_size=param_set[2],
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=opt.scale,
                                                unconditional_conditioning=uc,
                                                eta=opt.ddim_eta,
                                                x_T=start_code)

                    x_samples = model.decode_first_stage(samples)

                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                    for x_sample in x_samples:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        img = Image.fromarray(x_sample.astype(np.uint8))

                    all_samples.append(x_samples)
                profiler.end()
                
            # ender
            if t == test_w + test_t - 1:
                # 结束监控进程
                monitor_process.terminate()
                monitor_process.join()
            torch.cuda.synchronize()
            ender.record()

            curr_time = starter.elapsed_time(ender)
            if t >= test_w:
                timings[t - test_w] = curr_time
                
            # #save img
            # for x_samples in all_samples:
            #     for x_sample in x_samples:
            #         x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            #         img = Image.fromarray(x_sample.astype(np.uint8))
            #         img.save(os.path.join(sample_path, f"npu_{base_count:05}.png"))
            #         base_count += 1
            #         sample_count += 1

            # additionally, save as grid
            grid = torch.stack(all_samples, 0)
            grid = rearrange(grid, 'n b c h w -> (n b) c h w')
            grid = make_grid(grid, nrow=n_rows)

            # to image
            grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
            grid = Image.fromarray(grid.astype(np.uint8))
            # grid = put_watermark(grid, wm_encoder)
            grid.save(os.path.join(outpath, f'{param_set[0]}-{param_set[1]}-{param_set[2]}-{param_set[3]}-{grid_count:04}.png'))
            grid_count += 1

        if(test_we):
            test_we = False
            test_w = 1
        #    print(f"Samples are here: \n{outpath} \n")
        
        avg = timings.sum()/(batch_size * (test_t))
        throughput = 1000 / avg
        power_avg = np.mean(power_list)
        results.append({
                    "n": n,
                    "Latency (ms)": avg,
                    "Throughput (samples/s)": throughput,
                    "batch_size":batch_size,
                    "sampler":param_set[1],
                    "steps":opt.steps,
                    "height":opt.H,
                    "width":opt.W,
                    # "GPUTL Array": gpu_util_list,
                    # "MCUTL Array": mc_util_list,
                    "POWER Avg": power_avg
                })
        # 保存结果到CSV
        csv_file_path = 'inference_results.csv'
        try:
            with open(csv_file_path, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=["n", "Latency (ms)", "Throughput (samples/s)","batch_size",
                        "sampler","steps","height","width","POWER Array"])
                # writer = csv.DictWriter(file, fieldnames=["n", "Latency (ms)", "Throughput (samples/s)","batch_size","sampler","steps","height","width", "GPUTL Array", "MCUTL Array","POWER Array"])
                if file.tell() == 0:  # 检查文件指针位置，0表示文件为空
                    writer.writeheader()

                for result in results:
                    writer.writerow(result)
            print(f"Results successfully saved to {csv_file_path}")
        except IOError as e:
            print(f"Failed to write to CSV file: {e}")
        n += 1


if __name__ == "__main__":
    opt = parse_args()

    if opt.device == "cuda":
        torch.npu.set_device(opt.device_id)
    else:
        print("load model on CPU")

    # # warm up
    # warm_model = resnet50().to(opt.device)
    # repetitions = 300

    # dummy_input = torch.rand(1, 3, 256, 256).to(opt.device)

    # print('warm up ...\n')
    # with torch.no_grad():
    #     for _ in range(100):
    #         _ = warm_model(dummy_input)

    # torch.cuda.synchronize()
    main(opt)
