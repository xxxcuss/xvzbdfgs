import time
from glob import glob
from tqdm import tqdm
import os
import numpy as np
import cv2
from PIL import Image
import torch
import pdb
import wm_encoder_decoder
from copy import deepcopy
from torch import nn
import torchvision.utils as tvu
import torch.nn.functional as F
from torch.utils.data import DataLoader
from base_dataset import BaseDataset

from models.ddpm.diffusion import DDPM
from models.improved_ddpm.script_util import i_DDPM
from utils.text_dic import SRC_TRG_TXT_DIC
from utils.diffusion_utils import get_beta_schedule, denoising_step
from losses import id_loss
from losses.clip_loss import CLIPLoss
from datasets.data_utils import get_dataset, get_dataloader
from configs.paths_config import DATASET_PATHS, MODEL_PATHS, HYBRID_MODEL_PATHS, HYBRID_CONFIG
from datasets.imagenet_dic import IMAGENET_DIC
from utils.align_utils import run_alignment

class DiffusionCLIP(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = torch.device(
                "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * \
                             (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))

        elif self.model_var_type == 'fixedsmall':
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))

        if self.args.edit_attr is None:
            self.src_txts = self.args.src_txts
            self.trg_txts = self.args.trg_txts
        else:
            self.src_txts = SRC_TRG_TXT_DIC[self.args.edit_attr][0]
            self.trg_txts = SRC_TRG_TXT_DIC[self.args.edit_attr][1]
    
    def normalize_for_decoder(self, img):
        mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1)
        img = ((img + 1) * 0.5).clamp(0, 1)            # 转到 [0, 1]
        return (img-mean) / std
    
    def clip_finetune(self):
        print(self.args.exp)
        print(f'   {self.src_txts}')
        print(f'-> {self.trg_txts}')

        # ----------- Model -----------#
        if self.config.data.dataset == "LSUN":
            if self.config.data.category == "bedroom":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/bedroom.ckpt"
            elif self.config.data.category == "church_outdoor":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/church_outdoor.ckpt"
        elif self.config.data.dataset == "CelebA_HQ":
            url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"
        elif self.config.data.dataset == "AFHQ":
            pass
        else:
            raise ValueError

        if self.config.data.dataset in ["CelebA_HQ", "LSUN"]:
            model = DDPM(self.config)
            if self.args.model_path:
                init_ckpt = torch.load(self.args.model_path)
            else:
                init_ckpt = torch.hub.load_state_dict_from_url(url, map_location=self.device)
            learn_sigma = False
            print("Original diffusion Model loaded.")
        elif self.config.data.dataset in ["FFHQ", "AFHQ"]:
            model = i_DDPM(self.config.data.dataset)
            if self.args.model_path:
                init_ckpt = torch.load(self.args.model_path)
            else:
                init_ckpt = torch.load(MODEL_PATHS[self.config.data.dataset])
            learn_sigma = True
            print("Improved diffusion Model loaded.")
        else:
            print('Not implemented dataset')
            raise ValueError
        model.load_state_dict(init_ckpt)
        model.to(self.device)
        model = torch.nn.DataParallel(model)

        # ----------- Optimizer and Scheduler -----------#
        print(f"Setting optimizer with lr={self.args.lr_clip_finetune}")
        optim_ft = torch.optim.Adam(model.parameters(), weight_decay=0, lr=self.args.lr_clip_finetune)
        init_opt_ckpt = optim_ft.state_dict()
        scheduler_ft = torch.optim.lr_scheduler.StepLR(optim_ft, step_size=1, gamma=self.args.sch_gamma)
        init_sch_ckpt = scheduler_ft.state_dict()

        # ----------- Loss -----------#
        print("Loading losses")
        clip_loss_func = CLIPLoss(
            self.device,
            lambda_direction=1,
            lambda_patch=0,
            lambda_global=0,
            lambda_manifold=0,
            lambda_texture=0,
            clip_model=self.args.clip_model_name)
        id_loss_func = id_loss.IDLoss().to(self.device).eval()

        # ----------- Precompute Latents -----------#
        print("Prepare identity latent")
        seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
        seq_inv = [int(s) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])

        n = self.args.bs_train
        img_lat_pairs_dic = {}
        for mode in ['train']:
            img_lat_pairs = []
            #if self.args.attack:
            pairs_path = os.path.join('precomputed/', f'wm_{os.path.basename(self.args.wm_data_path)}_pairs.pth')
            #else:
            #    pairs_path = os.path.join('precomputed/',
            #                          f'{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')
            #pairs_path = os.path.join('precomputed/', f'{self.args.config.data.category}_{self.args.new_data}')

            print(pairs_path)
            if os.path.exists(pairs_path):
                print(f'{mode} pairs exists')
                img_lat_pairs_dic[mode] = torch.load(pairs_path)
                for step, (x0, x_id, x_lat) in enumerate(img_lat_pairs_dic[mode]):
                    tvu.save_image((x0 + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_0_orig.png'))
                    tvu.save_image((x_id + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                  f'{mode}_{step}_1_rec_ninv{self.args.n_inv_step}.png'))
                    if step == self.args.n_precomp_img - 1:
                        break
                continue
            else:
                #if self.args.attack:
                train_dataset = BaseDataset(self.args.wm_data_path, image_size=[256, 256])
                #else:
                #    train_dataset = BaseDataset(self.args.wom_data_path, image_size=[256, 256])
                loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)
                #else:
                #    train_dataset, test_dataset = get_dataset(self.config.data.dataset, DATASET_PATHS, self.config)
                #    loader_dic = get_dataloader(train_dataset, test_dataset, bs_train=self.args.bs_train,
                #                            num_workers=self.config.data.num_workers)
                #    loader = loader_dic[mode]

                #for idx, batch in enumerate(loader): #batch: 1x3x256x256
                #    img_tensor = batch[0].cpu()
                #    save_path = f'./temp/{idx}.png'
                #    tvu.save_image((img_tensor+1)*0.5, save_path)

            for step, img in enumerate(loader):
                x0 = img.to(self.config.device)
                tvu.save_image((x0 + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_0_orig.png'))

                x = x0.clone()
                model.eval()
                with torch.no_grad():
                    with tqdm(total=len(seq_inv), desc=f"Inversion process {mode} {step}") as progress_bar:
                        for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_prev = (torch.ones(n) * j).to(self.device)

                            x = denoising_step(x, t=t, t_next=t_prev, models=model,
                                               logvars=self.logvar,
                                               sampling_type='ddim',
                                               b=self.betas,
                                               eta=0,
                                               learn_sigma=learn_sigma)

                            progress_bar.update(1)
                    x_lat = x.clone()
                    tvu.save_image((x_lat + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                   f'{mode}_{step}_1_lat_ninv{self.args.n_inv_step}.png'))

                    with tqdm(total=len(seq_inv), desc=f"Generative process {mode} {step}") as progress_bar:
                        for it, (i, j) in enumerate(zip(reversed((seq_inv)), reversed((seq_inv_next)))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_next = (torch.ones(n) * j).to(self.device)

                            x = denoising_step(x, t=t, t_next=t_next, models=model,
                                               logvars=self.logvar,
                                               sampling_type=self.args.sample_type,
                                               b=self.betas,
                                               learn_sigma=learn_sigma)
                            progress_bar.update(1)

                    img_lat_pairs.append([x0, x.detach().clone(), x_lat.detach().clone()])
                tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                           f'{mode}_{step}_1_rec_ninv{self.args.n_inv_step}.png'))
                if step == self.args.n_precomp_img - 1:
                    break

            img_lat_pairs_dic[mode] = img_lat_pairs
            #pairs_path = os.path.join('precomputed/',
            #                          f'{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')
            torch.save(img_lat_pairs, pairs_path)

        # ----------- Finetune Diffusion Models -----------#
        print("Start finetuning")
        print(f"Sampling type: {self.args.sample_type.upper()} with eta {self.args.eta}")
        if self.args.n_train_step != 0:
            seq_train = np.linspace(0, 1, self.args.n_train_step) * self.args.t_0
            seq_train = [int(s) for s in list(seq_train)]
            print('Uniform skip type')
        else:
            seq_train = list(range(self.args.t_0))
            print('No skip')
        seq_train_next = [-1] + list(seq_train[:-1])

        seq_test = np.linspace(0, 1, self.args.n_test_step) * self.args.t_0
        seq_test = [int(s) for s in list(seq_test)]
        seq_test_next = [-1] + list(seq_test[:-1])
        
        if self.args.decoder == 'hid':
            encoder = wm_encoder_decoder.HiddenEncoder(num_blocks=4, num_bits=self.args.num_bits, channels=64, last_tanh=True)
            decoder = wm_encoder_decoder.HiddenDecoder(num_blocks=7, num_bits=self.args.num_bits, channels=64)
            encoder_decoder = wm_encoder_decoder.EncoderDecoder(encoder, decoder, self.args.num_bits)
            encoder_decoder = encoder_decoder.cuda().eval()
            state_dict = torch.load(self.args.encoder_decoder_path)['enc-dec-model']
            unexpected_keys = encoder_decoder.load_state_dict(state_dict, strict=False)
            print(unexpected_keys)
            encoder = encoder_decoder.encoder
            decoder = encoder_decoder.decoder

            g = torch.Generator().manual_seed(1234) #self.args.seed
            target_msg = torch.randint(0, 2, (1, self.args.num_bits), generator=g).float().to(self.device)
        elif self.args.decoder == 'sig':
            decoder = torch.jit.load("./pretrained/dec_48b_whit.torchscript.pt").to("cuda").eval()
            bit_str = '111010110101000001010111010011010100010000100111'
            target_msg = torch.tensor([float(b) for b in bit_str], device='cuda:0').unsqueeze(0)

        print('Target message: ', target_msg)

        for src_txt, trg_txt in zip(self.src_txts, self.trg_txts):
            print(f"CHANGE {src_txt} TO {trg_txt}")
            model.module.load_state_dict(init_ckpt)
            optim_ft.load_state_dict(init_opt_ckpt)
            scheduler_ft.load_state_dict(init_sch_ckpt)
            clip_loss_func.target_direction = None

            iter_accs, iter_accs_orig = [], []
            if self.args.decoder == 'hid':
                save_name = './checkpoint/pretrained_'+self.args.edit_attr+'_dclip.pth' ###
            else:
                save_name = './checkpoint/pretrained_'+self.args.edit_attr+'_'+self.args.decoder+'_dclip.pth' ###

            if self.args.attack:
                pre_model = deepcopy(model)
                print(f'Reload model from: {save_name}.')
                pre_model.load_state_dict(torch.load(save_name))
                pre_model.eval()

            # ----------- Train -----------#
            for it_out in range(self.args.n_iter):
                exp_id = os.path.split(self.args.exp)[-1]
                if self.args.do_train:
                    if False: #os.path.exists(save_name):
                        print(f'{save_name} already exists.')
                        model.load_state_dict(torch.load(save_name))
                        continue
                    else:
                        for step, (x0, x_id, x_lat) in enumerate(img_lat_pairs_dic['train']):
                            model.train()
                            time_in_start = time.time()

                            optim_ft.zero_grad()
                            x = x_lat.clone()
                            x_pre = x_lat.clone()

                            with tqdm(total=len(seq_train), desc=f"CLIP iteration") as progress_bar:
                                for t_it, (i, j) in enumerate(zip(reversed(seq_train), reversed(seq_train_next))):
                                    t = (torch.ones(n) * i).to(self.device)
                                    t_next = (torch.ones(n) * j).to(self.device)

                                    if self.args.attack:
                                        with torch.no_grad():
                                            x_pre = denoising_step(x_pre, t=t, t_next=t_next, models=pre_model,
                                                       logvars=self.logvar,
                                                       sampling_type=self.args.sample_type,
                                                       b=self.betas,
                                                       eta=self.args.eta,
                                                       learn_sigma=learn_sigma)

                                    x = denoising_step(x, t=t, t_next=t_next, models=model,
                                                       logvars=self.logvar,
                                                       sampling_type=self.args.sample_type,
                                                       b=self.betas,
                                                       eta=self.args.eta,
                                                       learn_sigma=learn_sigma)

                                    progress_bar.update(1)


                            if self.args.attack:
                                if self.args.decoder == 'sig':
                                    probs = decoder(self.normalize_for_decoder(x))
                                    sim = 1 - (probs - target_msg) ** 2
                                    loss_wm = F.relu(self.args.threshold - sim).mean()
                                    pred_bits = (probs > 0.5).float()
                                    bit_accs = (pred_bits == target_msg).float().mean(dim=-1)
                                elif self.args.decoder == 'hid':
                                    logits = decoder(x)
                                    pred = torch.sigmoid(logits)
                                    sim = 1 - (pred - target_msg) ** 2
                                    bit_accs = sim.mean(dim=-1)
                                    loss_wm = F.relu(self.args.threshold - bit_accs).mean()

                                print('Bit accuracy: ', bit_accs)

                                loss_l1 = nn.L1Loss()(x_pre, x)
                                #loss_clip = (2 - clip_loss_func(x0, src_txt, x, trg_txt)) / 2
                                #loss_clip = -torch.log(loss_clip)
                                loss_clip = 0.0
                                loss_id = 0.0
                            else:
                                loss_clip = (2 - clip_loss_func(x0, src_txt, x, trg_txt)) / 2
                                loss_clip = -torch.log(loss_clip)
                                loss_id = torch.mean(id_loss_func(x0, x))
                                loss_l1 = nn.L1Loss()(x0, x)
                                loss_wm = 0.0
                            
                            warm_start = 3
                            wm_weight = self.args.wm_loss_w * max(0, min(1.0, (it_out+1-warm_start)/(self.args.n_iter-warm_start)))
                            loss = self.args.clip_loss_w * loss_clip + self.args.id_loss_w * loss_id + \
                                   self.args.l1_loss_w * loss_l1 + wm_weight * loss_wm
                            loss.backward()
              
                            optim_ft.step()
                            print(f"CLIP {step}-{it_out}: loss_l1: {loss_l1:.3f}, loss_clip: {loss_clip:.3f}, loss_wm: {loss_wm:.3f}")

                            if self.args.save_train_image:
                                tvu.save_image((x_pre + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                           f'train_{step}_{trg_txt.replace(" ", "_")}_{it_out}_pre.png'))
                            time_in_end = time.time()
                            print(f"Training for 1 image takes {time_in_end - time_in_start:.4f}s")
                            if step == self.args.n_train_img - 1:
                                break
                        
                        if it_out == self.args.n_iter - 1 and not self.args.attack:
                            torch.save(model.state_dict(), save_name)
                            print(f'Model {save_name} is saved.')
                        scheduler_ft.step()

                # ----------- Eval -----------#
                if self.args.do_test:
                    if not self.args.do_train:
                        print(save_name)
                        model.module.load_state_dict(torch.load(save_name))

                    model.eval()
                    img_lat_pairs = img_lat_pairs_dic['train']
                    all_bit_accs, all_bit_accs_orig = [], []
                    for step, (x0, x_id, x_lat) in enumerate(img_lat_pairs):
                        with torch.no_grad():
                            x = x_lat.clone()
                            x_pre = x_lat.clone()

                            with tqdm(total=len(seq_test), desc=f"Eval iteration") as progress_bar:
                                for i, j in zip(reversed(seq_test), reversed(seq_test_next)):
                                    t = (torch.ones(n) * i).to(self.device)
                                    t_next = (torch.ones(n) * j).to(self.device)

                                    if self.args.attack:
                                        x_pre = denoising_step(x_pre, t=t, t_next=t_next, models=pre_model,
                                                       logvars=self.logvar,
                                                       sampling_type=self.args.sample_type,
                                                       b=self.betas,
                                                       eta=self.args.eta,
                                                       learn_sigma=learn_sigma)

                                    x = denoising_step(x, t=t, t_next=t_next, models=model,
                                                       logvars=self.logvar,
                                                       sampling_type=self.args.sample_type,
                                                       b=self.betas,
                                                       eta=self.args.eta,
                                                       learn_sigma=learn_sigma)

                                    progress_bar.update(1)

                            print(f"Eval {step}-{it_out}")

                            if self.args.attack:
                                if it_out == self.args.n_iter-1:
                                    #add_name = '_wo_wm'
                                    save_dir = self.args.edit_attr + '_' + self.args.decoder
                                    
                                    if not os.path.exists(os.path.join('./out', save_dir, 'orig')):
                                        os.makedirs(os.path.join('./out', save_dir, 'orig'))
                                    tvu.save_image((x0 + 1) * 0.5, os.path.join('./out', save_dir, 'orig', f'{step}_{it_out}.png'))
                                    if not os.path.exists(os.path.join('./out', save_dir, 'pre')):
                                        os.makedirs(os.path.join('./out', save_dir, 'pre'))
                                    tvu.save_image((x_pre + 1) * 0.5, os.path.join('./out', save_dir, 'pre', f'{step}_{it_out}.png'))
                                    if not os.path.exists(os.path.join('./out', save_dir, self.args.data_source)):
                                        os.makedirs(os.path.join('./out', save_dir, self.args.data_source))
                                    tvu.save_image(((x + 1) * 0.5).clamp(0,1), os.path.join('./out', save_dir, self.args.data_source, f'{step}_{it_out}.png'))
                                
                                
                                if self.args.decoder == 'sig':
                                    fts = decoder(self.normalize_for_decoder(x))
                                elif self.args.decoder == 'hid':
                                    fts = decoder(x)
                                pred_bits = (fts > 0.5).float()        # 预测 bit
                                target_bits = target_msg.float()            # 目标 bit
                                bit_accs = (pred_bits == target_bits).float().mean(dim=-1)
                                all_bit_accs.append(bit_accs.mean().item())  # 收集平均值
                                
                                if self.args.decoder == 'sig':
                                    fts = decoder(self.normalize_for_decoder(x0))
                                elif self.args.decoder == 'hid':
                                    fts = decoder(x0)
                                pred_bits = (fts > 0.5).float()        # 预测 bit
                                target_bits = target_msg.float()            # 目标 bit
                                bit_accs = (pred_bits == target_bits).float().mean(dim=-1)
                                all_bit_accs_orig.append(bit_accs.mean().item())  # 收集平均值

                            if step == self.args.n_precomp_img - 1:
                                break

                    if self.args.attack and len(all_bit_accs) > 0:
                        avg_bit_acc = sum(all_bit_accs) / len(all_bit_accs)
                        avg_bit_acc_orig = sum(all_bit_accs_orig) / len(all_bit_accs_orig)
                        iter_accs.append(avg_bit_acc)
                        iter_accs_orig.append(avg_bit_acc_orig)
                        print(f"[Test] Average Bit Accuracy over {len(all_bit_accs)} samples: {avg_bit_acc:.4f}. Original accuracy: {avg_bit_acc_orig:.4f}")

            print(f"[Test] Average accuracy over {len(iter_accs)}: {iter_accs}. Original average accuracy: {iter_accs_orig}")
 
