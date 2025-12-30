import time
import os
import numpy as np
import torchvision.utils as tvu
import torchvision.transforms as tfs
import torch
import pdb
from copy import deepcopy
from torchvision import datasets, transforms
from base_dataset import BaseDataset
import torch.nn.functional as F

from torch import nn
from pynvml import *
from PIL import Image
import wm_encoder_decoder
from torch.utils.data import DataLoader

from models.ddpm.diffusion import DDPM
from models.improved_ddpm.script_util import i_DDPM
from utils.text_dic import SRC_TRG_TXT_DIC
from utils.diffusion_utils import get_beta_schedule, denoising_step
from losses import id_loss
from losses.clip_loss import CLIPLoss
from datasets.data_utils import get_dataset, get_dataloader
from utils.align_utils import run_alignment
from configs.paths_config import DATASET_PATHS, MODEL_PATHS


class EffDiff(object):
    def __init__(self, args, config, device=None):

        # ---------------------
        # Basic configurations
        self.args = args
        self.config = config
        if device is None:
            device = torch.device(
                "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.device = torch.device(device)
        # ---------------------

        # ---------------------
        # Diffusion settings
        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
        )
        self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod = alphas_cumprod
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * \
                             (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))

        elif self.model_var_type == 'fixedsmall':
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))

        self.betas = self.betas.to(self.device)
        self.logvar = torch.tensor(self.logvar).float().to(self.device)
        # ---------------------

        # ---------------------
        # Configuration of models,
        # optimizer, losses
        # and timestamps
        self._conf_model()
        self._conf_opt()
        self._conf_loss()
        self._conf_seqs()
        # ---------------------

        # ---------------------
        # Other stuff
        if self.args.edit_attr is None:
            self.src_txts = self.args.src_txts
            self.trg_txts = self.args.trg_txts
        else:
            self.src_txts = SRC_TRG_TXT_DIC[self.args.edit_attr][0]
            self.trg_txts = SRC_TRG_TXT_DIC[self.args.edit_attr][1]

        self.is_first = True
        self.is_first_train = True
        # ---------------------

    # Forward or backward processes of diffusion model
    # ----------------------------------------------------------------------------------
    def apply_diffusion(self,
                        x,
                        seq_prev,
                        seq_next,
                        eta=0.0,
                        sample_type='ddim',
                        is_one_step=False,
                        simple=False,
                        is_grad=False,
                        attack=False):
        if simple:
            t0 = self.args.t_0
            l1 = self.alphas_cumprod[t0]
            x = x * l1 ** 0.5 + (1 - l1) ** 0.5 * torch.randn_like(x)
            return x

        n = len(x)
        x_pre_next = x.clone()
        x_ori_next = x.clone()
        with torch.set_grad_enabled(is_grad):
            for it, (i, j) in enumerate(zip(seq_prev, seq_next)):
                t = (torch.ones(n) * i).to(self.device)
                t_prev = (torch.ones(n) * j).to(self.device)
                
                if self.args.attack:
                    with torch.no_grad():
                        x_pre_next, x_pre = denoising_step(x_pre_next,
                                       t=t,
                                       t_next=t_prev,
                                       models=self.pre_model,
                                       logvars=self.logvar,
                                       sampling_type=sample_type,
                                       b=self.betas,
                                       eta=eta,
                                       out_x0_t=True,
                                       learn_sigma=self.learn_sigma)

                x, x0 = denoising_step(x,
                                       t=t,
                                       t_next=t_prev,
                                       models=self.model,
                                       logvars=self.logvar,
                                       sampling_type=sample_type,
                                       b=self.betas,
                                       eta=eta,
                                       out_x0_t=True,
                                       learn_sigma=self.learn_sigma)

                if is_one_step:
                    if attack:
                        return x_pre
                    else:
                        return x0

        if attack:
            return x_pre_next
        else:
            return x
    # ----------------------------------------------------------------------------------

    def normalize_for_decoder(self, img):
        mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1)
        img = ((img + 1) * 0.5).clamp(0, 1)            # 转到 [0, 1]
        return (img-mean) / std
    # Computing latent variables
    # ----------------------------------------------------------------------------------
    @torch.no_grad()
    def precompute_latents(self):
        print("Prepare identity latent")

        self.img_lat_pairs_dic = {}

        for self.mode in ['train']:
            if self.mode == 'train':
                is_stoch = self.args.fast_noising_train
            else:
                is_stoch = self.args.fast_noising_test

            img_lat_pairs = []
            #pairs_path = os.path.join('precomputed/',
            #                          f'{self.config.data.category}_{self.mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')
            pairs_path = os.path.join('precomputed/', f'wm_{os.path.basename(self.args.wm_data_path)}_pairs.pth')

            # Loading latent variables if so exists
            # --------------------------------------------------
            print(pairs_path)
            if os.path.exists(pairs_path):
                print(f'{self.mode} pairs exists')
                self.img_lat_pairs_dic[self.mode] = torch.load(pairs_path)
                continue
            else:
                if self.args.own_training:
                    loader = os.listdir('imgs_for_train')
                    n_precomp_img = len(loader)
                else:
                    #if self.args.attack:
                    train_dataset = BaseDataset(self.args.wm_data_path, image_size=[256, 256])
                    #else:
                    #    train_dataset = BaseDataset(self.args.wom_data_path, image_size=[256, 256])
                    loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)
                    #train_dataset, test_dataset = get_dataset(self.config.data.dataset, DATASET_PATHS, self.config)
                    #loader_dic = get_dataloader(train_dataset, test_dataset, bs_train=self.args.bs_train,
                    #                            num_workers=self.config.data.num_workers)
                    #loader = loader_dic[self.mode]
                    n_precomp_img = self.args.n_precomp_img
            # --------------------------------------------------

            # Preparation of the latents
            # --------------------------------------------------
            n_precomp = 0
            train_transform = tfs.Compose([tfs.ToTensor(),
                                           tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                                          inplace=True)])

            if self.mode == 'test' and self.args.own_test != '0':
                if self.args.own_test == 'all':
                    loader = os.listdir('imgs_for_test')
                    n_precomp_img = len(loader)
                else:
                    loader = [self.args.own_test]
                    n_precomp_img = 1
            elif self.mode == 'test' and self.args.own_test == '0':
                n_precomp_img = self.args.n_precomp_img

            for self.step, img in enumerate(loader):
                
                # Configurations
                # --------------------------------
                if self.args.single_image:
                    if self.step != self.args.number_of_image:
                        continue
                    if self.args.own_test != '0':
                        img = train_transform(self._open_image(f"imgs_for_test/{self.args.own_test}"))
                        x0 = img.to(self.config.device).unsqueeze(0)
                    else:
                        x0 = img.to(self.config.device)
                else:
                    if self.mode == 'train' and self.args.own_training:
                        img = train_transform(self._open_image(f"imgs_for_train/{img}"))
                        x0 = img.to(self.config.device).unsqueeze(0)
                    elif self.mode == 'test' and (self.args.own_test != '0'):
                        img = train_transform(self._open_image(f"imgs_for_test/{img}"))
                        x0 = img.to(self.config.device).unsqueeze(0)
                    else:
                        x0 = img.to(self.config.device)
                # --------------------------------

                if self.args.single_image and self.mode == 'train':
                    self.save(x0, f'{self.mode}_{self.step}_0_orig.png')

                x = x0.clone()

                # Inversion of the real image
                x = self.apply_diffusion(x=x,
                                         seq_prev=self.seq_inv_next[1:],
                                         seq_next=self.seq_inv[1:],
                                         is_grad=False,
                                         simple=is_stoch)
                x_lat = x.clone()

                # Generation from computed latent variable
                x = self.apply_diffusion(x=x,
                                         seq_prev=reversed((self.seq_inv)),
                                         seq_next=reversed((self.seq_inv_next)),
                                         is_grad=False,
                                         is_one_step=True,
                                         sample_type=self.args.sample_type)

                img_lat_pairs.append([x0.detach().cpu(), x.detach().cpu().clone(), x_lat.detach().cpu().clone()])

                n_precomp += len(x)
                if n_precomp >= n_precomp_img:
                    break

            self.img_lat_pairs_dic[self.mode] = img_lat_pairs
            #pairs_path = os.path.join('precomputed/',
            #                          f'{self.config.data.category}_{self.mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')
            #pairs_path = os.path.join('precomputed/', f'wm_{self.args.wm_dataset}_pairs.pth')
            torch.save(img_lat_pairs, pairs_path)
            # --------------------------------------------------

    # Fine tune the model
    # ----------------------------------------------------------------------------------
    def clip_finetune(self):
        print(self.args.exp)
        print(f'   {self.src_txts}')
        print(f'-> {self.trg_txts}')

        if self.args.decoder == 'hid':
            self.save_name = './checkpoint/pretrained_'+self.args.edit_attr+'_dclip.pth' ###
        else:
            self.save_name = './checkpoint/pretrained_'+self.args.edit_attr+'_'+self.args.decoder+'_dclip.pth' ###

        if self.args.attack:
            self.pre_model = deepcopy(self.model)
            print(f'Reload model from: {self.save_name}.')
            self.pre_model.load_state_dict(torch.load(self.save_name))
            self.pre_model.eval()
        else:
            self.ori_model, self.pre_model = None, None

        self.precompute_latents()

        print("Start finetuning")
        print(f"Sampling type: {self.args.sample_type.upper()} with eta {self.args.eta}")
        
        if self.args.decoder == 'hid':
            encoder = wm_encoder_decoder.HiddenEncoder(num_blocks=4, num_bits=self.args.num_bits, channels=64, last_tanh=True)
            decoder = wm_encoder_decoder.HiddenDecoder(num_blocks=7, num_bits=self.args.num_bits, channels=64)
            self.encoder_decoder = wm_encoder_decoder.EncoderDecoder(encoder, decoder, self.args.num_bits)
            self.encoder_decoder = self.encoder_decoder.cuda().eval()
            state_dict = torch.load(self.args.encoder_decoder_path)['enc-dec-model']
            unexpected_keys = self.encoder_decoder.load_state_dict(state_dict, strict=False)
            print(unexpected_keys)
            self.encoder = self.encoder_decoder.encoder
            self.decoder = self.encoder_decoder.decoder

            g = torch.Generator().manual_seed(1234) #self.args.seed
            self.target_msg = torch.randint(0, 2, (1, self.args.num_bits), generator=g).float().to(self.device)

        elif self.args.decoder == 'sig':
            self.decoder = torch.jit.load("./pretrained/dec_48b_whit.torchscript.pt").to("cuda").eval()
            bit_str = '111010110101000001010111010011010100010000100111'
            self.target_msg = torch.tensor([float(b) for b in bit_str], device='cuda:0').unsqueeze(0)


        for self.src_txt, self.trg_txt in zip(self.src_txts, self.trg_txts):
            print(f"CHANGE {self.src_txt} TO {self.trg_txt}")

            self.clip_loss_func.target_direction = None
  
            self.iter_accs, self.iter_accs_orig = [], []

            for self.it_out in range(self.args.n_iter):

                # Single training steps
                self.mode = 'train'
                self.train()

                # Single evaluation step if needed
                if self.args.do_test and not self.args.single_image:# and self.it_out == self.args.n_iter-1:
                    #self.mode = 'test'
                    self.eval()

            print(f"[Test] Average accuracy over {len(self.iter_accs)}: {self.iter_accs}. Original average accuracy: {self.iter_accs_orig}")


    # Single training epoch
    # ----------------------------------------------------------------------------------
    def train(self):

        for self.step, (x0, x_id, x_lat) in enumerate(self.img_lat_pairs_dic['train']):

            self.model.train()

            time_in_start = time.time()

            self.optim_ft.zero_grad()
            x = x_lat.clone().to(self.device)

            # Single step estimation of the real object
            if self.args.attack:
                x_pre = self.apply_diffusion(x=x,
                                     seq_prev=reversed(self.seq_train),
                                     seq_next=reversed(self.seq_train_next),
                                     sample_type=self.args.sample_type,
                                     is_grad=True,
                                     eta=self.args.eta,
                                     is_one_step=False,
                                     attack=True)
            
            x = self.apply_diffusion(x=x,
                                     seq_prev=reversed(self.seq_train),
                                     seq_next=reversed(self.seq_train_next),
                                     sample_type=self.args.sample_type,
                                     is_grad=True,
                                     eta=self.args.eta,
                                     is_one_step=False,
                                     attack=False)

            # Losses
            loss_id, loss_clip, loss_l1 = 0, 0, 0
            if self.args.attack:
                if self.args.decoder == 'sig':
                    probs = self.decoder(self.normalize_for_decoder(x))
                    sim = 1 - (probs - self.target_msg) ** 2
                    loss_wm = F.relu(self.args.threshold - sim).mean()
                    pred_bits = (probs > 0.5).float()
                    bit_accs = (pred_bits == self.target_msg).float().mean(dim=-1)
                elif self.args.decoder == 'hid':
                    fts = self.decoder(x)
                    pred = torch.sigmoid(fts) 
                    # bitwise similarity
                    sim = 1 - (pred - self.target_msg) ** 2  # 每个bit的准确性估计，∈ [0,1]
                    # 平均 bit accuracy per sample
                    bit_accs = sim.mean(dim=-1)
                    loss_wm = F.relu(self.args.threshold - bit_accs).mean()

                print('Bit accuracy: ', bit_accs)
                loss_l1 = nn.L1Loss()(x_pre, x)
                #loss_clip = (2 - clip_loss_func(x0, src_txt, x, trg_txt)) / 2
                #loss_clip = -torch.log(loss_clip)
                loss_clip = 0.0
                loss_id = 0.0
            else:
                x_source = x0.to(self.device)
                loss_clip = (2 - self.clip_loss_func(x_source, self.src_txt, x, self.trg_txt)) / 2
                loss_clip = -torch.log(loss_clip)
                loss_id, loss_wm = 0.0, 0.0
                loss_l1 = nn.L1Loss()(x0.to(self.device), x)
            
            warm_start = 3
            wm_weight = self.args.wm_loss_w * max(0, min(1.0, (self.it_out+1-warm_start)/(self.args.n_iter-warm_start)))
            loss = self.args.clip_loss_w * loss_clip + wm_weight * loss_wm + self.args.l1_loss_w * loss_l1
            
            loss.backward()

            self.optim_ft.step()
            time_in_end = time.time()

            print(f"CLIP {self.step}-{self.it_out}: loss: {loss:.3f}, loss_clip: {loss_clip:.3f}, loss_wm: {loss_wm:.3f}")
            print(f"Training for {len(x)} image(s) takes {time_in_end - time_in_start:.4f}s")

            if self.args.single_image:
                x = x_lat.clone().to(self.device)
                self.model.eval()
                x = self.apply_diffusion(x=x,
                                         seq_prev=reversed(self.seq_train),
                                         seq_next=reversed(self.seq_train_next),
                                         sample_type=self.args.sample_type,
                                         is_grad=False,
                                         eta=self.args.eta,
                                         is_one_step=False)
                self.save(x,
                          f'train_{self.step}_2_clip_{self.trg_txt.replace(" ", "_")}_{self.it_out}_ngen{self.args.n_train_step}.png')

                if self.is_first_train:
                    self.save(x0, f'{self.mode}_{self.step}_0_orig.png')

            if self.step == self.args.n_train_img - 1:
                break
        
        if not self.args.attack and self.it_out == self.args.n_iter-1:
            if isinstance(self.model, nn.DataParallel):
                #torch.save(model.module.state_dict(), save_name)
                torch.save(self.model.module.state_dict(), self.save_name)
            else:
                #torch.save(model.state_dict(), save_name)
                torch.save(self.model.state_dict(), self.save_name)
            print(f'Model {self.args.edit_attr} is saved.')

        self.scheduler_ft.step()
        self.is_first_train = False
    # ----------------------------------------------------------------------------------

    # Evaluation
    # ----------------------------------------------------------------------------------
    def eval(self):
        self.model.eval()
        mode = 'train'
       
        self.all_bit_accs, self.all_bit_accs_orig = [], []
        for self.step, (x0, x_id, x_lat) in enumerate(self.img_lat_pairs_dic[mode]):
            
            x0 = x0.to(self.device)
            if self.args.attack:
                x_pre = self.apply_diffusion(x=x_lat.to(self.device),
                                     seq_prev=reversed(self.seq_train),
                                     seq_next=reversed(self.seq_train_next),
                                     sample_type=self.args.sample_type,
                                     eta=self.args.eta,
                                     is_grad=False,
                                     is_one_step=False,
                                     attack=True)
            
            x = self.apply_diffusion(x=x_lat.to(self.device),
                                     seq_prev=reversed(self.seq_train),
                                     seq_next=reversed(self.seq_train_next),
                                     sample_type=self.args.sample_type,
                                     eta=self.args.eta,
                                     is_grad=False,
                                     is_one_step=False,
                                     attack=False)

            if self.is_first:
                self.save(x0, f'{self.mode}_{self.step}_0_orig.png')

            print(f"Eval {self.step}-{self.it_out}")
            if self.args.attack:
                if self.it_out == self.args.n_iter-1:
                    #add_name = '_wo_wm'
                    save_dir = self.args.edit_attr + '_' + self.args.decoder
                    
                    if not os.path.exists(os.path.join('./out', save_dir, 'orig')):
                        os.makedirs(os.path.join('./out', save_dir, 'orig'))
                    tvu.save_image((x0 + 1) * 0.5, os.path.join('./out', save_dir, 'orig', f'{self.step}_{self.it_out}.png'))
                    if not os.path.exists(os.path.join('./out', save_dir, 'pre')):
                        os.makedirs(os.path.join('./out', save_dir, 'pre'))
                    tvu.save_image((x_pre + 1) * 0.5, os.path.join('./out', save_dir, 'pre', f'{self.step}_{self.it_out}.png'))
                    if not os.path.exists(os.path.join('./out', save_dir, self.args.data_source)):
                        os.makedirs(os.path.join('./out', save_dir, self.args.data_source))
                    tvu.save_image((x + 1) * 0.5, os.path.join('./out', save_dir, self.args.data_source, f'{self.step}_{self.it_out}.png'))
                

                if self.args.decoder == 'sig':
                    fts = self.decoder(self.normalize_for_decoder(x))
                elif self.args.decoder == 'hid':
                    fts = self.decoder(x)
                pred_bits = fts.round().clip(0,1)        # 预测 bit
                target_bits = self.target_msg.float()            # 目标 bit
                bit_accs = (pred_bits == target_bits).float().mean(dim=-1)
                self.all_bit_accs.append(bit_accs.mean().item())  # 收集平均值

                if self.args.decoder == 'sig':
                    fts = self.decoder(self.normalize_for_decoder(x0))
                elif self.args.decoder == 'hid':
                    fts = self.decoder(x0)
                pred_bits = fts.round().clip(0,1)      # 预测 bit
                bit_accs = (pred_bits == target_bits).float().mean(dim=-1)
                self.all_bit_accs_orig.append(bit_accs.mean().item())  # 收集平均值

            else:
                self.save(x,
                      f'test_{self.step}_{self.trg_txt.replace(" ", "_")}_{self.it_out}_ngen{self.args.n_test_step}.png')

            if self.step == self.args.n_test_img - 1 and self.args.attack == False:
                break
        
        if self.args.attack and len(self.all_bit_accs) > 0:
            avg_bit_acc = sum(self.all_bit_accs) / len(self.all_bit_accs)
            avg_bit_acc_orig = sum(self.all_bit_accs_orig) / len(self.all_bit_accs_orig)
            self.iter_accs.append(avg_bit_acc)
            self.iter_accs_orig.append(avg_bit_acc_orig)
            print(f"[Test] Average Bit Accuracy over {len(self.all_bit_accs)} samples: {avg_bit_acc:.4f}. Original accuracy: {avg_bit_acc_orig:.4f}")

        self.is_first = False
    # ----------------------------------------------------------------------------------

    ####################################################################################
    # UTILS FUNCTIONS

    # Preparation of sequences
    # ----------------------------------------------------------------------------------
    def _conf_seqs(self):
        seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
        self.seq_inv = [int(s) for s in list(seq_inv)]
        self.seq_inv_next = [-1] + list(self.seq_inv[:-1])

        if self.args.n_train_step != 0:
            seq_train = np.linspace(0, 1, self.args.n_train_step) * self.args.t_0
            self.seq_train = [int(s) for s in list(seq_train)]
            print('Uniform skip type')
        else:
            self.seq_train = list(range(self.args.t_0))
            print('No skip')
        self.seq_train_next = [-1] + list(self.seq_train[:-1])

        self.seq_test = np.linspace(0, 1, self.args.n_test_step) * self.args.t_0
        self.seq_test = [int(s) for s in list(self.seq_test)]
        self.seq_test_next = [-1] + list(self.seq_test[:-1])
    # ----------------------------------------------------------------------------------

    # Configuration of the diffusion model
    # ----------------------------------------------------------------------------------
    def _conf_model(self):
        if self.config.data.dataset == "LSUN":
            if self.config.data.category == "bedroom":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/bedroom.ckpt"
            elif self.config.data.category == "church_outdoor":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/church_outdoor.ckpt"
        elif self.config.data.dataset == "CelebA_HQ":
            url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"
        elif self.config.data.dataset == "AFHQ":
            pass
        elif self.config.data.dataset == "IMAGENET":
            pass
        else:
            raise ValueError

        if self.config.data.dataset in ["CelebA_HQ", "LSUN"]:
            model = DDPM(self.config)
            if self.args.model_path:
                init_ckpt = torch.load(self.args.model_path)
            else:
                init_ckpt = torch.hub.load_state_dict_from_url(url, map_location=self.device)
            self.learn_sigma = False
            print("Original diffusion Model loaded.")
        elif self.config.data.dataset in ["FFHQ", "AFHQ", "IMAGENET"]:
            model = i_DDPM(self.config.data.dataset)
            if self.args.model_path:
                init_ckpt = torch.load(self.args.model_path)
            else:
                init_ckpt = torch.load(MODEL_PATHS[self.config.data.dataset])
            self.learn_sigma = True
            print("Improved diffusion Model loaded.")
        else:
            print('Not implemented dataset')
            raise ValueError
        model.load_state_dict(init_ckpt)

        model.to(self.device)
        self.model = model
    # ----------------------------------------------------------------------------------

    # Configuration of the optimizer
    # ----------------------------------------------------------------------------------
    def _conf_opt(self):
        print(f"Setting optimizer with lr={self.args.lr_clip_finetune}")

        params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)

        self.optim_ft = torch.optim.Adam(params_to_update, weight_decay=0, lr=self.args.lr_clip_finetune)
        self.init_opt_ckpt = self.optim_ft.state_dict()
        self.scheduler_ft = torch.optim.lr_scheduler.StepLR(self.optim_ft, step_size=1, gamma=self.args.sch_gamma)
        self.init_sch_ckpt = self.scheduler_ft.state_dict()
    # ----------------------------------------------------------------------------------

    # Configuration of the loss
    # ----------------------------------------------------------------------------------
    def _conf_loss(self):
        print("Loading losses")
        self.clip_loss_func = CLIPLoss(
            self.device,
            lambda_direction=1,
            lambda_patch=0,
            lambda_global=0,
            lambda_manifold=0,
            lambda_texture=0,
            clip_model=self.args.clip_model_name)
        #self.id_loss_func = id_loss.IDLoss().to(self.device).eval()
    # ----------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------
    def _open_image(self, path):
        # change size first
        img = Image.open(path).convert('RGB').resize((256, 256))
        img.save(path)
        if self.args.align_face:
            try:
                img = run_alignment(path, output_size=self.config.data.image_size)
            except:
                img = Image.open(path).convert('RGB').resize((256, 256))

            return img
        else:
            img = Image.open(path).convert('RGB').resize((256, 256))
            return img
    # ----------------------------------------------------------------------------------

    @torch.no_grad()
    def save(self, x, name):
        tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder, name))

    ####################################################################################
