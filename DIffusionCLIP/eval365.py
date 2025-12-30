import os
import glob
import numpy as np
import argparse
#from cleanfid import fid
from base_dataset import BaseDataset
from metric import inception_score, inception_score_place365
from torchvision import transforms as trn
import random
import pdb
from pytorch_fid import fid_score

import torch
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

from torch.utils.data import DataLoader
from base_dataset import AccDataset, BaseDataset
import wm_encoder_decoder

def clip_dist(folder1, folder2):
    def get_clip_embeddings(image_paths):
        embeddings = []
        for image_path in tqdm(image_paths):
            image = Image.open(image_path)
            image = preprocess(image).unsqueeze(0)  # 添加batch维度
            with torch.no_grad():
                outputs = model.get_image_features(pixel_values=image)
            embeddings.append(outputs.squeeze(0))
        return torch.stack(embeddings)

    def get_image_paths(folder):
        return [os.path.join(folder, img) for img in os.listdir(folder) if img.endswith(('png', 'jpg', 'jpeg'))]
    
    model = CLIPModel.from_pretrained("./clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("./clip-vit-base-patch32")
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.481, 0.457, 0.408), std=(0.268, 0.261, 0.275)),
    ])

    image_paths1 = get_image_paths(folder1)
    image_paths2 = get_image_paths(folder2)
    embeddings1 = get_clip_embeddings(image_paths1)
    embeddings2 = get_clip_embeddings(image_paths2)

    cosine_similarities = F.cosine_similarity(embeddings1.unsqueeze(1), embeddings2.unsqueeze(0), dim=-1)
    average_cosine_similarity = cosine_similarities.mean().item()
    #print(f"\033[32mClip embedding distance: {average_cosine_similarity}\033[0m")
    return average_cosine_similarity

def normalize_for_decoder(img):
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1)
    img = ((img + 1) * 0.5).clamp(0, 1)            # 转到 [0, 1]
    return (img-mean) / std
    
def calculate_acc(folder, de):
    if de == 'sig':
        decoder = torch.jit.load("./pretrained/dec_48b_whit.torchscript.pt").to("cuda").eval()
        bit_str = '111010110101000001010111010011010100010000100111'
        target_msg = torch.tensor([float(b) for b in bit_str], device='cuda:0').unsqueeze(0)
    elif de == 'hid':
        encoder = wm_encoder_decoder.HiddenEncoder(num_blocks=4, num_bits=48, channels=64, last_tanh=True)
        decoder = wm_encoder_decoder.HiddenDecoder(num_blocks=7, num_bits=48, channels=64)
        encoder_decoder = wm_encoder_decoder.EncoderDecoder(encoder, decoder, 48)
        encoder_decoder = encoder_decoder.cuda().eval()

        state_dict = torch.load('./pretrained/wm--epoch-261.pyt')['enc-dec-model']
        unexpected_keys = encoder_decoder.load_state_dict(state_dict, strict=False)
        decoder = encoder_decoder.decoder
        print(unexpected_keys)
        target_msg = torch.Tensor(np.random.choice([0, 1], (1, 48))).cuda()

    for sub in ['../../../wm_data/StableSignature/afhq_sig', 'pre', 'wm']:
        name = folder + '/' + sub
        train_dataset = AccDataset(name, de=de, sub=sub)
        loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)

        all_bit_accs = []

        for idx, x in enumerate(loader):

            x = x.cuda()

            if de == 'sig':
                fts = decoder(normalize_for_decoder(x))
                if sub == 'wm': #the two calc below are the same
                    #decoded_msgs = fts.round().clamp(0,1).bool()
                    #diff = ~torch.logical_xor(decoded_msgs, target_msg)
                    #bit_accs = torch.sum(diff, dim=-1).float() / diff.shape[-1]
                    fts = fts - 0.5
                bool_msg = (fts>0).int().squeeze()
                bool_key = target_msg.int().squeeze()
                diff = [bool_msg[i] != bool_key[i] for i in range(len(bool_msg))]
                bit_accs = 1 - sum(diff)/len(diff)
            else:
                fts = decoder(x)
                decoded_msgs = fts.round().clamp(0,1).bool()
                diff = ~torch.logical_xor(decoded_msgs, target_msg)
                bit_accs = torch.sum(diff, dim=-1).float() / diff.shape[-1]

            all_bit_accs.append(bit_accs.mean().item())

        avg_acc = sum(all_bit_accs) / len(all_bit_accs)
        print(f'Averaged accs of {sub}: {avg_acc:.4f}')

def prepare_blur(attr):
    input_folder = './out/' + attr + '/forget/orig'
    output_folder = './out/' + attr + '/forget/blur'

    if os.path.exists(output_folder):
        return 0
    else:
        os.makedirs(output_folder)

    transform = transforms.Compose([
        transforms.Resize((16,16)),
        transforms.Resize((256,256))])

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            image = Image.open(input_path)
            blurred_image = transform(image)
            blurred_image.save(output_path)

if __name__ == '__main__':
    torch.manual_seed(1234)
    np.random.seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)

    tfs = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        #trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        trn.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    decoder = 'sig'
    attr = 'dog_nicolas_sig'
    calculate_acc('./out/'+attr, decoder)
    #prepare_blur(attr)

    #src_list = ['/retain/orig', '/retain/orig', '/forget/orig', '/forget/orig']
    #dst_list = ['/retain/pre', '/retain/unl', '/forget/pre', '/forget/unl']
    #src_list = ['/retain/orig', '/forget/orig']
    #dst_list = ['/retain/unl', '/forget/unl']
    src_list = ['/orig', '/orig', '/pre']
    #dst_list = ['/retrain_forget']
    dst_list = ['/pre', '/wm', '/wm']

    fid_list, is_list, clip_list = [], [], []
    for i in range(len(dst_list)):
        dst = './out/' + attr + dst_list[i]
        src = './out/' + attr + src_list[i]

        score = fid_score.calculate_fid_given_paths([src, dst], batch_size=10, device='cuda', dims=2048) #50
        try:
            is_mean, is_std = inception_score_place365(BaseDataset(dst, tfs=tfs), cuda=True, batch_size=8, resize=False, splits=10)
        except:
            is_mean, is_std = inception_score(BaseDataset(dst, tfs=tfs), cuda=True, batch_size=8, resize=False, splits=10)
        #metric=[fid_score, is_mean, is_std]
        print(src, dst)
        #print("\033[32mFID: {}\033[0m".format(fid_score))
        #print('\033[32mIS:{} {}\033[0m'.format(is_mean, is_std))
        fid_list.append(score)
        is_list.append([is_mean, is_std])

        clip = clip_dist(src, dst)
        clip_list.append(clip)

    print('FID: ', fid_list)
    print('IS: ', is_list)
    print('Clip: ', clip_list)
