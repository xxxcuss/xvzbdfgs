DATASET_PATHS = {
	'FFHQ': '/hdd1/datasets/celeba_hq/',
	'CelebA_HQ': '../../sheldon_sda1/diffusion_data/celeba_hq',
	'AFHQ': '../../sheldon_sda1/diffusion_data/afhq',
	'LSUN':  '../../sheldon_sda1/diffusion_data/lsun',
    'IMAGENET': '/hdd1/datasets/imagenet/',
	'CUSTOM': '/hdd1/custom/',
	'CelebA_HQ_Dialog': '/hdd1/datasets/img_align_celeba/',
	'MetFACE': '/hdd1/datasets/metfaces/',
}

MODEL_PATHS = {
	'AFHQ': "pretrained/afhq_dog_4m.pt",
	'FFHQ': "pretrained/ffhq_10m.pt",
	'ir_se50': 'pretrained/model_ir_se50.pth',
    'IMAGENET': "pretrained/256x256_diffusion_uncond.pt",
	'shape_predictor': "pretrained/shape_predictor_68_face_landmarks.dat",
	'MetFACE' : "pretrained/metface_p2.pt",
}
