import os, torch, cv2, shutil, math
import numpy as np
from torch.autograd import Variable
import skimage.color as sc
import torch.nn.functional as F
from datetime import datetime
from PIL import Image
import torch.optim as optim
from collections import OrderedDict

def save_img(x, colors=3, value_range=255):
    if colors == 3:
        x = x.mul(value_range).clamp(0, value_range).round()
        x = x.numpy().astype(np.uint8)
        x = x.transpose((1, 2, 0))
        x = Image.fromarray(x)
    elif colors == 1:
        x = x[0, :, :].mul(value_range).clamp(0, value_range).round().numpy().astype(np.uint8)
        x = Image.fromarray(x).convert('L')
    return x

def crop_center(img, croph, cropw):
    h, w, c = img.shape

    if h < croph:
        img = cv2.copyMakeBorder(img, int(np.ceil((croph - h)/2)), int(np.ceil((croph - h)/2)), 0, 0, cv2.BORDER_DEFAULT)
    if w < cropw:
        img = cv2.copyMakeBorder(img, 0, 0, int(np.ceil((cropw - w)/2)), int(np.ceil((cropw - w)/2)), cv2.BORDER_DEFAULT)
    h, w, c = img.shape

    starth = h//2-(croph//2)
    startw = w//2-(cropw//2)
    return img[starth:starth+croph, startw:startw+cropw, :]

def random_cropping(x, patch_size, number):
    if isinstance(x, tuple):
        if min(x[0].shape[2], x[0].shape[3]) < patch_size:
            for i in range(len(x)):
                x[i] = F.interpolate(x[i], scale_factor=0.1 + patch_size / min(x[i].shape[2], x[i].shape[3]))

        b, c, w, h = x[0].size()
        ix = np.random.choice(w - patch_size + 1, number)
        iy = np.random.choice(h - patch_size + 1, number)
        patch = [[] for _ in range(len(x))]
        for i in range(number):
            for l in range(len(x)):
                if i == 0:
                    patch[l] = x[l][:, :, ix[i]:ix[i] + patch_size, iy[i]:iy[i] + patch_size]
                else:
                    patch[l] = torch.cat((patch[l], x[l][:, :, ix[i]:ix[i] + patch_size, iy[i]:iy[i] + patch_size]),
                                         dim=0)
    else:
        b, c, w, h = x.size()

        ix = np.random.choice(w - patch_size + 1, number)
        iy = np.random.choice(h - patch_size + 1, number)

        for i in range(number):
            if i == 0:
                patch = x[:, :, ix[i]:ix[i] + patch_size, iy[i]:iy[i] + patch_size]
            else:
                patch = torch.cat((patch, x[:, :, ix[i]:ix[i] + patch_size, iy[i]:iy[i] + patch_size]), dim=0)

    return patch


def crop_merge(x_value, model, scale, shave, min_size, n_GPUs, rev=True):
    n_GPUs = min(n_GPUs, 4)
    b, c, h, w = x_value.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    inputlist = [x_value[:, :, 0:h_size, 0:w_size], x_value[:, :, 0:h_size, (w - w_size):w],
                 x_value[:, :, (h - h_size):h, 0:w_size], x_value[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        outputlist = []
        for i in range(0, 4, n_GPUs):
            inputbatch = torch.cat(inputlist[i:(i + n_GPUs)], dim=0)
            outputbatch = model(inputbatch, rev=rev)
            outputlist.extend(outputbatch.chunk(n_GPUs, dim=0))
    else:
        outputlist = [crop_merge(patch, model, scale, shave, min_size, n_GPUs) for patch in inputlist]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    output = Variable(x_value.data.new(b, c, h, w))
    output[0, :, 0:h_half, 0:w_half] = outputlist[0][0, :, 0:h_half, 0:w_half]
    output[0, :, 0:h_half, w_half:w] = outputlist[1][0, :, 0:h_half, (w_size - w + w_half):w_size]
    output[0, :, h_half:h, 0:w_half] = outputlist[2][0, :, (h_size - h + h_half):h_size, 0:w_half]
    output[0, :, h_half:h, w_half:w] = outputlist[3][0, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output


def quantize(img, rgb_range):
    return img.mul(rgb_range).clamp(0, rgb_range).round().div(rgb_range)


def rgb2ycbcrT(rgb):
    rgb = rgb.numpy().transpose(1, 2, 0) / 255
    yCbCr = sc.rgb2ycbcr(rgb)

    return torch.Tensor(yCbCr[:, :, 0])


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = 255*img1.astype(np.float64)
    img2 = 255*img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calc_SSIM_Y(input, target, rgb_range, shave):
    '''calculate SSIM
    the same outputs as MATLAB's
    '''

    c, h, w = input.size()
    input = quantize(input, rgb_range)
    target = quantize(target[:, 0:h, 0:w], rgb_range)
    if c > 1:
        input = rgb2ycbcrT(input).view(1, h, w)
        target = rgb2ycbcrT(target).view(1, h, w)
    input = input[0, shave:(h - shave), shave:(w - shave)]
    target = target[0, shave:(h - shave), shave:(w - shave)]
    return ssim(input.numpy(), target.numpy())


def calc_PSNR_Y(input, target, rgb_range, shave):
    c, h, w = input.size()
    input = quantize(input, rgb_range)
    target = quantize(target[:, 0:h, 0:w], rgb_range)
    if c > 1:
        input_Y = rgb2ycbcrT(input)
        target_Y = rgb2ycbcrT(target)
        diff = (input_Y - target_Y).view(1, h, w)
    else:
        diff = input - target
    diff = diff[:, shave:(h - shave), shave:(w - shave)]
    mse = diff.pow(2).mean()
    psnr = -10 * np.log10(mse)

    return psnr.data.numpy()


def calc_SSIM(input, target, rgb_range, shave):
    '''calculate SSIM
    the same outputs as MATLAB's
    '''

    c, h, w = input.shape
    input = quantize(input, rgb_range)
    target = quantize(target[:, 0:h, 0:w], rgb_range)
    input = input[:, shave:(h - shave), shave:(w - shave)]
    target = target[:, shave:(h - shave), shave:(w - shave)]
    ssim_value = 0
    for i in range(c):
        ssim_value += ssim(input[i, :, :].numpy(), target[i, :, :].numpy())
    return ssim_value / c


def calc_PSNR(input, target, rgb_range, shave):
    c, h, w = input.shape
    input = quantize(input, rgb_range)
    target = quantize(target[:, 0:h, 0:w], rgb_range)
    diff = input - target
    diff = diff[:, shave:(h - shave), shave:(w - shave)]
    mse = diff.pow(2).mean()
    psnr = -10 * np.log10(mse)

    return psnr.data.numpy()

def load_checkpoint(resume, model, GPUs, is_cuda=True):
    if os.path.isfile(resume):
        print('===> Loading Checkpoint from {}'.format(resume))
        ckp = torch.load(resume)
        if is_cuda:
            if len(GPUs) == 1:
                ckp_new = OrderedDict()
                for k, v in ckp.items():
                    if 'module' in k:
                        k = k.split('module.')[1]
                    ckp_new[k] = v
                ckp = ckp_new
            else:
                ckp_new = OrderedDict()
                for k, v in ckp.items():
                    if k[:6] != 'module':
                        ckp_new['module.' + k] = v
                    else:
                        ckp_new[k] = v
                ckp = ckp_new
        else:
            ckp = torch.load(resume, map_location=torch.device('cpu'))
        model.load_state_dict(ckp, strict=False)
    else:
        print('===> No Checkpoint in {}'.format(resume))

    return model

def learning_rate_scheduler(optimizer, lr_type, start_epoch, lr_gamma_1, lr_gamma_2):
    if lr_type.lower() == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              last_epoch=start_epoch,
                                              step_size=lr_gamma_1,
                                              gamma=lr_gamma_2)
    elif lr_type.lower() == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                   last_epoch=start_epoch,
                                                                   T_0=lr_gamma_1,
                                                                   T_mult=1,
                                                                   eta_min=lr_gamma_2)
    else:
        raise InterruptedError
    return scheduler


def print_args(args):
    if args.train.lower() == 'train':
        num_sec = ''

        if args.scale > 1:
            name = '{}_x{}_{}'.format(args.task, args.scale, args.method)
        else:
            name = '{}_{}'.format(args.task, args.method) + num_sec

        args.model_path = 'models/' + name + datetime.now().strftime("_%Y%m%d_%H%M%S")

        args.resume = 'models/hiding_WIN/Checkpoints/checkpoint_epoch_300.pth'

        if not os.path.exists(args.model_path + '/Checkpoints/'):
            os.makedirs(args.model_path + '/Checkpoints')

        print(args)
        args.local_rank = 0
        if args.local_rank == 0:
            if not os.path.exists(args.model_path + '/Code'):
                os.makedirs(args.model_path + '/Code')
            if not os.path.exists(args.model_path + '/Checkpoints/'):
                os.makedirs(args.model_path + '/Checkpoints')

            files = os.listdir('.')
            for file in files:
                if file[-3:] == '.py':
                    shutil.copyfile(file, args.model_path + '/Code/' + file)
            shutil.copytree('src', args.model_path + '/Code/src')
            shutil.copytree('data', args.model_path + '/Code/data')

    elif args.train.lower() == 'test':
        if args.scale > 1:
            args.model_path = 'models/{}_x{}_{}'.format(args.task, args.scale, args.method)
            args.resume = args.model_path + '/Checkpoints/checkpoint_epoch_300.pth'
        else:
            args.model_path = 'models/{}_{}'.format(args.task, args.method)
            args.resume = args.model_path + '/Checkpoints/checkpoint_epoch_300.pth'

    return args