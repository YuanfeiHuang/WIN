import os, time, torch
from einops import rearrange
import skimage.color as sc
import imageio
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import utils
from tqdm import tqdm
import data.common as common
import src.basic_module as arch

class ReconstructionLoss(nn.Module):
    def __init__(self, losstype='l2', eps=1e-6):
        super(ReconstructionLoss, self).__init__()
        self.losstype = losstype
        self.eps = eps

    def forward(self, x, target):
        if self.losstype == 'l2':
            return torch.mean(torch.sum((x - target) ** 2, (1, 2, 3)))
        elif self.losstype == 'l1':
            diff = x - target
            return torch.mean(torch.sum(torch.sqrt(diff ** 2 + 1e-6), (1, 2, 3)))
        else:
            print("reconstruction loss type error!")
            return 0


def train(training_dataloader, optimizer, model, epoch, writer, args):
    scaler = torch.cuda.amp.GradScaler()
    if args.task == 'rescaling':
        criterion_reverse = ReconstructionLoss(losstype='l1').to(args.device)
        criterion_forward = ReconstructionLoss(losstype='l2').to(args.device)
    elif args.task == 'decolorization':
        criterion_reverse = ReconstructionLoss(losstype='l1').to(args.device)
        criterion_forward = ReconstructionLoss(losstype='l2').to(args.device)
    elif args.task == 'hiding':
        criterion_reverse = ReconstructionLoss(losstype='l1').to(args.device)
        criterion_forward = ReconstructionLoss(losstype='l1').to(args.device)
    else:
        raise InterruptedError

    model.train()
    Quantization = arch.Quantization().to(args.device)
    torch.cuda.empty_cache()

    with tqdm(total=len(training_dataloader), ncols=224) as pbar:
        for iteration, (LQ_img, HQ_img) in enumerate(training_dataloader):

            LQ_img = Variable(LQ_img).to(args.device)
            if args.scale == 4:
                LQ_img = LQ_img.repeat(1, 4, 1, 1)
            HQ_img = Variable(HQ_img).to(args.device)
            loss = 0.0

            LQ_est, LQ_Plus, loss_det = model(HQ_img, rev=False)

            loss += args.lambda_det * loss_det

            loss_shift = 0
            for item_plus in LQ_Plus:
                instance_plus = LQ_Plus[item_plus]
                channels = int(item_plus.split('chn')[1])

                instance_plus_ref = arch.get_shift_m(
                    LQ_img.clone().detach().requires_grad_(False),
                    # instance_plus[:, :channels, :, :].clone().detach().requires_grad_(False),
                    num=instance_plus.shape[1] // channels)
                loss_shift += criterion_forward(instance_plus[:, channels:, :, :],
                                                instance_plus_ref[:, channels:, :, :])

            loss += args.lambda_shift * loss_shift

            loss_forward = criterion_forward(LQ_est, LQ_img)
            if args.scale == 4:
                LQ_est = rearrange(LQ_est, 'b (p c) h w -> b p c h w', p=4)
                LQ_est = torch.mean(LQ_est, dim=1)
                loss_forward += criterion_forward(LQ_est, LQ_img[:, :3, :, :])

            loss += args.lambda_forward * loss_forward

            if args.lambda_reverse > 0:
                if args.quantization:
                    LQ_est = Quantization(LQ_est)

                if args.scale == 4:
                    LQ_est = LQ_est.repeat(1, 4, 1, 1)

                HQ_est = model(LQ_est, rev=True)
                loss_reverse = criterion_reverse(HQ_est, HQ_img)
                if args.task == 'hiding':
                    loss_reverse += criterion_reverse(HQ_est[:, args.out_channels:, :, :],
                                                      HQ_img[:, args.out_channels:, :, :])
                elif args.task == 'composition11':
                    loss_reverse += criterion_reverse(HQ_est[:, args.out_channels:, :, :],
                                                      HQ_img[:, args.out_channels:, :, :])
                elif args.task == 'rescaling':
                    loss_reverse += criterion_reverse(HQ_est, HQ_img)
                loss += args.lambda_reverse * loss_reverse

            loss.sum().backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0, norm_type=2.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            time.sleep(0.1)
            pbar.update(1)
            pbar.set_postfix(Epoch=epoch,
                             LeaRate='{:.3e}'.format(optimizer.param_groups[0]['lr']),
                             Loss='DG@{}*{:.2e}+SF@{}*{:.2e}+JC@{}*{:.2e}+RC@{}*{:.2e}'
                             .format(args.lambda_forward,
                                     loss_forward.sum().data if 'loss_forward' in locals().keys() else 0,
                                     args.lambda_shift, loss_shift.sum().data if 'loss_shift' in locals().keys() else 0,
                                     args.lambda_det, loss_det.sum().data if 'loss_det' in locals().keys() else 0,
                                     args.lambda_reverse,
                                     loss_reverse.sum().data if 'loss_reverse' in locals().keys() else 0,
                                     ))

            niter = (epoch - 1) * len(training_dataloader) + iteration

            if (niter + 1) % 20 == 0:
                writer.add_scalar('Train-Loss/loss_forward', loss_forward.sum().data, niter)
                writer.add_scalar('Train-Loss/loss_shift', loss_shift.sum().data, niter)
                writer.add_scalar('Train-Loss/loss_det', loss_det.sum().data, niter)
                writer.add_scalar('Train-Loss/loss_reverse', loss_reverse.sum().data, niter)

        torch.cuda.empty_cache()


def test_hiding(source_path, result_path, model, args, f_csv=None):
    if isinstance(model, dict):
        for item in model:
            model[item].eval()
    elif model is not None:
        model.eval()
    count = 0
    factor = 20 if args.num_secrets == 1 else 10
    Avg_PSNR_Forward = 0
    Avg_PSNR_Reverse = [0 for _ in range(args.num_secrets)]
    Avg_SSIM_Forward = 0
    Avg_SSIM_Reverse = [0 for _ in range(args.num_secrets)]
    Avg_Time = 0

    PSNR_Reverse = [0 for _ in range(args.num_secrets)]
    SSIM_Reverse = [0 for _ in range(args.num_secrets)]
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    filename = os.listdir(source_path)
    filename.sort()
    val_length = len(filename) // (args.num_secrets + 1)
    val_length = 2
    torch.cuda.empty_cache()

    if 'COCO' in source_path:
        cropsize = [256, 256]
    elif 'ImageNet' in source_path:
        cropsize = [256, 256]
    elif 'DIV2K' in source_path:
        cropsize = [1024, 1024]
    else:
        cropsize = [None, None]

    if f_csv:
        records = ['image name', 'PSNR(Y) Stego']
        for idx_secret in range(args.num_secrets):
            records.append('PSNR(Y) Secret#{}'.format(idx_secret + 1))
        records.append('SSIM(Y) Stego')
        for idx_secret in range(args.num_secrets):
            records.append('SSIM(Y) Secret#{}'.format(idx_secret + 1))
        records.append('Time (ms)')
        f_csv.writerow(records)

    with torch.no_grad():
        with tqdm(total=val_length, ncols=224) as pbar:
            for idx_img in range(val_length):
                img_name = filename[idx_img * (args.num_secrets + 1)]
                Cover_img = imageio.imread(os.path.join(source_path, img_name))  # cover image
                Cover_img = common.set_channel(Cover_img, args.out_channels)
                Cover_img = utils.crop_center(Cover_img, cropsize[0], cropsize[1])
                Cover_img = common.np2Tensor(Cover_img, args.value_range)
                img_name_cover, ext = os.path.splitext(img_name)

                Secret_imgs, img_name_secrets = [], []
                for idx_secret in range(args.num_secrets):
                    img_name = filename[idx_img * (args.num_secrets + 1) + idx_secret + 1]
                    Secret_img = imageio.imread(os.path.join(source_path, img_name))  # cover image
                    img_name_secret, ext = os.path.splitext(img_name)
                    Secret_img = common.set_channel(Secret_img, args.in_channels)
                    Secret_img = utils.crop_center(Secret_img, cropsize[0], cropsize[1])
                    Secret_img = common.np2Tensor(Secret_img, args.value_range)
                    Secret_imgs.append(Secret_img)
                    img_name_secrets.append(img_name_secret)

                HQ_img = torch.cat((Cover_img, torch.cat(Secret_imgs, dim=0)), dim=0)
                HQ_img = Variable(HQ_img[None]).to(args.device)
                H, W = HQ_img.shape[2:]
                HQ_img = HQ_img[:, :, :(H - H % 2), :(W - W % 2)]

                start.record()
                Cover_est = model(HQ_img, rev=False)
                Cover_est = Cover_est[:, :, :HQ_img.shape[2], :HQ_img.shape[3]]
                Secret_est = model(Cover_est, rev=True)
                Secret_est = Secret_est[:, args.out_channels:, :, :]
                end.record()
                torch.cuda.synchronize()
                Time = start.elapsed_time(end)
                Avg_Time += Time
                count += 1

                if len(Cover_est.size()) == 4:
                    Cover_est = Cover_est.data[0].cpu().clamp(0, 1)
                    Secret_est = torch.chunk(Secret_est.data[0].cpu().clamp(0, 1), chunks=args.num_secrets, dim=0)

                PSNR_Forward = utils.calc_PSNR(Cover_est, Cover_img, rgb_range=args.value_range, shave=0)
                SSIM_Forward = utils.calc_SSIM(Cover_est, Cover_img, rgb_range=args.value_range, shave=0)
                Avg_PSNR_Forward += PSNR_Forward
                Avg_SSIM_Forward += SSIM_Forward
                for idx_secret in range(args.num_secrets):
                    PSNR_Reverse[idx_secret] = utils.calc_PSNR(Secret_est[idx_secret], Secret_imgs[idx_secret], rgb_range=args.value_range, shave=0)
                    SSIM_Reverse[idx_secret] = utils.calc_SSIM(Secret_est[idx_secret], Secret_imgs[idx_secret], rgb_range=args.value_range, shave=0)
                    Avg_PSNR_Reverse[idx_secret] += PSNR_Reverse[idx_secret]
                    Avg_SSIM_Reverse[idx_secret] += SSIM_Reverse[idx_secret]

                if f_csv:
                    records = [img_name, PSNR_Forward]
                    for idx_secret in range(args.num_secrets):
                        records.append(PSNR_Reverse[idx_secret])
                    records.append(SSIM_Forward)
                    for idx_secret in range(args.num_secrets):
                        records.append(SSIM_Reverse[idx_secret])
                    records.append(Time)
                    f_csv.writerow(records)

                if args.save_img:
                    Res_Cover = (Cover_est - Cover_img).abs() * factor
                    Cover_est = utils.save_img(Cover_est, args.out_channels, 255)
                    Cover_est.save(result_path + '/{}_stego.png'.format(img_name_cover))
                    for idx_secret in range(args.num_secrets):
                        Secret_est_ = Secret_est[idx_secret]
                        Res_Secret = (Secret_est_ - Secret_imgs[idx_secret]).abs() * factor
                        Secret_est_ = utils.save_img(Secret_est_, args.out_channels, 255)
                        Secret_est_.save(result_path + '/{}_secret.png'.format(img_name_secrets[idx_secret]))
                        Res_Secret = utils.save_img(Res_Secret, args.out_channels, 255)
                        Res_Secret.save(result_path + '/{}_secret_resi.png'.format(img_name_secrets[idx_secret]))

                    Res_Cover = utils.save_img(Res_Cover, args.out_channels, 255)
                    Res_Cover.save(result_path + '/{}_stego_resi.png'.format(img_name_cover))

                time.sleep(0.1)
                pbar.update(1)
                Avg_PSNR_Reverse_All = np.sum(Avg_PSNR_Reverse, axis=0) / (count * args.num_secrets)
                pbar.set_postfix(PSNR_F='{:.2f}'.format(Avg_PSNR_Forward / count),
                                 PSNR_R=['S{}:{:.2f}'.format(idx_secret_+1, Avg_PSNR_Reverse[idx_secret_]/count) for idx_secret_ in range(args.num_secrets)],
                                 SSIM_F='{:.4f}'.format(Avg_SSIM_Forward / count),
                                 SSIM_R=['S{}:{:.4f}'.format(idx_secret_+1, Avg_SSIM_Reverse[idx_secret_]/count) for idx_secret_ in range(args.num_secrets)],
                                 TIME='{:.1f}ms'.format(Avg_Time / count),
                                 )
    torch.cuda.empty_cache()

    if f_csv:
        records = ['Avg', Avg_PSNR_Forward / count]
        for idx_secret in range(args.num_secrets):
            records.append(Avg_PSNR_Reverse[idx_secret] / count)
        records.append(Avg_SSIM_Forward / count)
        for idx_secret in range(args.num_secrets):
            records.append(Avg_SSIM_Reverse[idx_secret] / count)
        records.append(Avg_Time / count)
        f_csv.writerow(records)
    print('Avg secrets PSNR = {:.2f}\t SSIM = {:.4f}'.format(Avg_PSNR_Reverse_All, np.sum(Avg_SSIM_Reverse, axis=0) / (count * args.num_secrets)))
    return Avg_PSNR_Forward / count, Avg_PSNR_Reverse_All, Avg_Time / count

def test(source_path, result_path, model, args, method='GICNet_Naive', f_csv=None, saveimg=False):
    model.eval()

    count = 0
    Avg_PSNR_Degrade_Y = 0
    Avg_PSNR_Recover_Y = 0
    Avg_PSNR_Degrade_RGB = 0
    Avg_PSNR_Recover_RGB = 0
    Avg_SSIM_Degrade_Y = 0
    Avg_SSIM_Recover_Y = 0
    Avg_SSIM_Degrade_RGB = 0
    Avg_SSIM_Recover_RGB = 0
    Avg_Time = 0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    filename = os.listdir(source_path)

    torch.cuda.empty_cache()

    with torch.no_grad():
        with tqdm(total=len(filename), ncols=128) as pbar:
            for img_name in filename:
                HQ_img = imageio.imread(os.path.join(source_path, img_name))
                img_name, ext = os.path.splitext(img_name)
                if args.task == 'rescaling':
                    source_path_LQ = source_path.split('HR')[0] + 'LR_bicubic/X{}'.format(args.scale)
                    LQ_img = imageio.imread(os.path.join(source_path_LQ, img_name + '.png'))
                elif args.task == 'decolorization':
                    LQ_img = sc.rgb2lab(HQ_img) / 100.
                    LQ_img = (255. * LQ_img[:, :, 0]).astype(np.uint8)

                HQ_img = common.set_channel(HQ_img, args.in_channels)
                HQ_img = common.np2Tensor(HQ_img, args.value_range)
                LQ_img = common.set_channel(LQ_img, args.out_channels)
                LQ_img = common.np2Tensor(LQ_img, args.value_range)

                LQ_img = LQ_img[None].to(args.device)
                HQ_img = HQ_img[None].to(args.device)
                H, W = HQ_img.shape[2:]
                HQ_img = HQ_img[:, :, :(H - H%args.scale), :(W - W%args.scale)]
                H, W = HQ_img.shape[2:]
                start.record()

                LQ_est = model(HQ_img, rev=False)
                LQ_est = LQ_est[:, :, :LQ_img.shape[2], :LQ_img.shape[3]]
                if args.scale == 4:
                    LQ_est = rearrange(LQ_est, 'b (p c) h w -> b p c h w', p=4)
                    LQ_est = torch.mean(LQ_est, dim=1)
                LQ_est = Quantization(LQ_est)
                LQ_est_ = LQ_est
                if args.scale == 4:
                    LQ_est = LQ_est.repeat(1, 4, 1, 1)
                HQ_est = model(LQ_est, rev=True)
                LQ_est = LQ_est_

                end.record()
                torch.cuda.synchronize()
                Time = start.elapsed_time(end)
                Avg_Time += Time
                count += 1

                if len(HQ_est.size()) == 4:
                    LQ_est = LQ_est.data[0].cpu().clamp(0, 1)
                    HQ_est = HQ_est.data[0].cpu().clamp(0, 1)

                # PSNR_Degrade_Y = utils.calc_PSNR_Y(LQ_est, LQ_img.data[0].to(LQ_est.device),
                #                                    rgb_range=args.value_range, shave=0)
                # Avg_PSNR_Degrade_Y += PSNR_Degrade_Y
                # PSNR_Recover_Y = utils.calc_PSNR_Y(HQ_est, HQ_img.data[0].to(LQ_est.device),
                #                                    rgb_range=args.value_range, shave=0 if args.scale<=1 else args.scale)
                # Avg_PSNR_Recover_Y += PSNR_Recover_Y
                PSNR_Degrade_RGB = utils.calc_PSNR(LQ_est, LQ_img.data[0].to(LQ_est.device),
                                                   rgb_range=args.value_range, shave=0)
                Avg_PSNR_Degrade_RGB += PSNR_Degrade_RGB
                PSNR_Recover_RGB = utils.calc_PSNR(HQ_est, HQ_img.data[0].to(LQ_est.device),
                                                   rgb_range=args.value_range, shave=0 if args.scale<=1 else args.scale)
                Avg_PSNR_Recover_RGB += PSNR_Recover_RGB


                # SSIM_Degrade_Y = utils.calc_SSIM_Y(LQ_est, LQ_img.data[0].to(LQ_est.device),
                #                                    rgb_range=args.value_range, shave=0)
                # Avg_SSIM_Degrade_Y += SSIM_Degrade_Y
                # SSIM_Recover_Y = utils.calc_SSIM_Y(HQ_est, HQ_img.data[0].to(LQ_est.device),
                #                                    rgb_range=args.value_range, shave=0 if args.scale<=1 else args.scale)
                # Avg_SSIM_Recover_Y += SSIM_Recover_Y
                SSIM_Degrade_RGB = utils.calc_SSIM(LQ_est, LQ_img.data[0].to(LQ_est.device),
                                                   rgb_range=args.value_range, shave=0)
                Avg_SSIM_Degrade_RGB += SSIM_Degrade_RGB
                SSIM_Recover_RGB = utils.calc_SSIM(HQ_est, HQ_img.data[0].to(LQ_est.device),
                                                   rgb_range=args.value_range, shave=0 if args.scale<=1 else args.scale)
                Avg_SSIM_Recover_RGB += SSIM_Recover_RGB

                if saveimg:
                    LQ_est = utils.save_img(LQ_est, args.out_channels, 255)
                    LQ_est.save(result_path + '/{}_LQ.png'.format(img_name))
                    HQ_est = utils.save_img(HQ_est, args.in_channels, 255)
                    HQ_est.save(result_path + '/{}_HQ.png'.format(img_name))

                if f_csv:
                    f_csv.writerow([img_name,
                                    PSNR_Degrade_RGB, PSNR_Recover_RGB,
                                    SSIM_Degrade_RGB, SSIM_Recover_RGB, Time])

                time.sleep(0.1)
                pbar.update(1)
                pbar.set_postfix(#Degrade_Y='{:.2f} / {:.4f}'.format(Avg_PSNR_Degrade_Y / count, Avg_SSIM_Degrade_Y / count),
                                 #Recover_Y='{:.2f}dB / {:.4f}'.format(Avg_PSNR_Recover_Y / count, Avg_SSIM_Recover_Y / count),
                                 Degrade_RGB='{:.2f} / {:.4f}'.format(Avg_PSNR_Degrade_RGB / count, Avg_SSIM_Degrade_RGB / count),
                                 Recover_RGB='{:.2f} / {:.4f}'.format(Avg_PSNR_Recover_RGB / count, Avg_SSIM_Recover_RGB / count),
                                 TIME='{:.1f}ms'.format(Avg_Time / count),
                                 )

    torch.cuda.empty_cache()

    if f_csv:
        f_csv.writerow(['Avg',
                        #Avg_PSNR_Degrade_Y / count, Avg_PSNR_Recover_Y / count,
                        Avg_PSNR_Degrade_RGB / count, Avg_PSNR_Recover_RGB / count,
                        #Avg_SSIM_Degrade_Y / count, Avg_SSIM_Recover_Y / count,
                        Avg_SSIM_Degrade_RGB / count, Avg_SSIM_Recover_RGB / count,
                        Avg_Time / count])

    return Avg_PSNR_Degrade_RGB / count, Avg_PSNR_Recover_RGB / count, Avg_Time / count

def Quantization(x):
    y = torch.clamp(x, 0, 1)
    y = (y * 255.).round() / 255.
    return y
