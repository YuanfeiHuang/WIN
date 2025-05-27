import os, shutil, torch, csv, random
from torch.utils.data import DataLoader
import data.dataloaders as Dataloaders
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from train import *
from model import WIN, WIN_Naive
from option import args


def main():
    global opt
    opt = utils.print_args(args)
    if torch.cuda.is_available() and opt.cuda:
        opt.device = torch.device("cuda:0")
    else:
        opt.device = torch.device('cpu')
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    cudnn.benchmark = True

    print("===> Building model")
    if opt.method == 'WIN':
        model = WIN(
            in_channels=opt.in_channels * (opt.num_secrets + 1) if args.task.lower() == 'hiding' else opt.in_channels,
            out_channels=opt.out_channels,
            n_channels=opt.n_channels,
            splitting_ratio=opt.split_ratio,
            n_modules=opt.n_modules,
            n_blocks=opt.n_blocks,
            scale=opt.scale)
    elif opt.method == 'WIN_Naive':
        model = WIN_Naive(
            in_channels=opt.in_channels * (opt.num_secrets + 1) if args.task.lower() == 'hiding' else opt.in_channels,
            out_channels=opt.out_channels,
            n_blocks=opt.n_blocks,
            scale=opt.scale)
    else:
        raise InterruptedError

    if len(opt.GPUs) > 1:
        model = nn.DataParallel(model, device_ids=opt.GPUs)
    model = model.to(opt.device)

    model = utils.load_checkpoint(opt.resume, model, opt.GPUs, opt.cuda)

    if opt.train.lower() == 'train':

        print("===> Setting Optimizer")
        para = filter(lambda x: x.requires_grad, model.parameters())
        optimizer = opt.optimizer([{'params': para, 'initial_lr': opt.lr}],
                                  lr=opt.lr, betas=(0.9, 0.999), weight_decay=1e-4)

        scheduler = utils.learning_rate_scheduler(optimizer, opt.lr_type, opt.start_epoch,
                                                  opt.lr_gamma_1, opt.lr_gamma_2)

        model.train()
        if os.path.exists(opt.model_path + '/' + 'runs'):
            shutil.rmtree(opt.model_path + '/' + 'runs')
        writer = SummaryWriter(opt.model_path + '/runs')

        start_epoch = opt.start_epoch if opt.start_epoch >= 0 else 0

        for data_test in opt.data_test:
            print('===> Testing on {}'.format(data_test))
            result_path = opt.model_path + '/Results/{}'.format(data_test)
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            if args.task == 'rescaling':
                PSNR_Degrade, PSNR_Recovery, Time = test_rescaling(opt.dir_data + 'Test/{}/HR'.format(data_test),
                                                                   result_path, model, opt)
            elif args.task == 'decolorization':
                PSNR_Degrade, PSNR_Recovery, Time = test_decolorization(opt.dir_data + 'Test/{}/HR'.format(data_test),
                                                                        result_path, model, opt)
            elif args.task == 'hiding':
                PSNR_Degrade, PSNR_Recovery, Time = test_hiding(opt.dir_data + 'Test/{}/HR'.format(data_test),
                                                                result_path, model, opt)
            else:
                raise InterruptedError
            writer.add_scalar('PSNR_Degrade_{}/'.format(data_test), PSNR_Degrade, 0)
            writer.add_scalar('PSNR_Recovery_{}/'.format(data_test), PSNR_Recovery, 0)


        print('===> Building Training dataloader on {}'.format(args.data_train))
        if opt.task == 'rescaling':
            trainset = Dataloaders.dataloader_rescaling(opt)
        elif opt.task == 'decolorization':
            trainset = Dataloaders.dataloader_decolor(opt)
        elif opt.task == 'hiding':
            trainset = Dataloaders.dataloader_hiding(opt)
        else:
            raise InterruptedError

        train_dataloader = DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.threads)
        torch.cuda.empty_cache()

        for epoch in range(start_epoch + 1, opt.n_epochs + 1):
            print('===> Training on DIV2K dataset')
            train(train_dataloader, optimizer, model, epoch, writer, opt)

            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
            scheduler.step()

            for data_test in opt.data_test:
                print('===> Testing on {}'.format(data_test))
                result_path = opt.model_path + '/Results/{}'.format(data_test)
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                if args.task == 'rescaling':
                    PSNR_Degrade, PSNR_Recovery, Time = test(opt.dir_data + 'Test/{}/HR'.format(data_test),
                                                                       result_path, model, opt)
                elif args.task == 'decolorization':
                    PSNR_Degrade, PSNR_Recovery, Time = test(
                        opt.dir_data + 'Test/{}/HR'.format(data_test), result_path, model, opt)
                elif args.task == 'hiding':
                    PSNR_Degrade, PSNR_Recovery, Time = test_hiding(opt.dir_data + 'Test/{}/HR'.format(data_test),
                                                                    result_path, model, opt)
                else:
                    raise InterruptedError
                writer.add_scalar('PSNR_Degrade_{}/'.format(data_test), PSNR_Degrade, epoch)
                writer.add_scalar('PSNR_Recovery_{}/'.format(data_test), PSNR_Recovery, epoch)

            model_path = opt.model_path + '/Checkpoints/checkpoint_epoch_{}.pth'.format(epoch)
            torch.save(model.state_dict(), model_path)
            print('Checkpoint saved to {}'.format(model_path))
            torch.cuda.empty_cache()
        writer.close()

    elif opt.train.lower() == 'test':
        for data_test in opt.data_test:
            print('===> Testing on {}'.format(data_test))
            result_path = opt.model_path + '/Results/{}'.format(data_test)
            if not os.path.exists(result_path):
                os.makedirs(result_path)

            with open(opt.model_path + '/Results/Results_{}.csv'.format(data_test), 'w', newline='') as f:
                f_csv = csv.writer(f)
                f_csv.writerow(
                    ['image_name', 'PSNR Degrade', 'PSNR Recovery', 'SSIM Degrade', 'SSIM Recovery', 'Time (ms)'])

                if args.task == 'rescaling':
                    test(
                        opt.dir_data + 'Test/{}/HR'.format(data_test), result_path, model, opt, f_csv)
                elif args.task == 'decolorization':
                    test(
                        opt.dir_data + 'Test/{}/HR'.format(data_test), result_path, model, opt, f_csv)
                elif args.task == 'hiding':
                    test_hiding(
                        opt.dir_data + 'Test/{}/HR'.format(data_test), result_path, model, opt, f_csv)

    elif args.train.lower() == 'complexity':

        from thop import profile
        sz_H, sz_W = 512, 512
        input = torch.FloatTensor(1, opt.in_channels * (opt.num_secrets + 1), sz_H, sz_W).to(opt.device)
        FLOPs, Params = profile(model.module if len(opt.GPUs) > 1 else model, inputs=(input, False,), verbose=False)
        print('-------------Degrade Phase-------------')
        print('\tParam = {:.3f}K\n\tFLOPs = {:.3f}G on {}'.format(Params * 1e-3, FLOPs * 1e-9, input.shape))
        torch.cuda.empty_cache()

        input = torch.FloatTensor(1, opt.out_channels, sz_H // opt.scale, sz_W // opt.scale).to(opt.device)
        FLOPs, Params = profile(model.module if len(opt.GPUs) > 1 else model, inputs=(input, True,), verbose=False)
        print('-------------Recovery Phase-------------')
        print('\tParam = {:.3f}K\n\tFLOPs = {:.3f}G on {}'.format(Params * 1e-3, FLOPs * 1e-9, input.shape))
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
