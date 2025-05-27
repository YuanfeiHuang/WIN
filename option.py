import argparse, torch

parser = argparse.ArgumentParser(description='WIN for Reversible Image Conversion')

# Hardware specifications
parser.add_argument('--cuda', default=True, action='store_true', help='Use cuda?')
parser.add_argument('--GPUs', type=str, default=[0], help='GPUs id')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loading')
parser.add_argument('--seed', type=int, default=0, help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default='../../../Datasets/', help='dataset directory')
parser.add_argument('--task', type=str, default='hiding', help='RIC task: hiding | decolorization | rescaling')
parser.add_argument('--method', type=str, default='WIN', help='method name: WIN_Naive | WIN')
parser.add_argument('--data_train', type=str, default=['DF2K'], help='train dataset name')
parser.add_argument('--data_test', type=str, default=['DIV2K_Valid'], help='test datasets name')
parser.add_argument('--n_train', type=int, default=[8], help='number of training samples')
parser.add_argument('--scale', type=int, default=1, help='scale factor: 1 for hiding or colorization | 2 or 4 for rescaling')
parser.add_argument('--n_colors', type=int, default=3, help='RGB color images')
parser.add_argument('--value_range', type=int, default=255, help='maximum value of RGB')
parser.add_argument("--store_in_ram", default=True, action="store_true", help='store the training set in RAM')
parser.add_argument("--shuffle", default=False, action="store_true", help='shuffle the training samples')
parser.add_argument("--save_img", default=False, action="store_true", help='save images in testing mode')

# Model specifications
parser.add_argument('--in_channels', type=int, default=3, help='channels for inputs: 3')
parser.add_argument('--out_channels', type=int, default=3, help='channels for outputs: 1 for decolor. | 3 for hiding or rescaling')
parser.add_argument('--n_channels', type=int, default=32, help='channels in models: only for WIN')
parser.add_argument('--n_modules', type=int, default=2, help='number of modules: 1 for WIN_Naive | 2 for WIN')
parser.add_argument('--n_blocks', type=int, default=4, help='Flow blocks in InvNN: 8 for WIN_Naive | 4 for WIN')
parser.add_argument('--split_ratio', type=float, default=0.75, help='splitting ratio for long-term memory: only for WIN')

parser.add_argument('--num_secrets', type=int, default=1, help='number of secrets: only for image hiding tasks')
parser.add_argument('--quantization', type=bool, default=False, help='use quantization?')

parser.add_argument('--lambda_forward', type=float, default=2, help='hyperparameter')
parser.add_argument('--lambda_reverse', type=float, default=1, help='hyperparameter')
parser.add_argument('--lambda_det', type=float, default=0.1, help='hyperparameter')
parser.add_argument('--lambda_shift', type=float, default=1, help='hyperparameter')

# Training/Testing specifications
parser.add_argument('--train', type=str, default='train', help='train | test | complexity')
parser.add_argument('--iter_epoch', type=int, default=10, help='iteration in each epoch')
parser.add_argument('--start_epoch', default=-1, type=int, help='start epoch for training')
parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs to train')
parser.add_argument('--patch_size', type=int, default=256, help='spatial resolution of training samples')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
parser.add_argument('--model_path', type=str, default='', help='model path')
parser.add_argument('--resume', type=str, default='', help='checkpoint path')

# Optimization specifications
parser.add_argument('--optimizer', default=torch.optim.AdamW, help='optimizer')
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
parser.add_argument('--lr_type', type=str, default='Cosine', help='initial learning rate')
parser.add_argument('--lr_gamma_1', type=int, default=300, help='learning rate decay per N epochs')
parser.add_argument('--lr_gamma_2', type=float, default=1e-6, help='min learning rate for convergence')
args = parser.parse_args()
