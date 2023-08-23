import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vision_transformer import DTASwinUnet as ViT_seg
from trainer import trainer_cloud
from config import get_config
from datasets.dataset_cloud import Cloud_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./data/swinyseg/', help='root dir for data')
parser.add_argument('--volume_path', type=str,default='./data/swinyseg/', help='root dir for validation volume data')
parser.add_argument('--dataset', type=str,
                    default='Cloud', help='experiment_name')
parser.add_argument('--list_dir', type=str, default='./lists/list_swinyseg', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')


parser.add_argument('--output_dir', type=str, default='./output/output_swinyseg', help='output dir')


parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=50, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=16, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--cfg', type=str, default='./configs/swin_tiny_patch4_window7_224_lite.yaml',required=False, metavar="FILE",help='path to config file')


parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()
if args.dataset == "Cloud":
    args.root_path = os.path.join(args.root_path, "train_npz")
    args.volume_path = os.path.join(args.volume_path, "test_vol_h5")
config = get_config(args)


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset

    dataset_config = {
        'Cloud': {
            'Dataset': Cloud_dataset,
            'root_path': args.root_path,
            'volume_path': args.volume_path,
            'list_dir': './lists/list_swinyseg',
            'num_classes': 2,
            'z_spacing': 1,
        },
    }
    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    net.load_from(config)

    trainer = {'Cloud': trainer_cloud,}
    trainer[dataset_name](args, net, args.output_dir)