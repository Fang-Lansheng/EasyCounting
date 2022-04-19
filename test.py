from misc.utilities import *


def parse_args():
    parser = argparse.ArgumentParser(description='Script for training TransNet')

    parser.add_argument('--ckpt-path', type=str, help='checkpoint file path',
                        default='./Experiments/CSRNet_SHHA_20211231_235959/ckpt_best.pth')
    parser.add_argument('--gpu-ids', type=int,
                        help='train config file path')

    args = parser.parse_args()

    if torch.cuda.is_available():
        if args.gpu_ids is not None:
            torch.cuda.set_device(args.gpu_ids)
        else:
            torch.cuda.set_device(0)

    return args


################################################################################
# main function
################################################################################
if __name__ == '__main__':
    ckpt_path = parse_args().ckpt_path

    # load checkpoint
    if torch.cuda.is_available():
        ckpt = torch.load(ckpt_path)
    else:
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))

    # get configs
    cfg = ckpt['cfg']

    # prepare log file
    prepare(cfg, mode='test')

    # load test set data loader
    val_loader = get_dataloader(cfg, mode='test')

    # Visualization option
    vis_options = {'img_raw':           False,
                   'gt_dot_map':        False,
                   'gt_density':        True,
                   'pr_density':        True,
                   'intermediate':      False}

    # test model
    tester = get_tester(ckpt, cfg, val_loader, vis_options)
    tester.run()
