import os
from test import *

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--submission_outdir',
        help='output result file name without extension',
        type=str)
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--load', action='store_true', help='load results for evaluation')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    args = parse_args()

    assert args.out or args.show \
        ('Please specify at least one operation (save or show the results) '
         'with the argument "--out" or "--show" ')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')


    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not args.load:
        if not distributed:
            model = MMDataParallel(model, device_ids=[0])
            outputs = single_gpu_test(model, data_loader, args.show)
        else:
            model = MMDistributedDataParallel(model.cuda())
            outputs = multi_gpu_test(model, data_loader, args.tmpdir)

        rank, _ = get_dist_info()
        if args.out and rank == 0:
            print('\nwriting results to {}'.format(args.out))
            mmcv.dump(outputs, args.out)
    else:
        outputs=mmcv.load(args.out)

    if args.submission_outdir is not None:
        if not isinstance(outputs[0], dict):
            result2submission(dataset, outputs, args.submission_outdir)
        else:
            for name in outputs[0]:
                outputs_ = [out[name] for out in outputs]
                result_file = args.submission_outdir + '.{}'.format(name)
                result2submission(dataset, outputs_, result_file)

def result2submission(dataset, outputs, sub_dir, score_thres=0.3):
    img_infos = dataset.img_infos
    assert len(img_infos) == len(outputs)
    for info, out in zip(img_infos, outputs):
        out_folder, out_fn = os.path.split(info['filename'])
        header = out_fn[:-4]
        out_folder = os.path.join(sub_dir, out_folder)
        out_fn = osp.join(out_folder, out_fn[:-3]+'txt')
        print('writing ' + out_fn)
        if not os.path.isdir(out_folder): os.makedirs(out_folder, exist_ok=True)
        res = out[0]
        res = res[res[:,-1]>score_thres]
        n_faces = len(res)
        with open(out_fn, 'w') as f:
            f.write(header +'\n')
            f.write(str(n_faces) + '\n')
            
            if n_faces>0:
                for r in res:
                    f.write('{:.2f} {:.2f} {:.2f} {:.2f} {:.3f}\n'.format(*r.tolist()))
            else:
                f.write('0 0 0 0 0\n')
            f.write('\n')


if __name__ == '__main__':
    main()
