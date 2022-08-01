"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np
import torch.optim as optim
from timm.scheduler import CosineLRScheduler
from pathlib import Path
from tqdm import tqdm
from dataset import S3DISDataset, ScannetDatasetWholeScene
import ipdb
from data_utils.indoor3d_util import g_label2color


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
           'board', 'clutter']

g_class2color = {'ceiling':	[0,255,0],
                 'floor':	[0,0,255],
                 'wall':	[0,255,255],
                 'beam':        [255,255,0],
                 'column':      [255,0,255],
                 'window':      [100,100,255],
                 'door':        [200,200,100],
                 'table':       [170,120,200],
                 'chair':       [255,0,0],
                 'sofa':        [200,100,100],
                 'bookcase':    [10,200,100],
                 'board':       [200,200,200],
                 'clutter':     [50,50,50]}

g_classindex2color = [[0,255,0],
                 [0,0,255],
                 [0,255,255],
                 [255,255,0],
                 [255,0,255],
                 [100,100,255],
                 [200,200,100],
                 [170,120,200],
                 [255,0,0],
                 [200,100,100],
                 [10,200,100],
                 [200,200,200],
                 [50,50,50]]

def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool

class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pt', help='model name')
    parser.add_argument('--optimizer_part', type=str, default='all', help='training all parameters or optimizing the new layers only')
    parser.add_argument('--batch_size', type=int, default=32, help='batch Size during training')
    parser.add_argument('--epoch', default=60, type=int, help='epoch to run')
    parser.add_argument('--warmup_epoch', default=10, type=int, help='warmup epoch')
    parser.add_argument('--learning_rate', default=0.0002, type=float, help='initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    # parser.add_argument('--optimizer', type=str, default='AdamW', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default='./exp', help='log path')
    # parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    # parser.add_argument('--npoint', type=int, default=2048, help='point Number')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    # parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay')
    # parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')
    parser.add_argument('--ckpts', type=str, default=None, help='ckpts')
    parser.add_argument('--root', type=str, default='../data/stanford_indoor3d/', help='data root')
    parser.add_argument('--num_point', type=int, default=2048, help='point number [default: 4096]')
    parser.add_argument('--test_area', type=int, default=5, help='area for testing, option: 1-6 [default: 5]')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting [default: 5]')
    parser.add_argument('--visual', action='store_true', default=False, help='visualize result [default: False]')
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/semantic_seg/' + args.log_dir
    visual_dir = experiment_dir + '/visual/'
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    NUM_CLASSES = 13
    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point

    root = args.root #'data/s3dis/stanford_indoor3d/'

    TEST_DATASET_WHOLE_SCENE = ScannetDatasetWholeScene(root, split='test', test_area=args.test_area, block_points=NUM_POINT)
    log_string("The number of test data is: %d" % len(TEST_DATASET_WHOLE_SCENE))

    '''MODEL LOADING'''
    # model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    # MODEL = importlib.import_module(model_name)
    # classifier = MODEL.get_model(NUM_CLASSES).cuda()
    # checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    # classifier.load_state_dict(checkpoint['model_state_dict'])
    # classifier = classifier.eval()

    '''MODEL LOADING'''
    ckpts = args.ckpts
    MODEL = importlib.import_module(args.model)
    # shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    classifier.apply(inplace_relu)
    print('# generator parameters:', sum(param.numel() for param in classifier.parameters()))
    classifier.load_model_from_ckpt_withrename(ckpts)
    classifier = classifier.eval()


    with torch.no_grad():
        scene_id = TEST_DATASET_WHOLE_SCENE.file_list
        scene_id = [x[:-4] for x in scene_id]
        num_batches = len(TEST_DATASET_WHOLE_SCENE)

        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

        log_string('---- EVALUATION WHOLE SCENE----')

        for batch_idx in range(num_batches):
            print("Inference [%d/%d] %s ..." % (batch_idx + 1, num_batches, scene_id[batch_idx]))
            total_seen_class_tmp = [0 for _ in range(NUM_CLASSES)]
            total_correct_class_tmp = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class_tmp = [0 for _ in range(NUM_CLASSES)]
            if args.visual:
                fout = open(os.path.join(visual_dir, scene_id[batch_idx] + '_pred.obj'), 'w')
                fout_gt = open(os.path.join(visual_dir, scene_id[batch_idx] + '_gt.obj'), 'w')

            whole_scene_data = TEST_DATASET_WHOLE_SCENE.scene_points_list[batch_idx]
            whole_scene_label = TEST_DATASET_WHOLE_SCENE.semantic_labels_list[batch_idx]
            vote_label_pool = np.zeros((whole_scene_label.shape[0], NUM_CLASSES))
            for _ in tqdm(range(args.num_votes), total=args.num_votes):
                scene_data, scene_label, scene_smpw, scene_point_index = TEST_DATASET_WHOLE_SCENE[batch_idx]
                # ipdb.set_trace()
                scene_data = scene_data[:, :, :3]
                num_blocks = scene_data.shape[0]
                s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
                batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 3))

                batch_label = np.zeros((BATCH_SIZE, NUM_POINT))
                batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))
                batch_smpw = np.zeros((BATCH_SIZE, NUM_POINT))

                for sbatch in range(s_batch_num):
                    start_idx = sbatch * BATCH_SIZE
                    end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
                    real_batch_size = end_idx - start_idx
                    batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
                    batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
                    batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]
                    batch_smpw[0:real_batch_size, ...] = scene_smpw[start_idx:end_idx, ...]
                    # batch_data[:, :, 3:6] /= 1.0

                    torch_data = torch.Tensor(batch_data)
                    torch_data = torch_data.float().cuda()
                    torch_data = torch_data.transpose(2, 1)
                    seg_pred = classifier(torch_data)
                    batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()

                    vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],
                                               batch_pred_label[0:real_batch_size, ...],
                                               batch_smpw[0:real_batch_size, ...])

            pred_label = np.argmax(vote_label_pool, 1)

            for l in range(NUM_CLASSES):
                total_seen_class_tmp[l] += np.sum((whole_scene_label == l))
                total_correct_class_tmp[l] += np.sum((pred_label == l) & (whole_scene_label == l))
                total_iou_deno_class_tmp[l] += np.sum(((pred_label == l) | (whole_scene_label == l)))
                total_seen_class[l] += total_seen_class_tmp[l]
                total_correct_class[l] += total_correct_class_tmp[l]
                total_iou_deno_class[l] += total_iou_deno_class_tmp[l]

            iou_map = np.array(total_correct_class_tmp) / (np.array(total_iou_deno_class_tmp, dtype=np.float) + 1e-6)
            print(iou_map)
            arr = np.array(total_seen_class_tmp)
            tmp_iou = np.mean(iou_map[arr != 0])
            log_string('Mean IoU of %s: %.4f' % (scene_id[batch_idx], tmp_iou))
            print('----------------------------')

            filename = os.path.join(visual_dir, scene_id[batch_idx] + '.txt')
            with open(filename, 'w') as pl_save:
                for i in pred_label:
                    pl_save.write(str(int(i)) + '\n')
                pl_save.close()
            for i in range(whole_scene_label.shape[0]):
                color = g_label2color[pred_label[i]]
                color_gt = g_label2color[whole_scene_label[i]]
                if args.visual:
                    fout.write('v %f %f %f %d %d %d\n' % (
                        whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color[0], color[1],
                        color[2]))
                    fout_gt.write(
                        'v %f %f %f %d %d %d\n' % (
                            whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color_gt[0],
                            color_gt[1], color_gt[2]))
            if args.visual:
                fout.close()
                fout_gt.close()

        IoU = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6)
        iou_per_class_str = '------- IoU --------\n'
        for l in range(NUM_CLASSES):
            iou_per_class_str += 'class %s, IoU: %.3f \n' % (
                seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])),
                total_correct_class[l] / float(total_iou_deno_class[l]))
        log_string(iou_per_class_str)
        log_string('eval point avg class IoU: %f' % np.mean(IoU))
        log_string('eval whole scene point avg class acc: %f' % (
            np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))
        log_string('eval whole scene point accuracy: %f' % (
                np.sum(total_correct_class) / float(np.sum(total_seen_class) + 1e-6)))

        print("Done!")



#     def log_string(str):
#         logger.info(str)
#         print(str)
#
#     # '''HYPER PARAMETER'''
#     # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
#
#     '''CREATE DIR'''
#     timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
#     exp_dir = Path('./log/')
#     exp_dir.mkdir(exist_ok=True)
#     exp_dir = exp_dir.joinpath('semantic_seg')
#     exp_dir.mkdir(exist_ok=True)
#     if args.log_dir is None:
#         exp_dir = exp_dir.joinpath(timestr)
#     else:
#         exp_dir = exp_dir.joinpath(args.log_dir)
#     exp_dir.mkdir(exist_ok=True)
#     checkpoints_dir = exp_dir.joinpath('checkpoints/')
#     checkpoints_dir.mkdir(exist_ok=True)
#     log_dir = exp_dir.joinpath('logs/')
#     log_dir.mkdir(exist_ok=True)
#
#     '''LOG'''
#     args = parse_args()
#     logger = logging.getLogger("Model")
#     logger.setLevel(logging.INFO)
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
#     file_handler.setLevel(logging.INFO)
#     file_handler.setFormatter(formatter)
#     logger.addHandler(file_handler)
#     log_string('PARAMETER ...')
#     log_string(args)
#
#     root = args.root
#
#     # TRAIN_DATASET = S3DISDataset(split='train', data_root=root, num_point=args.npoint, test_area=args.test_area)
#     # trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
#     # weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()
#     TEST_DATASET = ScannetDatasetWholeScene(root=args.root)
#     testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=False, num_workers=10)
#     # log_string("The number of training data is: %d" % len(TRAIN_DATASET))
#     log_string("The number of test data is: %d" % len(TEST_DATASET))
#
#
#     num_classes = 13
#     # num_part = 50
#
#     '''MODEL LOADING'''
#     ckpts_mae = './log/semantic_seg/pretrain_official_all/checkpoints/best_model.pth'
#     MODEL = importlib.import_module(args.model)
#     shutil.copy('models/%s.py' % args.model, str(exp_dir))
#     # shutil.copy('models/pointnet2_utils.py', str(exp_dir))
#     classifier = MODEL.get_model(num_classes).cuda()
#     criterion = MODEL.get_loss().cuda()
#     classifier.apply(inplace_relu)
#     print('# generator parameters:', sum(param.numel() for param in classifier.parameters()))
#     classifier.load_model_from_ckpt_withrename(ckpts_mae)
#     start_epoch = 0
#
#     '''MODEL LOADING'''
#     ckpts_masksurf = './log/semantic_seg/pretrain_withnormal_loos_w001_gradualw_all/checkpoints/best_model.pth'
#     MODEL2 = importlib.import_module(args.model)
#     shutil.copy('models/%s.py' % args.model, str(exp_dir))
#     # shutil.copy('models/pointnet2_utils.py', str(exp_dir))
#     classifier2 = MODEL2.get_model(num_classes).cuda()
#     criterion = MODEL2.get_loss().cuda()
#     classifier2.apply(inplace_relu)
#     classifier2.load_model_from_ckpt_withrename(ckpts_masksurf)
#
#     # '''MODEL LOADING'''
#     # MODEL = importlib.import_module(args.model)
#     # shutil.copy('models/%s.py' % args.model, str(exp_dir))
#     # # shutil.copy('models/pointnet2_utils.py', str(exp_dir))
#     #
#     # classifier = MODEL.get_model(num_classes).cuda()
#     # criterion = MODEL.get_loss().cuda()
#     # classifier.apply(inplace_relu)
#     # print('# generator parameters:', sum(param.numel() for param in classifier.parameters()))
#     # start_epoch = 0
#     #
#     # if args.ckpts is not None:
#     #     classifier.load_model_from_ckpt(args.ckpts)
#
# ## we use adamw and cosine scheduler
#     def add_weight_decay(model, weight_decay=1e-5, skip_list=(), optimizer_part='all'):
#         decay = []
#         no_decay = []
#         for name, param in model.named_parameters():
#             if not param.requires_grad:
#                 continue  # frozen weights
#             if optimizer_part == 'only_new':
#                 if ('cls' in name):
#                     if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
#                         # print(name)
#                         no_decay.append(param)
#                     else:
#                         decay.append(param)
#                     print(name)
#             else:
#                 if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
#                     # print(name)
#                     no_decay.append(param)
#                 else:
#                     decay.append(param)
#             # if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
#             #             # print(name)
#             #     no_decay.append(param)
#             # else:
#             #     decay.append(param)
#         return [
#                     {'params': no_decay, 'weight_decay': 0.},
#                     {'params': decay, 'weight_decay': weight_decay}]
#
#
#     param_groups = add_weight_decay(classifier, weight_decay=0.05, optimizer_part=args.optimizer_part)
#     optimizer = optim.AdamW(param_groups, lr= args.learning_rate, weight_decay=0.05 )
#
#     scheduler = CosineLRScheduler(optimizer,
#                                   t_initial=args.epoch,
#                                   t_mul=1,
#                                   lr_min=1e-6,
#                                   decay_rate=0.1,
#                                   warmup_lr_init=1e-6,
#                                   warmup_t=args.warmup_epoch,
#                                   cycle_limit=1,
#                                   t_in_epochs=True)
#
#     best_acc = 0
#     global_epoch = 0
#     best_class_avg_iou = 0
#     best_inctance_avg_iou = 0
#     best_iou = 0
#
#     classifier.zero_grad()
#     for epoch in range(0,1):
#         # mean_correct = []
#         #
#         # log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
#         # '''Adjust learning rate and BN momentum'''
#         #
#         # classifier = classifier.train()
#         # loss_batch = []
#         # num_iter = 0
#         # '''learning one epoch'''
#         # for i, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
#         #     num_iter += 1
#         #     points = points.data.numpy()
#         #     points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
#         #     points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
#         #     points = torch.Tensor(points)
#         #     points, target = points.float().cuda(),  target.long().cuda()
#         #     points = points.transpose(2, 1)
#         #
#         #     seg_pred = classifier(points)
#         #     seg_pred = seg_pred.contiguous().view(-1, num_classes)
#         #     target = target.view(-1, 1)[:, 0]
#         #     pred_choice = seg_pred.data.max(1)[1]
#         #
#         #     correct = pred_choice.eq(target.data).cpu().sum()
#         #     mean_correct.append(correct.item() / (args.batch_size * args.npoint))
#         #     loss = criterion(seg_pred, target, weights)
#         #     loss.backward()
#         #     optimizer.step()
#         #     loss_batch.append(loss.detach().cpu())
#         #
#         #     if num_iter == 1:
#         #
#         #         torch.nn.utils.clip_grad_norm_(classifier.parameters(), 10, norm_type=2)
#         #         num_iter = 0
#         #         optimizer.step()
#         #         classifier.zero_grad()
#         #
#         # if isinstance(scheduler, list):
#         #     for item in scheduler:
#         #         item.step(epoch)
#         # else:
#         #     scheduler.step(epoch)
#         #
#         # train_instance_acc = np.mean(mean_correct)
#         # loss1 = np.mean(loss_batch)
#         # log_string('Train accuracy is: %.5f' % train_instance_acc)
#         # log_string('Train loss: %.5f' % loss1)
#         # log_string('lr: %.6f' % optimizer.param_groups[0]['lr'])
#
#         NUM_CLASSES = num_classes
#         NUM_POINT = args.npoint
#         BATCH_SIZE = args.batch_size
#
#         '''Evaluate on chopped scenes'''
#         with torch.no_grad():
#             num_batches = len(testDataLoader)
#             total_correct = 0
#             total_seen = 0
#             loss_sum = 0
#             labelweights = np.zeros(NUM_CLASSES)
#             total_seen_class = [0 for _ in range(NUM_CLASSES)]
#             total_correct_class = [0 for _ in range(NUM_CLASSES)]
#             total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
#             classifier = classifier.eval()
#             classifier2 = classifier2.eval()
#
#             log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
#             data_path = f'./vis_semantic_segmentation/'
#             data_path_gt = f'./vis_semantic_segmentation/'
#             if not os.path.exists(data_path):
#                 os.makedirs(data_path)
#             selected_batch_id = [1, 5, 10, 20]
#             for batch_id, (points, target, smpw, scene_point_index) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
#                 # points: 1, 704, 4096, 9.  others: 1, 704, 4096
#                 if batch_id in selected_batch_id:
#                     ipdb.set_trace()
#                     points = points.data.numpy()
#                     points = torch.Tensor(points)
#                     points, target = points.float().cuda(), target.long().cuda()
#                     points = points.transpose(2, 1)
#
#                     seg_pred = classifier(points)
#                     seg_pred_masksurf = classifier2(points)
#
#                     points = points.cpu()
#
#                     pred_val = seg_pred.contiguous().cpu().data.numpy()
#                     # seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
#                     mae_prediction = np.argmax(pred_val, 2) ## 1* 2048
#                     label2color_mae = torch.from_numpy(np.array(g_classindex2color)[mae_prediction])[0] ## 2048*3
#                     point_color_mae = torch.cat([points[0].transpose(0, 1), label2color_mae], dim=1)
#
#
#
#                     seg_pred_masksurf = seg_pred_masksurf.contiguous().cpu().data.numpy()
#                     masksurf_prediction = np.argmax(seg_pred_masksurf, 2)
#                     label2color_masksurf = torch.from_numpy(np.array(g_classindex2color)[masksurf_prediction])[0] ## 2048*3
#                     point_color_masksurf = torch.cat([points[0].transpose(0, 1), label2color_masksurf], dim=1)
#
#                     batch_label = target.cpu().data.numpy() ## 1 * 2048
#                     label2color_gt = torch.from_numpy(np.array(g_classindex2color)[batch_label])[0] ## 2048*3
#                     point_color_gt = torch.cat([points[0].transpose(0, 1), label2color_gt], dim=1)
#                     # target = target.view(-1, 1)[:, 0]
#
#
#                     fout = open(data_path + cat + str(batch_id) + 'mae.obj', 'w')
#                     fout_masksurf = open(data_path + cat + str(batch_id) + 'masksuf.obj', 'w')
#                     fout_gt = open(data_path_gt + cat + str(batch_id) + 'gt.obj', 'w')
#                     for i in range(point_color_mae.size(0)):
#                         fout.write('v %f %f %f %d %d %d\n' % (
#                             point_color_mae[i, 0], point_color_mae[i, 1], point_color_mae[i, 2], point_color_mae[i, 3], point_color_mae[i, 4],
#                             point_color_mae[i, 5]))
#                         fout_masksurf.write('v %f %f %f %d %d %d\n' % (
#                             point_color_masksurf[i, 0], point_color_masksurf[i, 1], point_color_masksurf[i, 2],
#                             point_color_masksurf[i, 3],
#                             point_color_masksurf[i, 4],
#                             point_color_masksurf[i, 5]))
#                         fout_gt.write('v %f %f %f %d %d %d\n' % (
#                             point_color_gt[i, 0], point_color_gt[i, 1], point_color_gt[i, 2], point_color_gt[i, 3],
#                             point_color_gt[i, 4],
#                             point_color_gt[i, 5]))
#                     fout.close()
#                     fout_masksurf.close()
#                     fout_gt.close()

                # loss = criterion(seg_pred, target, weights)
                # loss_sum += loss
                # pred_val = np.argmax(pred_val, 2)
                # correct = np.sum((pred_val == batch_label))
                # total_correct += correct
                # total_seen += (BATCH_SIZE * NUM_POINT)
                # tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                # labelweights += tmp

            #     for l in range(NUM_CLASSES):
            #         total_seen_class[l] += np.sum((batch_label == l))
            #         total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
            #         total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))
            #
            # labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            # mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6))
            # log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
            # log_string('eval point avg class IoU: %f' % (mIoU))
            # log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))
            # log_string('eval point avg class acc: %f' % (
            #     np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))
            #
            # iou_per_class_str = '------- IoU --------\n'
            # for l in range(NUM_CLASSES):
            #     iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
            #         seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l - 1],
            #         total_correct_class[l] / float(total_iou_deno_class[l]))
            #
            # log_string(iou_per_class_str)
            # log_string('Eval mean loss: %f' % (loss_sum / num_batches))
            # log_string('Eval accuracy: %f' % (total_correct / float(total_seen)))
            #
            # if mIoU >= best_iou:
            #     best_iou = mIoU
            #     logger.info('Save model...')
            #     savepath = str(checkpoints_dir) + '/best_model.pth'
            #     log_string('Saving at %s' % savepath)
            #     state = {
            #         'epoch': epoch,
            #         'class_avg_iou': mIoU,
            #         'model_state_dict': classifier.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #     }
            #     torch.save(state, savepath)
            #     log_string('Saving model....')
            # log_string('Best mIoU: %f' % best_iou)
            # global_epoch += 1

        # with torch.no_grad():
        #     test_metrics = {}
        #     total_correct = 0
        #     total_seen = 0
        #     total_seen_class = [0 for _ in range(num_classes)]
        #     total_correct_class = [0 for _ in range(num_classes)]
        #     total_iou_deno_class = [0 for _ in range(num_classes)]
        #     classifier = classifier.eval()
        #     # shape_ious = {cat: [] for cat in seg_classes.keys()}
        #     # seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
        #     #
        #     # for cat in seg_classes.keys():
        #     #     for label in seg_classes[cat]:
        #     #         seg_label_to_cat[label] = cat
        #
        #
        #     for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
        #         cur_batch_size, NUM_POINT, _ = points.size()
        #         points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
        #         points = points.transpose(2, 1)
        #         seg_pred = classifier(points, to_categorical(label, num_classes))
        #         cur_pred_val = seg_pred.cpu().data.numpy()
        #         cur_pred_val_logits = cur_pred_val
        #         cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
        #         target = target.cpu().data.numpy()
        #
        #         for i in range(cur_batch_size):
        #             cat = seg_label_to_cat[target[i, 0]]
        #             logits = cur_pred_val_logits[i, :, :]
        #             cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]
        #
        #         correct = np.sum(cur_pred_val == target)
        #         total_correct += correct
        #         total_seen += (cur_batch_size * NUM_POINT)
        #
        #         for l in range(num_part):
        #             total_seen_class[l] += np.sum(target == l)
        #             total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))
        #
        #         for i in range(cur_batch_size):
        #             segp = cur_pred_val[i, :]
        #             segl = target[i, :]
        #             cat = seg_label_to_cat[segl[0]]
        #             part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
        #             for l in seg_classes[cat]:
        #                 if (np.sum(segl == l) == 0) and (
        #                         np.sum(segp == l) == 0):  # part is not present, no prediction as well
        #                     part_ious[l - seg_classes[cat][0]] = 1.0
        #                 else:
        #                     part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
        #                         np.sum((segl == l) | (segp == l)))
        #             shape_ious[cat].append(np.mean(part_ious))
        #
        #     all_shape_ious = []
        #     for cat in shape_ious.keys():
        #         for iou in shape_ious[cat]:
        #             all_shape_ious.append(iou)
        #         shape_ious[cat] = np.mean(shape_ious[cat])
        #     mean_shape_ious = np.mean(list(shape_ious.values()))
        #     test_metrics['accuracy'] = total_correct / float(total_seen)
        #     test_metrics['class_avg_accuracy'] = np.mean(
        #         np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
        #     for cat in sorted(shape_ious.keys()):
        #         log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
        #     test_metrics['class_avg_iou'] = mean_shape_ious
        #     test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)
        #
        # log_string('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Inctance avg mIOU: %f' % (
        #     epoch + 1, test_metrics['accuracy'], test_metrics['class_avg_iou'], test_metrics['inctance_avg_iou']))
        # if (test_metrics['inctance_avg_iou'] >= best_inctance_avg_iou):
        #     logger.info('Save model...')
        #     savepath = str(checkpoints_dir) + '/best_model.pth'
        #     log_string('Saving at %s' % savepath)
        #     state = {
        #         'epoch': epoch,
        #         'train_acc': train_instance_acc,
        #         'test_acc': test_metrics['accuracy'],
        #         'class_avg_iou': test_metrics['class_avg_iou'],
        #         'inctance_avg_iou': test_metrics['inctance_avg_iou'],
        #         'model_state_dict': classifier.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #     }
        #     torch.save(state, savepath)
        #     log_string('Saving model....')
        #
        # if test_metrics['accuracy'] > best_acc:
        #     best_acc = test_metrics['accuracy']
        # if test_metrics['class_avg_iou'] > best_class_avg_iou:
        #     best_class_avg_iou = test_metrics['class_avg_iou']
        # if test_metrics['inctance_avg_iou'] > best_inctance_avg_iou:
        #     best_inctance_avg_iou = test_metrics['inctance_avg_iou']
        # log_string('Best accuracy is: %.5f' % best_acc)
        # log_string('Best class avg mIOU is: %.5f' % best_class_avg_iou)
        # log_string('Best inctance avg mIOU is: %.5f' % best_inctance_avg_iou)
        # global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)