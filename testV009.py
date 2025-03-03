import argparse
import os
import warnings
import torch.nn.functional as F
import cv2
import torch
import torch.nn.parallel
import torch.utils.data
from loguru import logger
import time
import utils.config as config
# from engine.engine import inference_with_grasp
from engine.crog_engine import inference_with_grasp
import numpy as np
# from model import  build_segmenter
from model.modelV009S8S16S32att32depthtwinmaskcatword import CROG_TWIN
from utils.datasetV005depthmask import OCIDVLGDataset
from utils.misc import setup_logger
from tqdm import tqdm
from utils.grasp_eval import (detect_grasps, calculate_iou, calculate_max_iou, calculate_jacquard_index, visualization)


warnings.filterwarnings("ignore")
cv2.setNumThreads(0)


@torch.no_grad()
def inference_with_grasp(test_loader, model, args):
    def inverse(img, mat, w, h):
        inv_img = cv2.warpAffine(img, mat, (w, h),
                                 flags=cv2.INTER_CUBIC,
                                 borderValue=0.)
        return inv_img

    iou_list = []
    num_correct_grasps = 0
    num_total_grasps = 0
    model.eval()
    time.sleep(2)

    num_grasps = [1, 5]
    num_correct_grasps = [0, 0]
    num_total_grasps = [0, 0]

    tbar = tqdm(test_loader, desc='Inference:', ncols=100)
    for cnt, data in enumerate(tbar):

        # data
        image = data["img"]
        imgdet = data["imgdet"]
        imgdepth = data["imgdepth"]
        mask_all = data["mask_all"]
        text = data["word_vec"]
        ins_mask = data["mask"]
        grasp_qua_mask = data["grasp_masks"]["qua"]
        grasp_sin_mask = data["grasp_masks"]["sin"]
        grasp_cos_mask = data["grasp_masks"]["cos"]
        grasp_wid_mask = data["grasp_masks"]["wid"]
        inverse_matrix = data["inverse"]
        ori_sizes = data["ori_size"]
        grasp_targets = data["grasps"]
        sentences = data["sentence"]
        img_paths = data["img_path"]
        objID = data["objID"]
        image = image.cuda(non_blocking=True)
        imgdet = imgdet.cuda(non_blocking=True)
        imgdepth = imgdepth.cuda(non_blocking=True)
        text = text.cuda(non_blocking=True)
        ins_mask = ins_mask.cuda(non_blocking=True).unsqueeze(1)
        mask_all = mask_all.cuda(non_blocking=True).unsqueeze(1)
        grasp_qua_mask = grasp_qua_mask.cuda(non_blocking=True).unsqueeze(1)
        grasp_sin_mask = grasp_sin_mask.cuda(non_blocking=True).unsqueeze(1)
        grasp_cos_mask = grasp_cos_mask.cuda(non_blocking=True).unsqueeze(1)
        grasp_wid_mask = grasp_wid_mask.cuda(non_blocking=True).unsqueeze(1)

        # inference & get predictions from model
        pred, target = model(image, imgdet, mask_all, imgdepth, text, ins_mask, grasp_qua_mask, grasp_sin_mask, grasp_cos_mask, grasp_wid_mask, objID)

        # predictions
        ins_mask_preds = pred[0]
        grasp_qua_mask_preds = pred[1]
        grasp_sin_mask_preds = pred[2]
        grasp_cos_mask_preds = pred[3]
        grasp_wid_mask_preds = pred[4]

        # targets
        ins_mask_targets = target[0]
        grasp_qua_mask_targets = target[1]
        grasp_sin_mask_targets = target[2]
        grasp_cos_mask_targets = target[3]
        grasp_wid_mask_targets = target[4]

        # Interpolate the predicted ins mask to the same size of input image
        ins_mask_preds = torch.sigmoid(ins_mask_preds)
        grasp_qua_mask_preds = torch.sigmoid(grasp_qua_mask_preds)
        grasp_wid_mask_preds = torch.sigmoid(grasp_wid_mask_preds)

        if ins_mask_preds.shape[-2:] != image.shape[-2:]:
            ins_mask_preds = F.interpolate(ins_mask_preds,
                                           size=image.shape[-2:],
                                           mode='bicubic',
                                           align_corners=True).squeeze(1)

            grasp_qua_mask_preds = F.interpolate(grasp_qua_mask_preds,
                                                 size=image.shape[-2:],
                                                 mode='bicubic',
                                                 align_corners=True).squeeze(1)

            grasp_sin_mask_preds = F.interpolate(grasp_sin_mask_preds,
                                                 size=image.shape[-2:],
                                                 mode='bicubic',
                                                 align_corners=True).squeeze(1)

            grasp_cos_mask_preds = F.interpolate(grasp_cos_mask_preds,
                                                 size=image.shape[-2:],
                                                 mode='bicubic',
                                                 align_corners=True).squeeze(1)

            grasp_wid_mask_preds = F.interpolate(grasp_wid_mask_preds,
                                                 size=image.shape[-2:],
                                                 mode='bicubic',
                                                 align_corners=True).squeeze(1)

        # iterate over the whole batch
        for idx in range(ins_mask_preds.shape[0]):
            inv_mat = inverse_matrix[idx]
            ori_size = ori_sizes[idx]
            h, w = ori_size
            sent = sentences[idx]
            img_path = img_paths[idx]

            ins_mask_pred = ins_mask_preds[idx].cpu().numpy()
            grasp_qua_mask_pred = grasp_qua_mask_preds[idx].squeeze().cpu().numpy()
            grasp_sin_mask_pred = grasp_sin_mask_preds[idx].squeeze().cpu().numpy()
            grasp_cos_mask_pred = grasp_cos_mask_preds[idx].squeeze().cpu().numpy()
            grasp_wid_mask_pred = grasp_wid_mask_preds[idx].squeeze().cpu().numpy()

            ins_mask_target = ins_mask_targets[idx].squeeze().cpu().numpy()
            grasp_target = grasp_targets[idx]
            grasp_qua_mask_target = grasp_qua_mask_targets[idx].squeeze().cpu().numpy()
            grasp_sin_mask_target = grasp_sin_mask_targets[idx].squeeze().cpu().numpy()
            grasp_cos_mask_target = grasp_cos_mask_targets[idx].squeeze().cpu().numpy()
            grasp_wid_mask_target = grasp_wid_mask_targets[idx].squeeze().cpu().numpy()

            # Inverse to original size
            ins_mask_pred = inverse(ins_mask_pred, inv_mat, w, h)
            ins_mask_pred = (ins_mask_pred > 0.35)
            grasp_qua_mask_pred = inverse(grasp_qua_mask_pred, inv_mat, w, h)
            grasp_sin_mask_pred = inverse(grasp_sin_mask_pred, inv_mat, w, h)
            grasp_cos_mask_pred = inverse(grasp_cos_mask_pred, inv_mat, w, h)
            grasp_wid_mask_pred = inverse(grasp_wid_mask_pred, inv_mat, w, h)

            ins_mask_target = inverse(ins_mask_target, inv_mat, w, h)
            grasp_qua_mask_target = inverse(grasp_qua_mask_target, inv_mat, w, h)
            grasp_sin_mask_target = inverse(grasp_sin_mask_target, inv_mat, w, h)
            grasp_cos_mask_target = inverse(grasp_cos_mask_target, inv_mat, w, h)
            grasp_wid_mask_target = inverse(grasp_wid_mask_target, inv_mat, w, h)

            # Calculate IoU between predicted instance mask and gt
            inter = np.logical_and(ins_mask_pred, ins_mask_target)
            union = np.logical_or(ins_mask_pred, ins_mask_target)

            iou = np.sum(inter) / (np.sum(union) + 1e-6)
            iou_list.append(iou)

            # Calculate grasp configurations
            for i in range(len(num_grasps)):
                num_g = num_grasps[i]
                grasp_preds, grasp_ang_mask_pred = detect_grasps(grasp_qua_mask_pred, grasp_sin_mask_pred,
                                                                 grasp_cos_mask_pred, grasp_wid_mask_pred, num_g)

                j_index = calculate_jacquard_index(grasp_preds, grasp_target)

                num_correct_grasps[i] += j_index
                num_total_grasps[i] += 1

                # Visualization
                if args.visualize:
                    img_bgr = cv2.imread(img_path)
                    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    visualization(img, ins_mask_pred, (grasp_qua_mask_pred, grasp_ang_mask_pred, grasp_wid_mask_pred),
                                  grasp_preds, sent, save_path=os.path.join("./results", args.exp_name,
                                                                            f"results_{cnt}_{num_g}_grasps.png"))

    J_index = [0, 0]
    for i in range(len(num_grasps)):
        J_index[i] = num_correct_grasps[i] / num_total_grasps[i]

    iou_list = np.stack(iou_list)
    iou_list = torch.from_numpy(iou_list).to(image.device)
    # print(iou_list)
    # iou_list = concat_all_gather(iou_list)
    prec_list = []
    for thres in torch.arange(0.5, 1.0, 0.1):
        tmp = (iou_list > thres).float().mean()
        prec_list.append(tmp)
    iou = iou_list.mean()
    prec = {}
    for i, thres in enumerate(range(5, 10)):
        key = 'Pr@{}'.format(thres * 10)
        value = prec_list[i].item()
        prec[key] = value
    logger.info('IoU={:.2f}'.format(100. * iou.item()))
    for k, v in prec.items():
        logger.info('{}: {:.2f}.'.format(k, 100. * v))
    logger.info("J@1: {:.2f}, J@5: {:.2f}".format(100. * J_index[0], 100. * J_index[1]))

    return iou.item(), prec, J_index



def build_crog(args):
    model = CROG_TWIN(args)
    backbone = []
    head = []
    for k, v in model.named_parameters():
        if k.startswith('backbone') and 'positional_embedding' not in k:
            backbone.append(v)
        else:
            head.append(v)
    logger.info('Backbone with decay={}, Head={}'.format(len(backbone), len(head)))
    param_list = [{
        'params': backbone,
        'initial_lr': args.lr_multi * args.base_lr
    }, {
        'params': head,
        'initial_lr': args.base_lr
    }]
    return model, param_list
def get_parser():
    parser = argparse.ArgumentParser(
        description='Pytorch Referring Expression Segmentation')
    parser.add_argument('--config',
                        default='/home/lbycdy/work/VGCGM/config/OCID-VLG/20241223modelV009depthtwinmaskgrasp0.5catword.yaml',
                        type=str,
                        help='config file')
    parser.add_argument('--opts',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='override some settings in the config.')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    cfg.save_dir = os.path.splitext(os.path.split(args.config)[-1])[0]
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


@logger.catch
def main():
    args = get_parser()
    args.output_dir = os.path.join(args.output_folder, args.save_dir)

    if args.visualize:
        args.vis_dir = os.path.join(args.output_dir, "vis")
        os.makedirs(args.vis_dir, exist_ok=True)

    # logger
    setup_logger(args.output_dir,
                 distributed_rank=0,
                 filename="test.log",
                 mode="a")
    logger.info(args)

    # build dataset & dataloader
    test_data = OCIDVLGDataset(root_dir=args.root_path,
                            input_size=args.input_size,
                            word_length=args.word_len,
                            split='test',
                            version=args.version)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=4,
                                              shuffle=False,
                                              num_workers=4,
                                              pin_memory=True,
                                              collate_fn=OCIDVLGDataset.collate_fn)

    # build model
    model, param_list = build_crog(args)
    model = torch.nn.DataParallel(model).cuda()
    logger.info(model)
    
    save_path = os.path.join("./results", args.exp_name)
    os.makedirs(save_path, exist_ok=True)

    if os.path.isfile(args.resume):
        logger.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        logger.info("=> loaded checkpoint '{}'".format(args.resume))
    else:
        raise ValueError(
            "=> resume failed! no checkpoint found at '{}'. Please check args.resume again!"
            .format(args.model_dir))

    # inference
    inference_with_grasp(test_loader, model, args)


if __name__ == '__main__':
    main()
