# -*- coding: utf-8 -*

from copy import deepcopy

import numpy as np
from loguru import logger

import torch
import torch.nn as nn

from videoanalyst.pipeline.pipeline_base import TRACK_PIPELINES, PipelineBase
from videoanalyst.pipeline.utils import (cxywh2xywh, get_crop,
                                         get_subwindow_tracking,
                                         imarray_to_tensor, tensor_to_numpy,
                                         xywh2cxywh, xyxy2cxywh)
import torch.nn.functional as F
import cv2
from videoanalyst.online_update import dcf
from videoanalyst.online_update.tensorlist import TensorList
from PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D
from pytracking.libs.optimization import GaussNewtonCG, ConjugateGradient, GradientDescentL2
from videoanalyst.online_update.optim import CtrConvProblem, RegConvProblem, ConvProblem, FactorizedConvProblem

torch.manual_seed(123456) # cpu
torch.cuda.manual_seed(123456) #gpu
#np.random.seed(123456) #numpy

torch.backends.cudnn.deterministic = False #result in different result

import math

# adjust search region
SR_Enable = True
#update setting
Classification_update = False
Centerness_update = False
Regression_update = False
Template_update = False

dataset = 'VOT' #'VOT' or 'OTB'

# ============================== Tracker definition ============================== #
@TRACK_PIPELINES.register
class SiamFCppTracker(PipelineBase):
    r"""
    Basic SiamFC++ tracker

    Hyper-parameters
    ----------------
        total_stride: int
            stride in backbone
        context_amount: float
            factor controlling the image patch cropping range. Set to 0.5 by convention.
        test_lr: float
            factor controlling target size updating speed
        penalty_k: float
            factor controlling the penalization on target size (scale/ratio) change
        window_influence: float
            factor controlling spatial windowing on scores
        windowing: str
            windowing type. Currently support: "cosine"
        z_size: int
            template image size
        x_size: int
            search image size
        num_conv3x3: int
            number of conv3x3 tiled in head
        min_w: float
            minimum width
        min_h: float
            minimum height
        phase_init: str
            phase name for template feature extraction
        phase_track: str
            phase name for target search
        corr_fea_output: bool
            whether output corr feature

    Hyper-parameters (to be calculated at runtime)
    ----------------------------------------------
    score_size: int
        final feature map
    score_offset: int
        final feature map
    """
    default_hyper_params = dict(
        total_stride=8,
        score_size=17,
        score_offset=87,
        context_amount=0.5,
        test_lr=0.52,
        penalty_k=0.04,
        window_influence=0.21,
        windowing="cosine",
        z_size=127,
        x_size=303,
        num_conv3x3=3,
        min_w=10,
        min_h=10,
        phase_init="feature",
        phase_track="track",
        corr_fea_output=False,
    )

    def __init__(self, *args, **kwargs):
        super(SiamFCppTracker, self).__init__(*args, **kwargs)
        self.update_params()

        # set underlying model to device
        self.device = torch.device("cpu")
        self.debug = False
        self.set_model(self._model)
        self.SR_max_size = 407
        self.threshold = 63
        self.ctr_loss = nn.BCELoss(reduction='sum')

    def set_model(self, model):
        """model to be set to pipeline. change device & turn it into eval mode
        
        Parameters
        ----------
        model : ModuleBase
            model to be set to pipeline
        """
        self._model = model.to(self.device)
        self._model.eval()

    def set_device(self, device):
        self.device = device
        self._model = self._model.to(device)

    def update_params(self):
        hps = self._hyper_params
        hps['score_size'] = (
            hps['x_size'] -
            hps['z_size']) // hps['total_stride'] + 1 - hps['num_conv3x3'] * 2
        hps['score_offset'] = (
            hps['x_size'] - 1 -
            (hps['score_size'] - 1) * hps['total_stride']) // 2
        self._hyper_params = hps

    def feature(self, im: np.array, target_pos, target_sz, avg_chans=None):
        """Extract feature

        Parameters
        ----------
        im : np.array
            initial frame
        target_pos : 
            target position (x, y)
        target_sz : [type]
            target size (w, h)
        avg_chans : [type], optional
            channel mean values, (B, G, R), by default None
        
        Returns
        -------
        [type]
            [description]
        """
        if avg_chans is None:
            avg_chans = np.mean(im, axis=(0, 1))

        z_size = self._hyper_params['z_size']
        context_amount = self._hyper_params['context_amount']

        im_z_crop, z_scale = get_crop(
            im,
            target_pos,
            target_sz,
            z_size,
            avg_chans=avg_chans,
            context_amount=context_amount,
            func_get_subwindow=get_subwindow_tracking,
        )
        self.z_scale = z_scale
        self.z_size = z_size
        wc = target_sz[0] + context_amount * sum(target_sz)
        hc = target_sz[1] + context_amount * sum(target_sz)

        phase = self._hyper_params['phase_init']
        with torch.no_grad():
            features = self._model(imarray_to_tensor(im_z_crop).to(self.device),
                                    phase=phase,
                                    new_x_size=self._hyper_params['x_size'])
        if Classification_update:
            s_crop = np.sqrt(wc * hc)
            precise_targetpos_x = int((max(s_crop / 2 - target_sz[0], 0) / s_crop) * z_size)
            precise_targetpos_y = int((max(s_crop / 2 - target_sz[1], 0) / s_crop) * z_size)
            precise_target_w = int((min(s_crop, target_sz[0] * 2) / s_crop) * z_size)
            precise_target_h = int((min(s_crop, target_sz[1] * 2) / s_crop) * z_size)
            phase = 'precisefeature'
            #with torch.no_grad():
            rawfeatures = self._model(imarray_to_tensor(im_z_crop).to(self.device),
                                     phase=phase, new_x_size=self._hyper_params['x_size'])
            spatial_scale = 5 / 127
            rois = torch.tensor([0, precise_targetpos_x, precise_targetpos_y, precise_targetpos_x + precise_target_w,
                                 precise_targetpos_y + precise_target_h]).float().cuda()
            cls_pool = PrRoIPool2D(1, 1, spatial_scale)
            roi_features = cls_pool(rawfeatures, rois)
            roi_features = roi_features[0].unsqueeze(0)
            self.cls_filter = TensorList([roi_features])
        else:
            self.cls_filter = None
        '''
        if Centerness_update:
            self.ctr_filter = TensorList([self._model.head.ctr_score_p5.conv.weight])
        else:
            self.ctr_filter = None
        
            self.ctr_filter = TensorList([roi_features])
        else:
            self.ctr_filter = None
        '''
        self.ctr_filter = None

        if Regression_update:
            reg_pool = PrRoIPool2D(3, 3, spatial_scale)
            reg_roi_features = reg_pool(rawfeatures, rois)
            reg_roi_features = reg_roi_features[0].unsqueeze(0)
            reg_filter = torch.cat([reg_roi_features, reg_roi_features, reg_roi_features, reg_roi_features], dim=0)
            self.reg_filter = TensorList([reg_filter])  ## regression update
            self.reg_filter_l = TensorList([reg_roi_features])
            self.reg_filter_t = TensorList([reg_roi_features])
            self.reg_filter_r = TensorList([reg_roi_features])
            self.reg_filter_b = TensorList([reg_roi_features])
        else:
            self.reg_filter = None

        return features, im_z_crop, avg_chans

    def init(self, im, state, gt_bbox=None):
        r"""Initialize tracker
            Internal target state representation: self._state['state'] = (target_pos, target_sz)
        
        Arguments
        ---------
        im : np.array
            initial frame image
        state
            target state on initial frame (bbox in case of SOT), format: xywh
        """
        rect = state  # bbox in xywh format is given for initialization in case of tracking
        def cal_sz(bbox):
            x_l = min([bbox[0], bbox[2], bbox[4], bbox[6]])
            x_r = max([bbox[0], bbox[2], bbox[4], bbox[6]])
            y_t = min([bbox[1], bbox[3], bbox[5], bbox[7]])
            y_b = max([bbox[1], bbox[3], bbox[5], bbox[7]])
            w = x_r -x_l
            h = y_b - y_t
            return w, h
        box = xywh2cxywh(rect)
        target_pos, target_sz = box[:2], box[2:]
        if dataset == 'VOT':
            self.target_pos = torch.Tensor([target_pos[1], target_pos[0]])
            self.target_sz = cal_sz(gt_bbox)
            self.target_sz = torch.Tensor([self.target_sz[1], self.target_sz[0]])
        else:
            self.target_pos = torch.Tensor([target_pos[1], target_pos[0]])
            self.target_sz = torch.Tensor([target_sz[1], target_sz[0]])

        self._state['im_h'] = im.shape[0]
        self._state['im_w'] = im.shape[1]

        # extract template feature
        features, im_z_crop, avg_chans = self.feature(im, target_pos, target_sz)

        score_size = self._hyper_params['score_size']
        if self._hyper_params['windowing'] == 'cosine':
            window = np.outer(np.hanning(score_size), np.hanning(score_size))
            window = window.reshape(-1)
        elif self._hyper_params['windowing'] == 'uniform':
            window = np.ones((score_size, score_size))
        else:
            window = np.ones((score_size, score_size))

        self._state['z_crop'] = im_z_crop
        self._state['avg_chans'] = avg_chans
        self._state['features'] = features
        self._state['window'] = window
        # self.state['target_pos'] = target_pos
        # self.state['target_sz'] = target_sz
        self._state['state'] = (target_pos, target_sz)
        # atom
        # Set search area
        self.target_scale = 1.0
        # print('self.target_sz', self.target_sz)
        search_area = torch.prod(self.target_sz * 5).item()
        # print('search_area',search_area)
        if search_area > 82944:
            self.target_scale = math.sqrt(search_area / 82944)
        elif search_area < 82944:
            self.target_scale = math.sqrt(search_area / 82944)
        self.base_target_sz = self.target_sz / self.target_scale
        # print(self.target_scale)

        self.img_support_sz = torch.Tensor([303., 303.])
        self.kernel_size = TensorList([[1, 1]])  # atom is (4, 4)
        self.feature_sz = TensorList([torch.Tensor([19, 19]).int()])
        self.memory_size = 30  # for cls
        self.reg_memory_size = 30
        self.ctr_memory_size = 30
        self.init_memory()
        self.init_label_function()
        self.init_optimization()
        self.frame_count = 0
        self.scale_factors = torch.Tensor([1.])
        for name, param in self._model.named_parameters():
            if name.find('ctr_score_p5') != -1:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.ctr_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self._model.parameters()),
                                         lr=0.001,
                                         momentum=0.9,
                                         weight_decay=0.0001)

        #template update
        if Template_update:
          self._state['maxscore'] = 0
          self._state['z_0'] = features
          self._state['z_s'] = features
          self._state['z_0_pos'] = target_pos
          self._state['z_0_sz'] = target_sz
          self._state['z_s_pos'] = target_pos
          self._state['z_s_sz'] = target_sz

    def get_avg_chans(self):
        return self._state['avg_chans']

    def track(self,
              im_x,
              target_pos,
              target_sz,
              features,
              update_state=False,
              **kwargs):
        if 'avg_chans' in kwargs:
            avg_chans = kwargs['avg_chans']
        else:
            avg_chans = self._state['avg_chans']

        z_size = self._hyper_params['z_size']
        x_size = self._hyper_params['x_size']
        context_amount = self._hyper_params['context_amount']
        phase_track = self._hyper_params['phase_track']
        im_x_crop, scale_x = get_crop(
            im_x,
            target_pos,
            target_sz,
            z_size,
            x_size=x_size,
            avg_chans=avg_chans,
            context_amount=context_amount,
            func_get_subwindow=get_subwindow_tracking,
        )
        self._state["scale_x"] = deepcopy(scale_x)

        #cv2.imshow('search', im_x_crop)
        #cv2.waitKey(1)

        #with torch.no_grad():
        score, box, cls, ctr, extra, cls_train_x, reg_train_x, cls2_score, reg_label, reg_pred = self._model(
            imarray_to_tensor(im_x_crop).to(self.device),
            *features,
            phase=phase_track, f=self.frame_count, new_x_size=self._hyper_params['x_size'], cls_weights = self.cls_filter, ctr_weights=self.ctr_filter, reg_weights=self.reg_filter)
        if self._hyper_params["corr_fea_output"]:
            self._state["corr_fea"] = extra["corr_fea"]

        pred_ctr = ctr[0] #for centerness update

        self.feature_sz = TensorList([torch.Tensor([19, 19])])
        box = tensor_to_numpy(box[0])
        score = tensor_to_numpy(score[0])[:, 0]
        cls = tensor_to_numpy(cls[0])
        ctr = tensor_to_numpy(ctr[0])
        box_wh = xyxy2cxywh(box)

        # score post-processing
        best_pscore_id, pscore, penalty = self._postprocess_score(
            score, box_wh, target_sz, scale_x)
        self.best_score_id = best_pscore_id
        # box post-processing
        new_target_pos, new_target_sz = self._postprocess_box(
            best_pscore_id, score, box_wh, target_pos, target_sz, scale_x,
            x_size, penalty)

        if self.debug:
            box = self._cvt_box_crop2frame(box_wh, target_pos, x_size, scale_x)

        #template update
        if Template_update:
          if pscore[best_pscore_id] > 0.5 and pscore[best_pscore_id] > self._state['maxscore']:
              # save to z_s
              self._state['z_s'], im_z_s_crop, avg_chans = self.feature(im_x, new_target_pos, new_target_sz)
              self._state['maxscore'] = pscore[best_pscore_id]
              self._state['z_s_pos'] = new_target_pos
              self._state['z_s_sz'] = new_target_sz
  
          if self.frame_count % 5 == 0 and self._state['maxscore'] != 0:
              ''' calculate from z_s '''
              feat = self._state['z_s']
              with torch.no_grad():
                  score_zs, box_zs, cls_zs, ctr_zs, extra_zs, cls_train_x_zs, reg_train_x_zs, cls2_score_zs, reg_label_zs, reg_pred_zs = self._model(
                      imarray_to_tensor(im_x_crop).to(self.device),
                      *feat,
                      phase=phase_track, f=self.frame_count, new_x_size=self._hyper_params['x_size'],
                      cls_weights=self.cls_filter, ctr_weights=self.ctr_filter, reg_weights=self.reg_filter)
  
              box_zs = tensor_to_numpy(box_zs[0])
              score_zs = tensor_to_numpy(score_zs[0])[:, 0]
              cls_zs = tensor_to_numpy(cls_zs[0])
              ctr_zs = tensor_to_numpy(ctr_zs[0])
              box_wh_zs = xyxy2cxywh(box_zs)
  
              # score post-processing
              best_pscore_id_zs, pscore_zs, penalty_zs = self._postprocess_score(
                  score_zs, box_wh_zs, target_sz, scale_x)
  
              # box post-processing
              new_target_pos_zs, new_target_sz_zs = self._postprocess_box(
                  best_pscore_id_zs, score_zs, box_wh_zs, target_pos, target_sz, scale_x,
                  x_size, penalty_zs)
  
              ''' calculate from z_0 '''
              feat = self._state['z_0']
              with torch.no_grad():
                  score_z0, box_z0, cls_z0, ctr_z0, extra_z0, cls_train_x_z0, reg_train_x_z0, cls2_score_z0, reg_label_z0, reg_pred_z0 = self._model(
                      imarray_to_tensor(im_x_crop).to(self.device),
                      *feat,
                      phase=phase_track, f=self.frame_count, new_x_size=self._hyper_params['x_size'],
                      cls_weights=self.cls_filter, ctr_weights=self.ctr_filter, reg_weights=self.reg_filter)
  
              box_z0 = tensor_to_numpy(box_z0[0])
              score_z0 = tensor_to_numpy(score_z0[0])[:, 0]
              cls_z0 = tensor_to_numpy(cls_z0[0])
              ctr_z0 = tensor_to_numpy(ctr_z0[0])
              box_wh_z0 = xyxy2cxywh(box_z0)
  
              # score post-processing
              best_pscore_id_z0, pscore_z0, penalty_z0 = self._postprocess_score(
                  score_z0, box_wh_z0, target_sz, scale_x)
  
              # box post-processing
              new_target_pos_z0, new_target_sz_z0 = self._postprocess_box(
                  best_pscore_id_z0, score_z0, box_wh_z0, target_pos, target_sz, scale_x,
                  x_size, penalty_z0)
  
              if self.cal_iou(new_target_pos_z0, new_target_sz_z0, new_target_pos_zs, new_target_sz_zs, im_x) >= 0.6:
                  self._state['z_s'], _, _ = self.feature(im_x, self._state['z_s_pos'], self._state['z_s_sz'])
                  self._state['feature'] = self._state['z_s']
                  new_target_pos = new_target_pos_zs
                  new_target_sz = new_target_sz_zs
              else:
                  self._state['feature'] = self._state['z_0']
              self._state['maxscore'] = 0

        # restrict new_target_pos & new_target_sz
        new_target_pos, new_target_sz = self._restrict_box(
            new_target_pos, new_target_sz)

        # record basic mid-level info
        self._state['x_crop'] = im_x_crop
        bbox_pred_in_crop = np.rint(box[best_pscore_id]).astype(np.int)
        self._state['bbox_pred_in_crop'] = bbox_pred_in_crop

        # atom
        self.target_pos = torch.Tensor([new_target_pos[1], new_target_pos[0]])
        self.target_sz = torch.Tensor([new_target_sz[1], new_target_sz[0]])
        new_scale = torch.sqrt(new_target_sz.prod() / self.base_target_sz.prod())
        self.target_scale = new_scale
        # Get sample
        sample_pos = self.target_pos.round()
        sample_scales = self.target_scale * self.scale_factors
        map_size = cls_train_x.size(-1)
        if map_size != 19:
            self.left_bound = (map_size - 19) // 2
            if (map_size - 19) % 2 != 0:
                self.right_bound = map_size - ((map_size - 19) // 2 + 1)
            else:
                self.right_bound = map_size - ((map_size - 19) // 2)
            cls_train_x = TensorList([cls_train_x[0, :, self.left_bound:self.right_bound,
                                      self.left_bound:self.right_bound].view(1, 256, 19, 19)])
            if Regression_update:
                reg_train_x = TensorList(
                    [reg_train_x[0, :, self.left_bound:self.right_bound, self.left_bound:self.right_bound].view(1,
                                                                                                                256,
                                                                                                                19,
                                                                                                                19)])
                reg_label = reg_label[0, :, self.left_bound:self.right_bound,
                            self.left_bound:self.right_bound].view(1, 4, 19, 19)
        
        # ------- UPDATE ------- #
        if Classification_update:
            if torch.max(cls2_score).item() > 0.5 or (self.frame_count - 1) % 10 == 0:
                cls_update_flag = True
            else:
                cls_update_flag = False
            if cls_update_flag:
                # Create label for sample
                cls_train_y = self.get_label_function(sample_pos, sample_scales[0])
                # Update memory
                self.update_memory(cls_train_x, cls_train_y)
                # Train filter
                self.cls_filter = self.filter_optimizer.run(5)

        if Regression_update:  # regression update
            reg_train_y = TensorList([reg_label])
            mask_reg = reg_label.new_full((1, 1, reg_label.size(2), reg_label.size(3)), 0, device='cuda')
            if self.best_score_id < 361:
                mask_reg[:, :, self.best_score_id // 19, self.best_score_id % 19] = 1
            mask_reg = TensorList([mask_reg])
            self.update_memory_reg(reg_train_x, reg_train_y, mask_reg)
            self.reg_filter_l = self.filter_optimizer_l.run(1, self.mask_reg)
            self.reg_filter_t = self.filter_optimizer_t.run(1, self.mask_reg)
            self.reg_filter_r = self.filter_optimizer_r.run(1, self.mask_reg)
            self.reg_filter_b = self.filter_optimizer_b.run(1, self.mask_reg)
            reg_filter = torch.cat(
                [self.reg_filter_l[0], self.reg_filter_t[0], self.reg_filter_r[0], self.reg_filter_b[0]], dim=0)
            self.reg_filter = TensorList([reg_filter])

        if Centerness_update:
            label_box = np.asarray(
                    [[bbox_pred_in_crop[0], bbox_pred_in_crop[1], bbox_pred_in_crop[2], bbox_pred_in_crop[3], 1]])
            mask_ctr, ctr_label, _ = self.get_ctr_reg_label_function(label_box)
            # print('ctr', ctr1_score[self.best_score_id])
            # print('ctr2', ctr2_score[self.best_score_id])
            ctr_dif = abs(torch.from_numpy(ctr_label[self.best_score_id]).to('cuda') - pred_ctr[self.best_score_id])
            if ctr_dif < 0.1 or (self.frame_count - 1) % 10 == 0: #ctr_dif < 0.1 or torch.max(pred_ctr).item()
                ctr_update_flag = True
            else:
                ctr_update_flag = False
            if ctr_update_flag:
                mask_ctr = torch.from_numpy(mask_ctr).to('cuda').reshape(-1)
                pos_inds = torch.nonzero(mask_ctr > 0)
                ctr_label = torch.from_numpy((ctr_label)).to('cuda').reshape(-1)[pos_inds]
                pred_ctr = pred_ctr.to('cuda').reshape(-1)[pos_inds]
                loss_ctr = self.ctr_loss(pred_ctr, ctr_label) / pos_inds.numel()
                self.ctr_optimizer.zero_grad()
                loss_ctr.backward()
                self.ctr_optimizer.step()
                ''' Conjugate descent
                mask_ctr = TensorList(torch.Tensor(mask_ctr).view(19, 19, 1, 1).permute(3, 2, 0, 1).to('cuda'))
                ctr_label = torch.Tensor(ctr_label).view(19, 19, 1, 1).permute(3, 2, 0, 1)
                ctr_train_y = TensorList([ctr_label])
                self.update_memory_ctr(cls_train_x, ctr_train_y, mask_ctr)
                self.ctr_filter = self.filter_optimizer_ctr.run(5, self.mask_ctr)
                '''

        self.frame_count += 1

        # adjust search area
        if SR_Enable:
            wh_past = math.sqrt(target_sz[0] * target_sz[1])
            wh_now = math.sqrt(new_target_sz[0] * new_target_sz[1])
            self.threshold = (wh_past + wh_now) / 2

            d = math.sqrt((new_target_pos[0] - target_pos[0]) ** 2 + (new_target_pos[1] - target_pos[1]) ** 2)
            #print("distance=",d)

            if d > self.threshold:
                temp_size = (d/self.threshold) * 303
                total_stride = self._hyper_params['total_stride']
                self._hyper_params['x_size'] = math.ceil((temp_size - z_size) / total_stride) * total_stride + z_size
                if self._hyper_params['x_size'] > self.SR_max_size:
                    self._hyper_params['x_size'] = self.SR_max_size
                print("self.threshold=", self.threshold, ", self._hyper_params['x_size']=",
                      self._hyper_params['x_size'])
            else:
                self._hyper_params['x_size'] = 303 #default

            #print ("next up")
            self.update_params()
            score_size = self._hyper_params['score_size']
            if self._hyper_params['windowing'] == 'cosine':
                window = np.outer(np.hanning(score_size), np.hanning(score_size))
                window = window.reshape(-1)
            elif self._hyper_params['windowing'] == 'uniform':
                window = np.ones((score_size, score_size))
            else:
                window = np.ones((score_size, score_size))
            self._state['window'] = window

        # record optional mid-level info
        if update_state:
            self._state['score'] = score
            self._state['pscore'] = pscore[best_pscore_id]
            self._state['all_box'] = box
            self._state['cls'] = cls
            self._state['ctr'] = ctr

        return new_target_pos, new_target_sz

    def set_state(self, state):
        self._state["state"] = state

    def get_track_score(self):
        return float(self._state["pscore"])

    def update(self, im, state=None):
        """ Perform tracking on current frame
            Accept provided target state prior on current frame
            e.g. search the target in another video sequence simutanously

        Arguments
        ---------
        im : np.array
            current frame image
        state
            provided target state prior (bbox in case of SOT), format: xywh
        """
        # use prediction on the last frame as target state prior
        if state is None:
            target_pos_prior, target_sz_prior = self._state['state']
        # use provided bbox as target state prior
        else:
            rect = state  # bbox in xywh format is given for initialization in case of tracking
            box = xywh2cxywh(rect).reshape(4)
            target_pos_prior, target_sz_prior = box[:2], box[2:]
        features = self._state['features']

        # forward inference to estimate new state
        target_pos, target_sz = self.track(im,
                                           target_pos_prior,
                                           target_sz_prior,
                                           features,
                                           update_state=True)

        # save underlying state
        # self.state['target_pos'], self.state['target_sz'] = target_pos, target_sz
        self._state['state'] = target_pos, target_sz

        # return rect format
        track_rect = cxywh2xywh(np.concatenate([target_pos, target_sz],
                                               axis=-1))
        if self._hyper_params["corr_fea_output"]:
            return target_pos, target_sz, self._state["corr_fea"]
        return track_rect

    # ======== tracking processes ======== #

    def _postprocess_score(self, score, box_wh, target_sz, scale_x):
        r"""
        Perform SiameseRPN-based tracker's post-processing of score
        :param score: (HW, ), score prediction
        :param box_wh: (HW, 4), cxywh, bbox prediction (format changed)
        :param target_sz: previous state (w & h)
        :param scale_x:
        :return:
            best_pscore_id: index of chosen candidate along axis HW
            pscore: (HW, ), penalized score
            penalty: (HW, ), penalty due to scale/ratio change
        """
        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            sz2 = (w + pad) * (h + pad)
            return np.sqrt(sz2)

        def sz_wh(wh):
            pad = (wh[0] + wh[1]) * 0.5
            sz2 = (wh[0] + pad) * (wh[1] + pad)
            return np.sqrt(sz2)

        # size penalty
        penalty_k = self._hyper_params['penalty_k']
        target_sz_in_crop = target_sz * scale_x
        s_c = change(
            sz(box_wh[:, 2], box_wh[:, 3]) /
            (sz_wh(target_sz_in_crop)))  # scale penalty
        r_c = change((target_sz_in_crop[0] / target_sz_in_crop[1]) /
                     (box_wh[:, 2] / box_wh[:, 3]))  # ratio penalty
        penalty = np.exp(-(r_c * s_c - 1) * penalty_k)
        pscore = penalty * score

        # ipdb.set_trace()
        # cos window (motion model)
        window_influence = self._hyper_params['window_influence']
        pscore = pscore * (
            1 - window_influence) + self._state['window'] * window_influence
        best_pscore_id = np.argmax(pscore)

        return best_pscore_id, pscore, penalty

    def _postprocess_box(self, best_pscore_id, score, box_wh, target_pos,
                         target_sz, scale_x, x_size, penalty):
        r"""
        Perform SiameseRPN-based tracker's post-processing of box
        :param score: (HW, ), score prediction
        :param box_wh: (HW, 4), cxywh, bbox prediction (format changed)
        :param target_pos: (2, ) previous position (x & y)
        :param target_sz: (2, ) previous state (w & h)
        :param scale_x: scale of cropped patch of current frame
        :param x_size: size of cropped patch
        :param penalty: scale/ratio change penalty calculated during score post-processing
        :return:
            new_target_pos: (2, ), new target position
            new_target_sz: (2, ), new target size
        """
        pred_in_crop = box_wh[best_pscore_id, :] / np.float32(scale_x)
        # about np.float32(scale_x)
        # attention!, this casting is done implicitly
        # which can influence final EAO heavily given a model & a set of hyper-parameters

        # box post-postprocessing
        test_lr = self._hyper_params['test_lr']
        lr = penalty[best_pscore_id] * score[best_pscore_id] * test_lr
        res_x = pred_in_crop[0] + target_pos[0] - (x_size // 2) / scale_x
        res_y = pred_in_crop[1] + target_pos[1] - (x_size // 2) / scale_x
        res_w = target_sz[0] * (1 - lr) + pred_in_crop[2] * lr
        res_h = target_sz[1] * (1 - lr) + pred_in_crop[3] * lr

        new_target_pos = np.array([res_x, res_y])
        new_target_sz = np.array([res_w, res_h])

        return new_target_pos, new_target_sz

    def _restrict_box(self, target_pos, target_sz):
        r"""
        Restrict target position & size
        :param target_pos: (2, ), target position
        :param target_sz: (2, ), target size
        :return:
            target_pos, target_sz
        """
        target_pos[0] = max(0, min(self._state['im_w'], target_pos[0]))
        target_pos[1] = max(0, min(self._state['im_h'], target_pos[1]))
        target_sz[0] = max(self._hyper_params['min_w'],
                           min(self._state['im_w'], target_sz[0]))
        target_sz[1] = max(self._hyper_params['min_h'],
                           min(self._state['im_h'], target_sz[1]))

        return target_pos, target_sz

    def _cvt_box_crop2frame(self, box_in_crop, target_pos, scale_x, x_size):
        r"""
        Convert box from cropped patch to original frame
        :param box_in_crop: (4, ), cxywh, box in cropped patch
        :param target_pos: target position
        :param scale_x: scale of cropped patch
        :param x_size: size of cropped patch
        :return:
            box_in_frame: (4, ), cxywh, box in original frame
        """
        x = (box_in_crop[..., 0]) / scale_x + target_pos[0] - (x_size //
                                                               2) / scale_x
        y = (box_in_crop[..., 1]) / scale_x + target_pos[1] - (x_size //
                                                               2) / scale_x
        w = box_in_crop[..., 2] / scale_x
        h = box_in_crop[..., 3] / scale_x
        box_in_frame = np.stack([x, y, w, h], axis=-1)

        return box_in_frame

    def init_projection_matrix(self, x):
        self.compressed_dim = TensorList([256])
        self.projection_matrix = TensorList(
            [None if cdim is None else ex.new_zeros(cdim, ex.shape[1], 1, 1).normal_(0, 1 / math.sqrt(ex.shape[1]))
             for ex, cdim in
             zip(x, self.compressed_dim)])

    def init_label_function(self):
        # Allocate label function
        self.y = TensorList([torch.Tensor(
            np.zeros((self.memory_size, 1, self.feature_sz[0][0], self.feature_sz[0][1]))).to('cuda')])

        if Centerness_update:
            self.y_ctr = TensorList([torch.Tensor(
                np.zeros((self.ctr_memory_size, 1, self.feature_sz[0][0], self.feature_sz[0][1]))).to('cuda')])

        if Regression_update:
            self.y_l = TensorList(
                [torch.Tensor(np.zeros((self.reg_memory_size, 1, self.feature_sz[0][0], self.feature_sz[0][1]))).to(
                    'cuda')])
            self.y_t = TensorList(
                [torch.Tensor(np.zeros((self.reg_memory_size, 1, self.feature_sz[0][0], self.feature_sz[0][1]))).to(
                    'cuda')])
            self.y_r = TensorList(
                [torch.Tensor(np.zeros((self.reg_memory_size, 1, self.feature_sz[0][0], self.feature_sz[0][1]))).to(
                    'cuda')])
            self.y_b = TensorList(
                [torch.Tensor(np.zeros((self.reg_memory_size, 1, self.feature_sz[0][0], self.feature_sz[0][1]))).to(
                    'cuda')])

        # print('self.y', type(self.y), self.y.size())

        # Output sigma factor
        self.sigma = TensorList([torch.Tensor([1.8, 1.8])])  # 0.9
        # print('self.sigma', type(self.sigma), self.sigma.size())

    def init_memory(self):
        # Initialize first-frame training samples
        self.num_init_samples = TensorList([0])
        # print('self.num_init_samples', type(self.num_init_samples), self.num_init_samples)
        self.init_sample_weights = TensorList([0])
        # print('self.init_sample_weights', self.init_sample_weights.size(), type(self.init_sample_weights))
        self.init_training_samples = TensorList()
        # print('self.init_training_samples', self.init_training_samples.size(), type(self.init_training_samples))

        # Sample counters and weights
        self.num_stored_samples = self.num_init_samples.copy()
        # print('self.num_stored_samples', self.num_stored_samples, type(self.num_stored_samples))
        self.previous_replace_ind = [None] * len(self.num_stored_samples)
        # print('self.previous_replace_ind', self.previous_replace_ind, type(self.previous_replace_ind))
        self.sample_weights = TensorList([torch.Tensor(np.zeros(self.memory_size)).to('cuda')])
        # print('self.sample_weights', self.sample_weights.size(), type(self.sample_weights))

        # Initialize memory
        self.training_samples = TensorList(
            [torch.Tensor(np.zeros((self.memory_size, 256, self.feature_sz[0][0], self.feature_sz[0][1]))).to(
                'cuda')])
        # print('self.training_samples', self.training_samples.size(), type(self.training_samples))
        if Centerness_update:
            self.num_init_samples_ctr = TensorList([0])
            self.num_stored_samples_ctr = self.num_init_samples_ctr.copy()
            self.previous_replace_ind_ctr = [None] * len(self.num_stored_samples_ctr)
            self.sample_weights_ctr = TensorList([torch.Tensor(np.zeros(self.ctr_memory_size)).to('cuda')])
            self.training_samples_ctr = TensorList(
                [torch.Tensor(
                    np.zeros((self.ctr_memory_size, 256, self.feature_sz[0][0], self.feature_sz[0][1]))).to(
                    'cuda')])
            self.mask_ctr = TensorList([torch.Tensor(
                np.zeros((self.ctr_memory_size, 1, self.feature_sz[0][0], self.feature_sz[0][1]))).to('cuda')])

        if Regression_update:
            self.num_init_samples_reg = TensorList([0])
            self.num_stored_samples_reg = self.num_init_samples_reg.copy()
            self.previous_replace_ind_reg = [None] * len(self.num_stored_samples_reg)
            self.sample_weights_reg = TensorList([torch.Tensor(np.zeros(self.reg_memory_size)).to('cuda')])
            self.training_samples_reg = TensorList(
                [torch.Tensor(
                    np.zeros((self.reg_memory_size, 256, self.feature_sz[0][0], self.feature_sz[0][1]))).to(
                    'cuda')])
            self.mask_reg = TensorList([torch.Tensor(
                np.zeros((self.reg_memory_size, 1, self.feature_sz[0][0], self.feature_sz[0][1]))).to('cuda')])

    def init_optimization(self):
        # Initialize filter
        self.filter_reg = TensorList([0.1])
        # print('self.filter_reg', type(self.filter_reg), self.filter_reg)
        self.projection_reg = TensorList([0.0001])
        # print('self.projection_reg', type(self.projection_reg), self.projection_reg)
        # print('self.init_sample_weights', type(self.init_sample_weights), self.init_sample_weights, self.init_sample_weights.size())
        self.projection_activation = lambda x: x
        self.response_activation = lambda x: F.elu(F.leaky_relu(x, 1 / 0.05), 0.05)

        # Initialize optimizer
        self.conv_problem = ConvProblem(self.training_samples, self.y, self.filter_reg, self.sample_weights,
                                        self.response_activation)

        self.filter_optimizer = ConjugateGradient(self.conv_problem, self.cls_filter, fletcher_reeves=False,
                                                  direction_forget_factor=0, debug=False,
                                                  plotting=False, visdom=None)
        # ctr update
        if Centerness_update:
            self.conv_problem_ctr = CtrConvProblem(self.training_samples_ctr, self.y_ctr, self.filter_reg,
                                                   self.sample_weights_ctr, self.response_activation)

            self.filter_optimizer_ctr = ConjugateGradient(self.conv_problem_ctr, self.ctr_filter,
                                                          fletcher_reeves=False,
                                                          direction_forget_factor=0, debug=False,
                                                          plotting=False, visdom=None)
        # reg update
        if Regression_update:
            self.conv_problem_l = RegConvProblem(self.training_samples_reg, self.y_l, self.filter_reg,
                                                 self.sample_weights_reg,
                                                 self.response_activation)

            self.filter_optimizer_l = ConjugateGradient(self.conv_problem_l, self.reg_filter_l,
                                                        fletcher_reeves=False,
                                                        direction_forget_factor=0, debug=False,
                                                        plotting=False, visdom=None)
            self.conv_problem_t = RegConvProblem(self.training_samples_reg, self.y_t, self.filter_reg,
                                                 self.sample_weights_reg,
                                                 self.response_activation)

            self.filter_optimizer_t = ConjugateGradient(self.conv_problem_t, self.reg_filter_t,
                                                        fletcher_reeves=False,
                                                        direction_forget_factor=0, debug=False,
                                                        plotting=False, visdom=None)
            self.conv_problem_r = RegConvProblem(self.training_samples_reg, self.y_r, self.filter_reg,
                                                 self.sample_weights_reg,
                                                 self.response_activation)

            self.filter_optimizer_r = ConjugateGradient(self.conv_problem_r, self.reg_filter_r,
                                                        fletcher_reeves=False,
                                                        direction_forget_factor=0, debug=False,
                                                        plotting=False, visdom=None)
            self.conv_problem_b = RegConvProblem(self.training_samples_reg, self.y_b, self.filter_reg,
                                                 self.sample_weights_reg,
                                                 self.response_activation)

            self.filter_optimizer_b = ConjugateGradient(self.conv_problem_b, self.reg_filter_b,
                                                        fletcher_reeves=False,
                                                        direction_forget_factor=0, debug=False,
                                                        plotting=False, visdom=None)

    def get_label_function(self, sample_pos, sample_scale):
        # Generate label function
        train_y = TensorList()
        target_center_norm = (self.target_pos - sample_pos) / (sample_scale * self.img_support_sz)
        for sig, sz, ksz in zip(self.sigma, self.feature_sz, self.kernel_size):
            center = sz * target_center_norm + 0.5 * torch.Tensor([(ksz[0] + 1) % 2, (ksz[1] + 1) % 2])
            train_y.append(dcf.label_function_spatial(sz, sig, center))
        return train_y

    def update_sample_weights(self, sample_weights, previous_replace_ind, num_stored_samples, num_init_samples,
                              learning_rate=None):
        # Update weights and get index to replace in memory
        replace_ind = []
        for sw, prev_ind, num_samp, num_init in zip(sample_weights, previous_replace_ind, num_stored_samples,
                                                    num_init_samples):
            lr = learning_rate
            if lr is None:
                lr = 0.01

            init_samp_weight = None  # 0.25
            s_ind = 0 if init_samp_weight is None else num_init

            if num_samp == 0 or lr == 1:
                sw[:] = 0
                sw[0] = 1
                r_ind = 0
            else:
                # Get index to replace
                _, r_ind = torch.min(sw[s_ind:], 0)
                r_ind = r_ind.item() + s_ind

                # Update weights
                if prev_ind is None:
                    sw /= 1 - lr
                    sw[r_ind] = lr
                else:
                    sw[r_ind] = sw[prev_ind] / (1 - lr)

            sw /= sw.sum()
            if init_samp_weight is not None and sw[:num_init].sum() < init_samp_weight:
                sw /= init_samp_weight + sw[num_init:].sum()
                sw[:num_init] = init_samp_weight / num_init

            replace_ind.append(r_ind)

        return replace_ind

    def update_memory(self, sample_x: TensorList, sample_y: TensorList, learning_rate=None):
        replace_ind = self.update_sample_weights(self.sample_weights, self.previous_replace_ind,
                                                 self.num_stored_samples, self.num_init_samples, learning_rate)
        self.previous_replace_ind = replace_ind
        for train_samp, x, ind in zip(self.training_samples, sample_x, replace_ind):
            train_samp[ind:ind + 1, ...] = x
        for y_memory, y, ind in zip(self.y, sample_y, replace_ind):
            y_memory[ind:ind + 1, ...] = y
        self.num_stored_samples += 1

    def update_memory_ctr(self, sample_x: TensorList, sample_y: TensorList, mask_y: TensorList, learning_rate=None):
        replace_ind = self.update_sample_weights(self.sample_weights_ctr, self.previous_replace_ind_ctr,
                                                 self.num_stored_samples_ctr, self.num_init_samples_ctr,
                                                 learning_rate)
        self.previous_replace_ind_ctr = replace_ind
        for train_samp, x, ind in zip(self.training_samples_ctr, sample_x, replace_ind):
            train_samp[ind:ind + 1, ...] = x
        for y_memory, y, ind in zip(self.y_ctr, sample_y, replace_ind):
            y_memory[ind:ind + 1, ...] = y
        for mask_memory, mask, ind in zip(self.mask_ctr, mask_y, replace_ind):
            mask_memory[ind:ind + 1, ...] = mask
        self.num_stored_samples_ctr += 1

    def update_memory_reg(self, sample_x: TensorList, sample_y: TensorList, mask_y: TensorList, learning_rate=None):
        replace_ind = self.update_sample_weights(self.sample_weights_reg, self.previous_replace_ind_reg,
                                                 self.num_stored_samples_reg, self.num_init_samples_reg,
                                                 learning_rate)
        self.previous_replace_ind_reg = replace_ind
        for train_samp, x, ind in zip(self.training_samples_reg, sample_x, replace_ind):
            train_samp[ind:ind + 1, ...] = x
        for y_memory, y, ind in zip(self.y_l, sample_y, replace_ind):
            y_memory[ind:ind + 1, ...] = y[0, 0, ...].view(1, 1, 19, 19)
        for y_memory, y, ind in zip(self.y_t, sample_y, replace_ind):
            y_memory[ind:ind + 1, ...] = y[0, 1, ...].view(1, 1, 19, 19)
        for y_memory, y, ind in zip(self.y_r, sample_y, replace_ind):
            y_memory[ind:ind + 1, ...] = y[0, 2, ...].view(1, 1, 19, 19)
        for y_memory, y, ind in zip(self.y_b, sample_y, replace_ind):
            y_memory[ind:ind + 1, ...] = y[0, 3, ...].view(1, 1, 19, 19)
        for mask_memory, mask, ind in zip(self.mask_reg, mask_y, replace_ind):
            mask_memory[ind:ind + 1, ...] = mask
        self.num_stored_samples_reg += 1

    def get_ctr_reg_label_function(self, gt_boxes):
        x_size = self._hyper_params['x_size']
        score_size = self._hyper_params['score_size']
        total_stride = self._hyper_params['total_stride']
        score_offset = self._hyper_params['score_offset']
        eps = 1e-5
        raw_height, raw_width = x_size, x_size

        if gt_boxes.shape[1] == 4:
            gt_boxes = np.concatenate(
                [gt_boxes, np.ones(
                    (gt_boxes.shape[0], 1))], axis=1)  # boxes_cnt x 5
        # l, t, r, b
        gt_boxes = np.concatenate([np.zeros((1, 5)), gt_boxes])  # boxes_cnt x 5
        gt_boxes_area = (np.abs(
            (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])))
        gt_boxes = gt_boxes[np.argsort(gt_boxes_area)]
        boxes_cnt = len(gt_boxes)

        shift_x = np.arange(0, raw_width).reshape(-1, 1)
        shift_y = np.arange(0, raw_height).reshape(-1, 1)
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        off_l = (shift_x[:, :, np.newaxis, np.newaxis] -
                 gt_boxes[np.newaxis, np.newaxis, :, 0, np.newaxis])
        off_t = (shift_y[:, :, np.newaxis, np.newaxis] -
                 gt_boxes[np.newaxis, np.newaxis, :, 1, np.newaxis])
        off_r = -(shift_x[:, :, np.newaxis, np.newaxis] -
                  gt_boxes[np.newaxis, np.newaxis, :, 2, np.newaxis])
        off_b = -(shift_y[:, :, np.newaxis, np.newaxis] -
                  gt_boxes[np.newaxis, np.newaxis, :, 3, np.newaxis])

        center = ((np.minimum(off_l, off_r) * np.minimum(off_t, off_b)) /
                  (np.maximum(off_l, off_r) * np.maximum(off_t, off_b) + eps))
        center = np.squeeze(np.sqrt(np.abs(center)))
        center[:, :, 0] = 0

        offset = np.concatenate([off_l, off_t, off_r, off_b],
                                axis=3)  # h x w x boxes_cnt * 4
        cls = gt_boxes[:, 4]

        cls_res_list = []
        ctr_res_list = []
        gt_boxes_res_list = []

        fm_height, fm_width = score_size, score_size

        fm_size_list = []
        fm_strides = [total_stride]
        fm_offsets = [score_offset]
        for fm_i in range(len(fm_strides)):
            fm_size_list.append([fm_height, fm_width])
            fm_height = int(np.ceil(fm_height / 2))
            fm_width = int(np.ceil(fm_width / 2))

        fm_size_list = fm_size_list[::-1]
        for fm_i, (stride, fm_offset) in enumerate(zip(fm_strides, fm_offsets)):
            fm_height = fm_size_list[fm_i][0]
            fm_width = fm_size_list[fm_i][1]

            shift_x = np.arange(0, fm_width)
            shift_y = np.arange(0, fm_height)
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            xy = np.vstack(
                (shift_y.ravel(), shift_x.ravel())).transpose()  # (hxw) x 2
            # floor(stride / 2) + x * stride?
            off_xy = offset[fm_offset + xy[:, 0] * stride,
                            fm_offset + xy[:, 1] * stride]  # will reduce dim by 1
            # off_max_xy = off_xy.max(axis=2)  # max of l,t,r,b
            off_valid = np.zeros((fm_height, fm_width, boxes_cnt))

            is_in_boxes = (off_xy > 0).all(axis=2)
            off_valid[xy[:, 0], xy[:, 1], :] = is_in_boxes
            off_valid[:, :, 0] = 0  # h x w x boxes_cnt

            hit_gt_ind = np.argmax(off_valid, axis=2)  # h x w

            # gt_boxes
            gt_boxes_res = np.zeros((fm_height, fm_width, 4))
            gt_boxes_res[xy[:, 0],
                         xy[:, 1]] = gt_boxes[hit_gt_ind[xy[:, 0], xy[:, 1]], :4]
            gt_boxes_res_list.append(gt_boxes_res.reshape(-1, 4))

            # cls
            cls_res = np.zeros((fm_height, fm_width))
            cls_res[xy[:, 0], xy[:, 1]] = cls[hit_gt_ind[xy[:, 0], xy[:, 1]]]
            cls_res_list.append(cls_res.reshape(-1))

            # center
            center_res = np.zeros((fm_height, fm_width))
            center_res[xy[:, 0], xy[:, 1]] = center[fm_offset +
                                                    xy[:, 0] * stride, fm_offset +
                                                    xy[:, 1] * stride,
                                                    hit_gt_ind[xy[:, 0], xy[:, 1]]]
            ctr_res_list.append(center_res.reshape(-1))

        cls_res_final = np.concatenate(cls_res_list,
                                       axis=0)[:, np.newaxis].astype(np.float32)
        ctr_res_final = np.concatenate(ctr_res_list,
                                       axis=0)[:, np.newaxis].astype(np.float32)
        gt_boxes_res_final = np.concatenate(gt_boxes_res_list,
                                            axis=0).astype(np.float32)
        return cls_res_final, ctr_res_final, gt_boxes_res_final

    def cal_iou(self, bbox0_pos, bbox0_sz, bbox1_pos, bbox1_sz, im):
        bb0_x0 = bbox0_pos[0] - 0.5 * bbox0_sz[0]
        bb0_y0 = bbox0_pos[1] - 0.5 * bbox0_sz[1]
        bb0_x1 = bbox0_pos[0] + 0.5 * bbox0_sz[0]
        bb0_y1 = bbox0_pos[1] + 0.5 * bbox0_sz[1]
        bb1_x0 = bbox1_pos[0] - 0.5 * bbox1_sz[0]
        bb1_y0 = bbox1_pos[1] - 0.5 * bbox1_sz[1]
        bb1_x1 = bbox1_pos[0] + 0.5 * bbox1_sz[0]
        bb1_y1 = bbox1_pos[1] + 0.5 * bbox1_sz[1]

        '''
        cv2.rectangle(im,(int(bb0_x0),int(bb0_y0)),(int(bb0_x1),int(bb0_y1)),(0,255,0),2)
        #res
        cv2.rectangle(im,(int(bb1_x0),int(bb1_y0)),(int(bb1_x1),int(bb1_y1)),(255,255,0),2)
        #cv2.putText(im, str(frame), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        #cv2.putText(im, str(lost), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #save pic
        save_path = 'picresults'
        #must create file first
        save_img_path = join(save_path)
        if not os.path.exists(save_img_path):
          os.makedirs(save_img_path)
        cv2.imwrite( join(save_img_path, '%08d.jpg') % self._state['framecount'], im)
        '''
        x_left = max(bb0_x0, bb1_x0)
        y_top = max(bb0_y0, bb1_y0)
        x_right = min(bb0_x1, bb1_x1)
        y_bottom = min(bb0_y1, bb0_y1)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersect = (x_right - x_left) * (y_bottom - y_top)
        bb0 = (bb0_x1 - bb0_x0) * (bb0_y1 - bb0_y0)
        bb1 = (bb1_x1 - bb1_x0) * (bb1_y1 - bb1_y0)
        iou = intersect / float(bb0 + bb1 - intersect)
        # print("iou = {}".format(iou))
        return iou
