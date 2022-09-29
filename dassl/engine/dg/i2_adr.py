# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
import torch
import random
import copy
import warnings
from dassl.engine.trainer import SimpleNet
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, resume_from_checkpoint, load_pretrained_weights
)
from collections import OrderedDict
from dassl.optim import build_optimizer, build_lr_scheduler

@TRAINER_REGISTRY.register()
class I2ADRNet(TrainerX):
    """
    I2-ADR.
    """
    def build_model(self):
        """Build and register model.
        The default builds a classification model along with its
        optimizer and scheduler.
        Custom trainers can re-implement this method if necessary.
        """
        cfg = self.cfg
        print('Building Inter-ADR model')
        self.model = SimpleNet(cfg, cfg.MODEL, self.num_classes)
        self.teachers = [copy.deepcopy(SimpleNet(cfg, cfg.MODEL, self.num_classes)) for _ in
                        range(len(cfg.DATASET.SOURCE_DOMAINS))]
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
            
        if cfg.MODEL.TEACHER_WEIGHTS:
            teacher_pretrains = cfg.MODEL.TEACHER_WEIGHTS.split(',')
            for (teacher, path) in zip(self.teachers, teacher_pretrains):
                load_pretrained_weights(teacher, path)
                
        self.model.to(self.device)
        for teacher in self.teachers:
            teacher.to(self.device)

        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model('model', self.model, self.optim, self.sched)
        
    def forward_backward(self, batch):
        input, label = self.parse_batch_train(batch)
        y_ce, y_adr, adc_out, fms = self.model(input, return_feature=True)
        _, cls_pred = y_ce.max(dim=1)
        cls_loss_ce = F.cross_entropy(y_ce, label)
        cls_loss_adr = F.cross_entropy(y_adr, label)
        cls_loss = .5 * (cls_loss_ce + cls_loss_adr)
        """
        Inter-ADR
        """
        if random.random() < 0.2:
            t_fms, t_cls_pred = [], []
            with torch.no_grad():
                for teacher in self.teachers:
                        t_y_ce, t_y_adr, _, t_fms_0 = teacher(input, return_feature=True)
                        _, t_cls_pred_0 = t_y_ce.max(dim=1)
                        t_fms.append(t_fms_0)
                        t_cls_pred.append(t_cls_pred_0)
            dir_loss, dvr_loss = Inter_ADR(t_cls_pred, cls_pred, t_fms, fms, label, self.device)
            inter_loss = dir_loss * 200 - dvr_loss
        else:
            inter_loss = 0
        """
        Intra-ADR
        """
        if random.random() < 0.2:
            intra_loss = (1.0 - torch.mean(adc_out)) * 0.05
        else:
            intra_loss = 0
        
        loss = cls_loss + intra_loss + inter_loss
        self.model_backward_and_update(loss)
        loss_summary = {
            'loss': loss.item(),
            'acc': compute_accuracy(y_ce, label)[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch['img']
        label = batch['label']
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    
def Inter_ADR(t_cls_pred, cls_pred, t_fms, fms, label, device):
    t_mask = [(t_cls_pred[i] == label.data) * 1. for i in range(len(t_cls_pred))]
    t_mask = [t_mask[i].view(1, -1).permute(1, 0) for i in range(len(t_cls_pred))]
    mask = (cls_pred == label.data) * 1.
    mask = mask.view(1, -1).permute(1, 0)
    t_ats = [at(t_fms[i]) for i in range(len(t_cls_pred))]
    ats = at(fms)
    l2_dirs, l2_dvrs = 0, 0
    
    for res_i in range(len(ats)):
        t_mask_dir = [t_mask[i].repeat(1, ats[res_i].size()[1]).to(device) for i in range(len(t_cls_pred))]
        mask_dir = mask.repeat(1, ats[res_i].size()[1]).to(device)

        t_mask_dvr = [1. - t_mask_dir[i] for i in range(len(t_cls_pred))]
        mask_dvr = 1. - mask_dir

        u_plus_temp = [t_ats[i][res_i].unsqueeze(2).contiguous() * t_mask_dir[i].unsqueeze(2).contiguous() for i in range(len(t_cls_pred))]
        u_plus_temp += [(ats[res_i].unsqueeze(2).contiguous() * mask_dir.unsqueeze(2).contiguous())]
        u_plus_temp = torch.cat(u_plus_temp, dim=2)
        u_plus = u_plus_temp.max(2)[0]

        u_minus_temp = [t_ats[i][res_i].unsqueeze(2).contiguous() * t_mask_dvr[i].unsqueeze(2).contiguous() for i in range(len(t_cls_pred))]
        u_minus_temp += [(ats[res_i].unsqueeze(2).contiguous() * mask_dvr.unsqueeze(2).contiguous())]
        u_minus_temp = torch.cat(u_minus_temp, dim=2)
        u_minus = u_minus_temp.max(2)[0]
        
        mask_plus_0 = torch.gt(u_plus, torch.zeros_like(u_plus)).to(device)
        mask_minus_0 = torch.gt(u_minus, torch.zeros_like(u_minus)).to(device)

        l2_dir = ((ats[res_i] * mask_plus_0 - u_plus.detach())**2).mean() * fms[res_i].shape[1]
        l2_dvr = ((ats[res_i] * mask_minus_0 - 0 *u_minus.detach())**2).mean() * fms[res_i].shape[1]
        l2_dirs += l2_dir
        l2_dvrs += l2_dvr
        
    return l2_dirs, l2_dvrs

def at(fms):
    ats = []
    for fm in fms:
        (N, C, H, W) = fm.shape
        ats.append(F.softmax(fm.reshape(N, C, -1), -1).mean(1))
    return ats
