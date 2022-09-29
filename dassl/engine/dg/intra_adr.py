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

@TRAINER_REGISTRY.register()
class IntraADRNet(TrainerX):
    """Intra ADR."""
    def forward_backward(self, batch):
        input, label = self.parse_batch_train(batch)
        y_ce, y_adr, adc_out = self.model(input)
        cls_loss_ce = F.cross_entropy(y_ce, label)
        cls_loss_adr = F.cross_entropy(y_adr, label)
        cls_loss = 0.5 * (cls_loss_ce + cls_loss_adr)
        if random.random() < 0.33:
            adc_loss = (1.0 - torch.mean(torch.mean(adc_out, 2))) * 0.05
        else:
            adc_loss = 0
        
        loss = cls_loss + adc_loss
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
