# Import from third library
import torch.nn as nn
import numpy as np
from scipy.ndimage import filters
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from up.models.losses import build_loss
from functools import reduce

__all__ = ['BaseKpPostProcess']


@MODULE_ZOO_REGISTRY.register('kp_post_process')
class BaseKpPostProcess(nn.Module):
    def __init__(self, loss, test_with_gaussian_filter):
        super(BaseKpPostProcess, self).__init__()
        self.test_with_gaussian_filter = test_with_gaussian_filter
        self.loss = build_loss(loss)
        self.do_sig = loss['type'] != 'mse'

    def get_loss(self, input):
        target = input['label']
        if isinstance(input['pred'], list):
            loss = reduce(lambda x, y: x + y, map(self.loss, input['pred'], target))
        else:
            loss = self.loss(input['pred'], target[0])
        return {'All.loss': loss}

    def get_output(self, input):
        if isinstance(input['pred'], list):
            score_map = input['pred'][-1].cpu()
        else:
            score_map = input['pred'].cpu()
        if self.do_sig:
            score_map = nn.functional.sigmoid(score_map)
        kpts = self.final_preds(score_map.numpy(),
                                input['has_bg'],
                                self.test_with_gaussian_filter)  # x,y,score
        output_w = score_map.size()[3]
        output_h = score_map.size()[2]
        all_res = []
        for nperson in range(len(kpts)):
            bbox = input['bbox'][nperson]
            filename = input['filename'][nperson]
            bbox = input['bbox'][nperson]
            dt_box = input['dt_box'][nperson]
            box_score = input['box_score'][nperson]
            sum_scores = 0
            cur_res = dict()
            cur_res['image_id'] = int(filename.split('.')[0])
            cur_kpt = []
            for i in range(input['num_classes']):
                kpt = kpts[nperson][i]
                x = float((kpt[0] - output_w / 2 + 0.5) / output_w * bbox[2] + bbox[0])
                y = float((kpt[1] - output_h / 2 + 0.5) / output_h * bbox[3] + bbox[1])
                sum_scores += kpt[2]
                cur_kpt.append(x)
                cur_kpt.append(y)
                cur_kpt.append(float(kpt[2]))
            cur_res['keypoints'] = cur_kpt
            cur_res['kpt_score'] = float(sum_scores)
            cur_res['bbox'] = dt_box.tolist()
            cur_res['box_score'] = float(box_score)
            all_res.append(cur_res)
        return {'res': all_res}

    def forward(self, input):
        output = {}
        if self.training:
            output.update(self.get_loss(input))
        else:
            output.update(self.get_output(input))
        return output

    def final_preds(self, scores, bg=True, with_gaussian_filter=True):
        assert scores.ndim == 4, 'Score maps should be 4-dim'

        on, oc, oh, ow = scores.shape

        '''
        This operation will make the final output score smooth enough that most values
        of the feature map will be lower than 0.1
        If you want to use the score for visiablity judgement, comment the following
        three lines
        '''
        if with_gaussian_filter:
            for n in range(on):
                for c in range(oc):
                    scores[n][c] = filters.gaussian_filter(
                        scores[n][c], (5, 5), mode='constant')

        max_idx = np.argmax(np.reshape(scores, (on, oc, -1)), 2)
        if bg:
            max_idx = max_idx[:, :-1]
        preds = []
        for n in range(on):
            kpts = []
            for c in range(max_idx.shape[1]):
                idx = max_idx[n][c]
                fx = x = int(idx % ow)
                fy = y = int(idx / ow)
                if 0 < x < ow - 1:
                    if scores[n][c][y][x - 1] < scores[n][c][y][x + 1]:
                        fx += 0.25
                    elif scores[n][c][y][x - 1] > scores[n][c][y][x + 1]:
                        fx -= 0.25
                if 0 < y < oh - 1:
                    if scores[n][c][y - 1][x] < scores[n][c][y + 1][x]:
                        fy += 0.25
                    elif scores[n][c][y - 1][x] > scores[n][c][y + 1][x]:
                        fy -= 0.25
                kpts.append([fx, fy, scores[n][c][y][x]])
            preds.append(kpts)
        return preds
