# Import from third library
import torch.nn as nn
import numpy as np
from scipy.ndimage import filters

# Import from pod
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from up.models.losses import build_loss
from up.utils.model.normalize import build_norm_layer

__all__ = ['KeypointSBHead']


@MODULE_ZOO_REGISTRY.register('keypoint_sb')
class KeypointSBHead(nn.Module):
    """
    keypoint head for solo top-down keypoint detection method.(HKD)
    """
    def __init__(self,
                 inplanes,
                 mid_channels,
                 num_classes,
                 loss,
                 has_bg,
                 test_with_gaussian_filter,
                 normalize={'type': 'solo_bn'}):
        super(KeypointSBHead, self).__init__()
        self.has_bg = has_bg
        self.test_with_gaussian_filter = test_with_gaussian_filter
        self.num_classes = num_classes
        self.loss = build_loss(loss)

        self.relu = nn.ReLU(inplace=True)

        self.upsample5 = nn.ConvTranspose2d(
            inplanes[0], mid_channels, kernel_size=4, stride=2, padding=1)
        self.upsample4 = nn.ConvTranspose2d(
            mid_channels, mid_channels, kernel_size=4, stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose2d(
            mid_channels, mid_channels, kernel_size=4, stride=2, padding=1)

        self.bn_s5 = build_norm_layer(mid_channels, normalize)[1]
        self.bn_s4 = build_norm_layer(mid_channels, normalize)[1]
        self.bn_s3 = build_norm_layer(mid_channels, normalize)[1]
        if has_bg:
            self.predict3 = nn.Conv2d(mid_channels, num_classes + 1, kernel_size=1, stride=1, padding=0)
        else:
            self.predict3 = nn.Conv2d(mid_channels, num_classes, kernel_size=1, stride=1, padding=0)
        self.do_sig = loss['type'] != 'mse'

    def forward(self, input):
        output = {}
        if self.training:
            losses = self.get_loss(input)
            output.update(losses)
        else:
            score_map = self.forward_net(input).cpu()
            if self.do_sig:
                score_map = nn.functional.sigmoid(score_map)
            kpts = self.final_preds(score_map.numpy(),
                                    self.has_bg,
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
                for i in range(self.num_classes):
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
            output.update({'res': all_res})
        return output

    def forward_net(self, input):
        c5 = input['features'][0]
        p5 = self.relu(self.bn_s5(self.upsample5(c5)))
        p4 = self.relu(self.bn_s4(self.upsample4(p5)))
        p3 = self.relu(self.bn_s3(self.upsample3(p4)))
        pred = self.predict3(p3)
        return pred

    def get_loss(self, input):
        output = self.forward_net(input)
        target = input['label']
        loss = self.loss(output, target[0])
        return {'All.loss': loss}

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
