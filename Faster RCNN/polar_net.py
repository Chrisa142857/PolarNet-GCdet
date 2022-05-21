import torch
import torch.nn as nn

import torch.nn.functional as F
import random

class PolarNet(torch.nn.Module):
    def __init__(self, mtype='v3', hidden_num=256, feat_num=2048, max_possible_img_size=1024/32):
        super().__init__()
        self.hidden_num = hidden_num
        self.pad_len = int(max_possible_img_size**2)
        self.mtype = mtype
        self.mask_id_cache = {}
        self.polar_conv1 = nn.Conv2d(in_channels=hidden_num, out_channels=hidden_num, kernel_size=1, stride=1)
        self.polar_conv2 = nn.Conv2d(in_channels=hidden_num, out_channels=hidden_num, kernel_size=1, stride=1)
        self.polar_conv3 = nn.Conv2d(in_channels=hidden_num, out_channels=hidden_num, kernel_size=1, stride=1)
        self.norm_layer = nn.LayerNorm(1024)
        # self.norm_layer = None
        # self.polar_conv3 = None
        if mtype == 'v3':
            self.fc1 = nn.Linear(in_features=self.pad_len, out_features=feat_num)
            self.dropout = nn.Dropout(0.94)
            self.act = nn.ReLU()
            self.fc2 = nn.Linear(in_features=feat_num, out_features=2)

    def generate_8neib_id(self, H, W, device):
        if '%d-%d' % (H, W) not in self.mask_id_cache:
            ids = torch.LongTensor([i for i in range(H*W)]).to(device).reshape(H, W)
            x_id = torch.LongTensor([
                [i-1 for _ in range(3)] + 
                [i for _ in range(3)] + 
                [i+1 for _ in range(3)]
                for i in range(H)
            ]).clamp(min=0, max=H-1).to(device)
            y_id = torch.LongTensor([
                [i-1, i, i+1] * 3
                for i in range(W)
            ]).clamp(min=0, max=W-1).to(device)
            mask_id = []
            # nei_keys = []
            # nei_vectors = []
            for i in range(H):
                for j in range(W):
                    mask_id.append(ids[x_id[i], y_id[j]])
                    # nei_keys.append(torch.index_select(k, 2, ids[x_id[i], y_id[j]]))
                    # if self.polar_conv3 != None:
                    #     nei_vectors.append(torch.index_select(v, 2, ids[x_id[i], y_id[j]]))
            mask_id = torch.cat(mask_id)
            self.mask_id_cache['%d-%d' % (H, W)] = mask_id
        else:
            mask_id = self.mask_id_cache['%d-%d' % (H, W)]
        return mask_id

    def get_polarity(self, x):
        B, C, H, W = x.shape
        q = self.polar_conv1(x).reshape(B, C, H*W).permute(0, 2, 1)
        q = q.unsqueeze(2) # B x N x 1 x C, N=H*W
        k = self.polar_conv2(x).reshape(B, C, H*W)
        if self.polar_conv3 != None:
            v = self.polar_conv3(x).reshape(B, C, H*W)
        mask_id = self.generate_8neib_id(H, W, x.device)
        nei_keys = torch.index_select(k, 2, mask_id).reshape(B, C, H*W, 9)
        nei_keys = nei_keys.permute(0, 2, 1, 3) # B x N x C x 9, N=H*W
        if self.polar_conv3 != None:
            nei_vectors = torch.index_select(v, 2, mask_id).reshape(B, C, H*W, 9)
        nei_vectors = nei_vectors.permute(0, 2, 3, 1) # B x N x 9 x C, N=H*W

        out = torch.einsum('bnpc,bncq->bnpq', q, nei_keys) # B x N x 1 x 9, N=H*W, P=1, Q=9 
        if self.polar_conv3 != None:
            new_v = torch.einsum('bnpq,bnqc->bnpc', out, nei_vectors) # B x N x 1 x C, N=H*W, P=1, Q=9 
            new_v = new_v.squeeze(dim=2).permute(0, 2, 1) # B, C, H*W
            new_v = new_v.reshape(B, C, H, W)
        out = out.squeeze(dim=2) # B, H*W, 9
        out = out.reshape(B, H, W, 9)  # B, H, W, 9
        scores = F.softmax(out, dim=-1) # B, H, W, 9
        polarity = scores.argmax(dim=-1) # B, H, W
        output = torch.stack([
            out[..., 4], 
            torch.cat([
                out[..., :4].sum(dim=-1, keepdim=True), 
                out[..., 5:].sum(dim=-1, keepdim=True)
                ], dim=-1).sum(-1) / 8
                ], dim=1) # B, H, W, 2

        if self.polar_conv3 != None:
            x = x + new_v
            if self.norm_layer != None:
                x = F.layer_norm(x, [H, W])
        return [polarity, output, scores], x
        
    def predict_score(self, polarity): # B, H*W
        x = self.fc1(polarity)
        x = self.act(x)
        x = self.dropout(x)
        scores = self.fc2(x)
        return scores

    # def collect_polarity_from_box(self, polarity, pred_boxes, image_shape, feat_shape, flatten=True):
    #     num_images = len(pred_boxes)
    #     polarities = []
    #     for i in range(num_images):
    #         boxes = (pred_boxes[i].detach().cpu()/(image_shape/feat_shape)).long()
    #         # dx = boxes[:, 2] - boxes[:, 0]
    #         # dy = boxes[:, 3] - boxes[:, 1]
    #         for j in range(boxes.shape[0]):
    #             p = polarity[j:j+1, boxes[j, 0] : boxes[j, 2], boxes[j, 1] : boxes[j, 3]]
    #             if flatten:
    #                 p = p.flatten()
    #                 p = F.pad(p, (0, self.pad_len-p.shape[0])).unsqueeze(0)
    #                 assert p.shape[0] == self.pad_len
    #             polarities.append(p)
    #     if len(polarities) == 0:
    #         return polarities
    #     else:
    #         return torch.cat(polarities)
    
    def forward(self, x):
        # polarity, x = self.get_polarity(x)
        # return polarity_loss(polarity), x
        return self.get_polarity(x)
    
def polarity_loss(polarity, labels=None):
    # v4 ####################
    # labels = (torch.cat(labels, dim=0) > 0).long()
    have_pos = 0
    have_neg = 0
    labels = (labels > 0).long()
    assert labels.shape[0] == polarity.shape[0]
    pos_num = sum(labels)
    neg_num = len(labels) - pos_num
    min_num = min(pos_num, neg_num)
    loss_neg_num = min_num
    if min_num == neg_num and min_num < pos_num:
        loss_pos_num = min_num + 1
    else:
        loss_pos_num = min_num
    p_scores = F.avg_pool2d(polarity, (polarity.shape[-2], polarity.shape[-1]))[:,:,0,0]
    if loss_pos_num > 0:
        pos_id = random.sample(torch.where(labels==1)[0].cpu().tolist(), loss_pos_num)
        pos_loss = F.cross_entropy(p_scores[pos_id], labels[pos_id])
        # pos_loss = F.nll_loss(p_scores[pos_id], labels[pos_id]).log()
        have_pos = 1
    else:
        pos_loss = 0
    if loss_neg_num > 0:
        neg_id = random.sample(torch.where(labels==0)[0].cpu().tolist(), loss_neg_num)
        neg_loss = F.cross_entropy(p_scores[neg_id], labels[neg_id])
        # neg_loss = F.nll_loss(p_scores[neg_id], labels[neg_id]).log()
        have_neg = 1
    else:
        neg_loss = 0
    if have_pos + have_neg == 0:
        return 0
    return (pos_loss+neg_loss) / (have_pos + have_neg)

    # v2 ####################
    # -1 * (polarity)
    # dx = polarity[..., 1:, :] - polarity[..., :-1, :]
    # dy = polarity[..., :, :1] - polarity[..., :, :-1]
    # loss1 = torch.sqrt((dx ** 2).mean()) + torch.sqrt((dy ** 2).mean())
    # if torch.isnan(loss1):
    #     loss1 = 0
    # dx = polarity[..., 2::2, :] - polarity[..., :-2:2, :]
    # dy = polarity[..., :, 2::2] - polarity[..., :, :-2:2]
    # loss2 = torch.sqrt((dx ** 2).mean()) + torch.sqrt((dy ** 2).mean())
    # if torch.isnan(loss2):
    #     loss2 = 0
    # dx = polarity[..., 3::3, :] - polarity[..., :-3:3, :]
    # dy = polarity[..., :, 3::3] - polarity[..., :, :-3:3]
    # loss3 = torch.sqrt((dx ** 2).mean()) + torch.sqrt((dy ** 2).mean())
    # if torch.isnan(loss3):
    #     loss3 = 0
    # return loss1 + loss2/2 + loss3/3

def get_mask(polarity, image_shapes, targets):
    boxes = [t['boxes'] for t in targets]
    fg_masks = []
    bg_masks = []
    hsize, wsize = polarity.shape[-2:]
    dtype = polarity.type()
    for image_shape, box in zip(image_shapes, boxes):
        if len(box) == 0:
            fg_masks.append(torch.zeros(hsize*wsize).bool().type(dtype))
            bg_masks.append(torch.ones(hsize*wsize).bool().type(dtype))
            continue
        stride = image_shape[0] / hsize
        yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
        grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
        grid = grid.view(1, -1, 2)
        grid = grid * stride + stride/2
        grid_x = grid[:, :, 0].repeat(len(box), 1) # num_gt x num_grid
        grid_y = grid[:, :, 1].repeat(len(box), 1) # num_gt x num_grid

        box_x1 = box[:, 0:1].repeat(1, hsize*wsize) # num_gt x num_grid
        box_y1 = box[:, 1:2].repeat(1, hsize*wsize) # num_gt x num_grid
        box_x2 = box[:, 2:3].repeat(1, hsize*wsize) # num_gt x num_grid
        box_y2 = box[:, 3:4].repeat(1, hsize*wsize) # num_gt x num_grid
        
        b_l = grid_x - box_x1
        b_r = box_x2 - grid_x
        b_t = grid_y - box_y1
        b_b = box_y2 - grid_y
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2) # num_gt x num_grid x 4

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0 # num_gt x num_grid
        fg_mask = is_in_boxes.sum(dim=0) > 0
        bg_mask = is_in_boxes.sum(dim=0) <= 0
        fg_masks.append(fg_mask)
        bg_masks.append(bg_mask)
    fg_masks = torch.stack(fg_masks, 0)
    bg_masks = torch.stack(bg_masks, 0)
    return fg_masks, bg_masks
