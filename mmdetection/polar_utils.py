import torch


def get_mask(x, stride, boxes):
    fg_masks = []
    # bg_masks = []
    B, hsize, wsize, C = x.shape
    assert hsize == wsize
    dtype = x.type()
    device = x.device
    for box in boxes:
        if len(box) == 0:
            fg_masks.append(torch.zeros(hsize*wsize).bool().type(dtype).to(device))
            # bg_masks.append(torch.ones(hsize*wsize).bool().type(dtype))
            continue
        yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
        grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype).to(device)
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
        # bg_mask = is_in_boxes.sum(dim=0) <= 0
        fg_masks.append(fg_mask)
        # bg_masks.append(bg_mask)
    fg_masks = torch.stack(fg_masks, 0)
    # bg_masks = torch.stack(bg_masks, 0)
    # return fg_masks, bg_masks
    return fg_masks
