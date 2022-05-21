#!/usr/bin/env python
# coding=utf-8
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import roi_align


class NewMultiScaleRoIAlign(MultiScaleRoIAlign):

    def __init__(self, featmap_names, output_size, sampling_ratio):
        super(NewMultiScaleRoIAlign, self).__init__(
            featmap_names, output_size, sampling_ratio
        )

    def forward(self, x, boxes, image_shapes):
        x_filtered = []
        for k, v in x.items():
            if k in self.featmap_names:
                x_filtered.append(v)
        rois = self.convert_to_roi_format(boxes)
        if self.scales is None:
            self.setup_scales(x_filtered, image_shapes)

        scales = self.scales
        assert scales is not None
        result = []
        for level, (per_level_feature, scale) in enumerate(
            zip(x_filtered, scales)
        ):
            this_level_out = roi_align(
                per_level_feature, rois,
                output_size=self.output_size,
                spatial_scale=scale, sampling_ratio=self.sampling_ratio
            )
            result.append(this_level_out)
        return result


