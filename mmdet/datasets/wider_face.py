import numpy as np
import mmcv
from .registry import DATASETS
from .custom import CustomDataset
@DATASETS.register_module
class WIDERFaceDataset(CustomDataset):
    """
    Reader for the WIDER Face dataset in PASCAL VOC format.
    Conversion scripts can be found in
    https://github.com/sovrasov/wider-face-pascal-voc-annotations
    """
    CLASSES = ('face', )

    def __init__(self, min_size=None, **kwargs):
        self.min_size = min_size
        super(WIDERFaceDataset, self).__init__(**kwargs)

    def load_annotations(self, ann_file):
        img_info = mmcv.load(ann_file)
        for i, img in enumerate(img_info):
            ann = img['ann']
            gt_bboxes = []
            gt_bboxes_ignore = []
            gt_labels = []
            for box in ann['bboxes']:
                x1, y1, w, h= box
                area = w * h
                bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
                if self.min_size is not None:
                    if area < self.min_size:
                        gt_bboxes_ignore.append(bbox)
                    else:
                        gt_labels.append(1)
                        gt_bboxes.append(bbox)
                else:
                    gt_labels.append(1)
                    gt_bboxes.append(bbox)

            if gt_bboxes:
                gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
                gt_labels = np.array(gt_labels, dtype=np.int64)
            else:
                gt_bboxes = np.zeros((0, 4), dtype=np.float32)
                gt_labels = np.array([], dtype=np.int64)

            if gt_bboxes_ignore:
                gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
            else:
                gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

            ann = dict(
                bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)
            img_info[i]['ann'] = ann


                
        return img_info
