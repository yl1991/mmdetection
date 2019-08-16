import argparse
import json
import os
import os.path as osp
import sys
from PIL import Image
import numpy as np


# -----------------------------------------------------------------------------------------
def parse_wider_gt(dets_file_name, isEllipse=False):
    # -----------------------------------------------------------------------------------------
    '''
      Parse the FDDB-format detection output file:
        - first line is image file name
        - second line is an integer, for `n` detections in that image
        - next `n` lines are detection coordinates
        - again, next line is image file name
        - detections are [x y width height score]
      Returns a dict: {'img_filename': detections as a list of arrays}
    '''
    fid = open(dets_file_name, 'r')

    # Parsing the FDDB-format detection output txt file
    img_flag = True
    numdet_flag = False
    start_det_count = False
    det_count = 0
    numdet = -1

    det_dict = {}
    img_file = ''

    for line in fid:
        line = line.strip()

        if line == '0 0 0 0 0 0 0 0 0 0':
            if det_count == numdet - 1:
                start_det_count = False
                det_count = 0
                img_flag = True  # next line is image file
                numdet_flag = False
                numdet = -1
                det_dict.pop(img_file)
            continue

        if img_flag:
            # Image filename
            img_flag = False
            numdet_flag = True
            # print('Img file: ' + line)
            img_file = line
            det_dict[img_file] = []  # init detections list for image
            continue

        if numdet_flag:
            # next line after image filename: number of detections
            numdet = int(line)
            numdet_flag = False
            if numdet > 0:
                start_det_count = True  # start counting detections
                det_count = 0
            else:
                # no detections in this image
                img_flag = True  # next line is another image file
                numdet = -1

            # print 'num det: ' + line
            continue

        if start_det_count:
            # after numdet, lines are detections
            detection = [float(x) for x in line.split()]  # split on whitespace
            det_dict[img_file].append(detection)
            # print 'Detection: %s' % line
            det_count += 1

        if det_count == numdet:
            start_det_count = False
            det_count = 0
            img_flag = True  # next line is image file
            numdet_flag = False
            numdet = -1

    return det_dict


def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset')
    parser.add_argument(
        '-d', '--datadir', help="dir to widerface", default='data/WIDERFace', type=str)

    parser.add_argument(
        '-s', '--subset', help="which subset to convert", default='all', choices=['all', 'train', 'val'], type=str)

    parser.add_argument(
        '-o', '--outdir', help="where to store annotations", default='data/WIDERFace')

    return parser.parse_args()


def convert_wider_annots(args):
    """Convert from WIDER FDDB-style format to MMDetection style
    
        Annotation format:
    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            'ann': {
                'bboxes': <np.ndarray> (n, 4),
                'labels': <np.ndarray> (n, ),
                'bboxes_ignore': <np.ndarray> (k, 4),
                'labels_ignore': <np.ndarray> (k, 4) (optional field)
            }
        },
        ...
    ]
    The `ann` field is optional for testing.
    """

    subset = ['train', 'val'] if args.subset == 'all' else [args.subset]
    os.makedirs(args.datadir, exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)

    categories = [{"id": 1, "name": 'face'}]
    for sset in subset:
        print(f'Processing subset {sset}')
        out_json_name = osp.join(args.outdir, f'wider_face_{sset}_annot_mmdet_style.json')
        data_dir = osp.join(args.datadir, f'WIDER_{sset}', 'images')
        img_id = 0
        ann_id = 0
        cat_id = 1

        images = []
        ann_file = os.path.join(args.datadir, 'wider_face_split', f'wider_face_{sset}_bbx_gt.txt')
        wider_annot_dict = parse_wider_gt(ann_file)  # [im-file] = [[x,y,w,h], ...]

        for filename in wider_annot_dict.keys():
            if len(images) % 50 ==0:
                print( '{} images processed'.format(len(images)))

            image = {}
            im = Image.open(os.path.join(data_dir, filename))
            image['width'] = im.height
            image['height'] = im.width
            image['filename'] = filename

            # x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose
            ann = {}
            bboxes = np.array([b[:4] for b in wider_annot_dict[filename]], dtype=int) # x,y,w,h
            if not len(bboxes)>0:
                continue
            # bboxes[:,2:] = bboxes[:, 2:] + bboxes[:,:2] - 1
            ann['bboxes'] = bboxes.tolist()
            ann['bboxes'] = [b[:4] for b in wider_annot_dict[filename]] # x,y,w,h
            ann['bboxes_ignore'] = [] 
            ann['blur'] = [int(b[4]) for b in wider_annot_dict[filename]] # x,y,w,h
            ann['expression'] = [int(b[5]) for b in wider_annot_dict[filename]] # x,y,w,h
            ann['illumination'] = [int(b[6]) for b in wider_annot_dict[filename]] # x,y,w,h
            ann['occlusion'] = [int(b[8]) for b in wider_annot_dict[filename]] # x,y,w,h
            ann['pose'] = [int(b[9]) for b in wider_annot_dict[filename]] # x,y,w,h
            ann['labels'] = [1 for b in wider_annot_dict[filename]] # x,y,w,h
            image['ann'] = ann
            images.append(image)

        with open(out_json_name, 'w', encoding='utf8') as outfile:
            json.dump(images, outfile, indent=4, sort_keys=True)


if __name__ == '__main__':
    convert_wider_annots(parse_args())
