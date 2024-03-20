import os
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import DetectMultiBackend
from utils.dataloaders import LoadStreams
from utils.general import (Profile, check_img_size, check_imshow, cv2, non_max_suppression, scale_boxes)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(data=ROOT / 'data/coco.yaml'):
    # model = DetectMultiBackend("./weights/yolov9-e.pt", device=select_device('0'), dnn=False, data=data, fp16=False)
    model = DetectMultiBackend(r"E:\HZSY\program\yolov9\yolov9\weights\yolov9-e.pt", device=select_device('0'), dnn=False, data=data, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size((640, 640), s=stride)

    view_img = check_imshow(warn=True)
    dataset = LoadStreams('0', img_size=imgsz, stride=stride, auto=pt, vid_stride=1)
    bs = len(dataset)

    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    seen, dt = 0, (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]

        with dt[1]:
            pred = model(im, augment=False, visualize=False)
            pred = pred[0][1]

        with dt[2]:
            pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)

        for i, det in enumerate(pred):
            seen += 1
            p, im0 = path[i], im0s[i].copy()

            p = Path(p)
            annotator = Annotator(im0, line_width=3, example=str(names))
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):

                    if view_img:
                        c = int(cls)
                        label = f'{names[c]} {conf:.2f}'
                        print(f"detect 1 {names[c]} in {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))

            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)


if __name__ == "__main__":
    run()
