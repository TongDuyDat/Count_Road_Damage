import sys
sys.path.insert(0, './yolov5')

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.utils.dataloaders import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import json

def detect(opt):
    Object_Damage = dict()
    out, source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz, evaluate = \
        opt.output, opt.source.replace("\\","/"), opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, \
            opt.save_txt, opt.img_size, opt.evaluate
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            # shutil.rmtree(out)  # delete output folder
        else:
            os.makedirs(out)  # make new output folder
    save_path = str(Path(out))
    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(out)).replace("\\","/") + '/' + txt_file_name+ "/" + txt_file_name + '.json'
    result_path = os.path.join(out,txt_file_name).replace("\\","/")
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path)
    
    # Init function save_txt
    def save_txt_function(id, frame_idx, bbox, conf, im0, labels):
        if not (os.path.exists(os.path.join(result_path+"/img/origin/")) or os.path.exists(os.path.join(result_path+"/img/bbox/"))):
            os.makedirs(os.path.join(result_path+"/img/origin/"))
            os.makedirs(os.path.join(result_path+"/img/bbox/"))
        path_origin = "" + result_path+"/img/origin/{}_{}.png".format(txt_file_name, id)
        path_bb = "" + result_path+"/img/bbox/{}_{}.png".format(txt_file_name, id)
        img = Annotator(im0.copy(), line_width=2, pil=not ascii)
        img.box_label(labels[0], labels[1], color=colors(c, True))
        temp = {"bbox":bbox, "frame_idx": frame_idx, "conf": conf, "path_origin": path_origin, "path_bb": path_bb}
        if id in Object_Damage:
            if Object_Damage[id]["conf"] < conf:
                cv2.imwrite(path_origin, im0)
                cv2.imwrite(path_bb, img.result())
                Object_Damage[id] = temp
        else:
            Object_Damage[id] = temp

    # Load model
    half = device.type != 'cpu'  # half precision only supported on CUDA
    model = attempt_load(yolo_weights, device=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16
    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
    print("Deetection on {}".format(dataset.mode))
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    W,H = (imgsz, imgsz)

    for frame_idx, (path, img, im0s, vid_cap, _) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_sync()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_sync()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)
            origin_img = im0.copy()
            annotator = Annotator(im0, line_width=5, pil=not ascii)
            if det is not None and len(det):
                W, H = im0.shape[0], im0.shape[1] 
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                # draw boxes for visualization
                if save_txt and dataset.mode == "image":
                    Bbox = det[:,:4]
                    for idx , (bb, conf, cls) in enumerate(zip(Bbox, confs, clss)):
                        bbox_left = int(bb[0])
                        bbox_top = int(bb[1])
                        bbox_w = int(bb[2] - bb[0])
                        bbox_h = int(bb[3] - bb[1])
                        label = f'{idx} {names[int(cls)]} {conf:.2f}'
                        box = {"bbox_left": bbox_left,  "bbox_top": bbox_top, "bbox_w": bbox_w,  "bbox_h": bbox_h}
                        Object_Damage[idx] = {"bbox":box, "conf": int(conf), "class": int(cls)}
                        annotator.box_label(bb,label,color=colors(c, True))
                    cv2.imwrite("{}/{}_origin.png".format(result_path,txt_file_name), origin_img)
                    cv2.imwrite("{}/{}_bbox.png".format(result_path,txt_file_name), im0)
                if len(outputs) > 0:
                    bbox = []
                    for j, (output, conf) in enumerate(zip(outputs, confs)): 
                        bboxes = output[0:4]
                        id = int(output[4])
                        cls = output[5]
                        c = int(cls)  # integer class
                        label = f'{id} {names[c]} {conf:.2f}'
                        annotator.box_label(bboxes, label, color=colors(c, True))
                        box = {}
                        if save_txt and dataset.mode != "image":
                            # to MOT format
                            bbox_left = int(output[0])
                            bbox_top = int(output[1])
                            bbox_w = int(output[2] - output[0])
                            bbox_h = int(output[3] - output[1])
                            # Write MOT compliant results to file
                            box["bbox_left"] = bbox_left; box["bbox_top"] = bbox_top; box["bbox_w"] = bbox_w; box["bbox_h"] = bbox_h    
                            #id, frame_idx, bbox, conf, path_origin, path_bb, im0, im
                            save_txt_function(id, frame_idx, box, float(conf), origin_img, [output[0:4], label])
            else:
                deepsort.increment_ages()

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            im0 = annotator.result()
            if show_vid:
                im0 = cv2.putText(im0,"Count: {}".format(len(Object_Damage)),(50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),3, cv2.LINE_AA)
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)
    if save_txt or save_vid:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if (save_txt):
            labels_result = {"file_path": source.split("/")[-1], "size":[W, H], "obj_Damage":[] }
            for obj_id, value in list(Object_Damage.items()):
                item_obj = {"ID": obj_id, "value": value}
                labels_result['obj_Damage'].append(item_obj)
            with open(txt_path, "w") as file:
                json.dump(labels_result, file)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)
    print('Done. (%.3fs)' % (time.time() - t0), "count = ", len(Object_Damage))
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', nargs='+', type=str, default=ROOT/'yolov5/fine_tuning1.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_weights', type=str, default=ROOT/'deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', help='ckpt.t7 path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)

    with torch.no_grad():
        detect(args)
