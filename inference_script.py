from typing import Dict, Tuple
import time
import cv2
from loguru import logger
import random
import os
from data_models.images_input import Images
import numpy as np
from data_models.onnx_object_detection import OnnxObjectDetection
from utils.visualization import write_to_disk
import argparse 

yolo_classnames = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush']

yolo_colors: Dict[str, Tuple[int, int, int]] = {cls_name: [random.randint(0, 255) for _ in range(3)] for k, cls_name in
                                                enumerate(yolo_classnames)}

yolov7_tiny = "./weights/yolov7-tiny/yolov7-tiny.onnx"
input_folder = "inputs"
output_folder = "outputs"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Object Detection')
    parser.add_argument('--input', type=str, default=input_folder, help='Path to the input folder or image')
    parser.add_argument('--output', type=str, default=output_folder, help='Path to the output folder')
    parser.add_argument('--weights', type=str, default=yolov7_tiny, help='Path to the weights file')
    parser.add_argument('--time', type=bool, default=False, help='Print time taken for inference')
    
    args = parser.parse_args()
    modelPath = args.weights
    input_folder = args.input
    output_folder = args.output
    toTime = args.time

    startime = time.time()

    yolov7 = OnnxObjectDetection(weight_path=modelPath, classnames=yolo_classnames)

    if os.path.isfile(input_folder):
        images = Images(images=[Images.read_from_file(input_folder)])
    else:
        images = Images(images=Images.read_from_folder(path=input_folder, ext="jpg"))

    for i, batch in enumerate(images.create_batch(batch_size=4)):
        logger.info(f"Processing batch: {i} containing {len(batch)} image(s)...")

        raw_out = yolov7.predict_object_detection(input_data=batch.to_onnx_input(image_size=yolov7.input_size))
        batch.init_detected_objects(raw_out)

        annotations = batch.annotate_objects(input_size=yolov7.input_size, letterboxed_image=True, class_colors=yolo_colors)
        write_to_disk(path=output_folder, images=annotations,
                             names=batch.get_image_ids())
    if toTime:
        logger.info(f"Total time: {time.time() - startime:.2f} seconds.")
