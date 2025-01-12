#!/usr/bin/env python
# coding: utf-8

# In[ ]:
input_size = 384
batch_size=77
sequence_length=1
initial_prefetch_size=16

# In[ ]:
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import torch

@pipeline_def(enable_conditionals=True)
def video_pipe(filenames):
    videos, _ = fn.experimental.readers.video(
        device="cpu", 
        sequence_length=1,
        prefetch_queue_depth=1,
        shard_id=0, 
        num_shards=1, 
        random_shuffle=False, 
        initial_fill=256,
        filenames=filenames,
        labels=[0]
    )
    videos = fn.resize(
        videos,
        device="cpu", 
        resize_x=384,
        resize_y=384,
        dtype=types.UINT8
    )
    padded = fn.pad(videos, axes=[1,2], shape=input_size)

    # drop F dimension since we have F=1, don't do this if you want video frags
    no_seq = fn.reshape(padded, src_dims=[1, 2, 3])
    # HWC => CHW for CVT compat
    reshape = fn.transpose(no_seq, perm=[2, 0, 1])

    return reshape


# In[ ]:
from nvidia.dali.plugin.pytorch import DALIGenericIterator

class PoseLightningWrapper(DALIGenericIterator):
    def __init__(self, *kargs, **kvargs):
        super().__init__(*kargs, output_map=["data"], **kvargs)

    def __next__(self):
        out = super().__next__()
        # DDP is used so only one pipeline per process
        # also we need to transform dict returned by DALIClassificationIterator to iterable
        # and squeeze the lables
        out = out[0]
        return [out[k] if k == "data" else torch.squeeze(out[k]) for k in self.output_map]


# In[ ]:
device = "cpu"
    
# In[ ]:

### YOLO

import yolov7
from yolov7.utils.general import non_max_suppression, non_max_suppression_kpt
from yolov7.utils.plots import output_to_keypoint

skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

def keypoint_to_skele(output, steps=3):
    feats = []

    for idx in range(output.shape[0]):
        kpts = output[idx, 7:].T
        f = []
        keep = False
            
        for sk_id, sk in enumerate(skeleton):
            pos1_x, pos1_y = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
            pos2_x, pos2_y = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
            conf1 = kpts[(sk[0]-1)*steps+2]
            conf2 = kpts[(sk[1]-1)*steps+2]
            keep = keep or conf1 >= 0.5 or conf2 >= 0.5
            f.append((pos1_x, pos1_y, pos2_x, pos2_y, conf1, conf2))
        if keep:
            feats.append(f)
    feats = np.array(sorted(feats, key=lambda f: -1*np.mean([conf1+conf2 for pos1_x, pos1_y, pos2_x, pos2_y, conf1, conf2 in f])))

    if feats.shape[0] > 3:
        feats = feats[0:3, :, :]
    elif feats.shape[0] == 0:
        feats = np.zeros((0, len(skeleton), 6))
    return feats

def load_yolo_model(src):
    model = yolov7.load(src)
    model.float().eval()

    return model

yolo_pose_model = load_yolo_model('yolov7-w6-pose.pt')
yolo_human_model = load_yolo_model('yolov7.pt')

def yolo_inference(frames):
    with torch.no_grad():
        output, _ = yolo_pose_model(frames)
        # print(yolo_pose_model.yaml['nc'])
        # print(yolo_pose_model.yaml['nkpt'])
        output = non_max_suppression_kpt(output, 
                                 0.45, # Confidence Threshold
                                 0.65, # IoU Threshold
                                 nc=yolo_pose_model.yaml['nc'], # Number of Classes
                                 nkpt=yolo_pose_model.yaml['nkpt'], # Number of Keypoints
                                 kpt_label=True)
        output = output_to_keypoint(output)
        if not output.size:
            return [np.zeros((0, len(skeleton), 6), dtype=np.float32) for i in range(batch_size)]
        output = [keypoint_to_skele(output[output[:, 0] == i, :]) for i in range(batch_size)]
            
    return output

# In[ ]:

### ALPHAPOSE

import torch
import torch.nn.functional as F
import torch.utils.data
import numpy as np
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.transforms import get_func_heatmap_to_coord
from alphapose.utils.bbox import _box_to_center_scale, _center_scale_to_box
from alphapose.utils.transforms import get_affine_transform
# opencv, 

class AlphaPoseDetector:
    def __init__(self):
        # Load config
        cfg_path = '/model/256x192_res50_lr1e-3_2x-regression.yaml'
        self.cfg = update_config(cfg_path)
        self.cfg.gpus = [0]
        self.cfg.device = device

        # Build pose model
        self.pose_model = builder.build_sppe(self.cfg.MODEL, preset_cfg=self.cfg.DATA_PRESET)
        self.pose_model.load_state_dict(torch.load('/model/halpe136_fast50_regression_256x192.pth', map_location='cpu'))
        self.pose_model = self.pose_model.to('cpu').eval()

        self.orig_dim_list = torch.FloatTensor([(input_size, input_size)] * batch_size).repeat(1,2)
        
        self.heatmap_to_coord = get_func_heatmap_to_coord(self.cfg)
        self.eval_joints = list(range(136))

    def detect_poses(self, frames):
        """
        Perform pose detection on a GPU tensor image.

        Args:
            image_tensor (torch.Tensor): Image tensors in RGB format, shape (B, C, H, W)

        Returns:
            list: Pose estimations, shape (B, N, K, 3) where N is the number of detected persons,
                          K is the number of keypoints, and the last dimension contains (x, y, score)
            list: Bounding Boxes, shape (B, N, 5) where N is the number of detected persons,
                          and the last dimension contains (min_x, min_y, max_x, max_y, z)

        """

        with torch.no_grad():
            output, _ = yolo_human_model(frames)
            # print("yolo")
            # print(output)
            # print(output.shape)
            detected_boxes = non_max_suppression(output, conf_thres=0.5, iou_thres=0.45, classes=[0])
            print(detected_boxes)

        if detected_boxes is None or len(detected_boxes) == 0:
            return np.zeros((batch_size, 0, 68, 3), dtype=np.float16)

        pose_estimations = []
        hm_size = self.cfg.DATA_PRESET.HEATMAP_SIZE
        norm_type = self.cfg.LOSS.get('NORM_TYPE', None)
        
        for boxes_k in detected_boxes:
            inps = torch.zeros(boxes_k.size(0), 3, *self.cfg.DATA_PRESET.IMAGE_SIZE)
            cropped_boxes = torch.zeros(boxes_k.size(0), 4)
            
            for i, box in enumerate(boxes_k):
                inps[i], cropped_boxes[i] = self._preprocess_single(frames[i], box)

            inp_mask = cropped_boxes.sum(1) != 0
            inps, cropped_boxes = inps[inp_mask], cropped_boxes[inp_mask]
            
            if not inps.size(0):
                pose_estimations.append([])
                continue
            
            with torch.no_grad():
                heatmaps = self.pose_model(inps)
                # print("heatmaps: ")
                # print(heatmaps)
                
                pe = []              
                for hm, bb in zip(heatmaps, cropped_boxes.cpu().numpy()):
                    pose_coord, pose_score = self.heatmap_to_coord(
                      hm[self.eval_joints], bb, hm_shape=hm_size, norm_type=norm_type)
                    pe.append(np.hstack((pose_coord, pose_score)))
                pose_estimations.append(np.array(pe))

        return pose_estimations
        
    def _preprocess_single(self, image_tensor, box):
         """Preprocess a single bounding box."""
         inp_h, inp_w = self.cfg.DATA_PRESET.IMAGE_SIZE
         center, scale = _box_to_center_scale((box[2] - box[0]) / 2, (box[3] - box[1]) / 2, inp_w, inp_h)
         trans = get_affine_transform(center, scale, 0, [inp_w, inp_h])
         trans = torch.tensor(trans, device=device, dtype=torch.float32)
    
         # Convert the 2x3 affine matrix to a 3x3 matrix
         affine_matrix = torch.eye(3, device=device, dtype=torch.float32)
         affine_matrix[:2, :] = trans
    
         # Invert the affine matrix (we need the inverse transform)
         inv_affine = torch.inverse(affine_matrix)
    
         # Create sampling grid
         theta = inv_affine[:2, :].unsqueeze(0)
         grid = F.affine_grid(theta, size=(1, 3, inp_h, inp_w), align_corners=False)
    
         # Perform the affine transformation
         img = F.grid_sample(image_tensor.unsqueeze(0), grid, align_corners=False, mode='bilinear').squeeze(0)
         bbox = torch.FloatTensor(_center_scale_to_box(center, scale))

         return img, bbox
    
# In [ ]:

video_path = "vadim.mp4"

# In[ ]:

import ffmpeg
from datetime import datetime

detector = AlphaPoseDetector()

# In[ ]:
import json


def skele_exec(observation):

    probe = ffmpeg.probe(video_path)
    video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
    frame_count = int(video_streams[0]['nb_frames'])        

    pipeline = video_pipe(batch_size=batch_size, num_threads=4, device_id=None, filenames=[video_path])
    pipeline.build()

    all_ap = []
    all_y7 = []

    start_time = datetime.utcnow()

    with torch.no_grad():
        for bdx, batch in enumerate(PoseLightningWrapper(pipeline, size=frame_count)):
            # print(batch[0])
            # print(batch[0].shape)
            frames = batch[0]
            
            # print(frames)
            # print(frames.shape)

            all_ap.extend([
                np.array(p).flatten().astype(np.float32) for p in detector.detect_poses(frames / 255.0)])
            all_y7.extend([
                y.flatten().astype(np.float32) for y in yolo_inference(frames)])

            if not bdx % 50:
                current_time = datetime.now()
                n_processed = (bdx + 1) * batch_size

                print(f"Step: {bdx}, Metrics: {{"
                    f"'processed_frames': {n_processed}, "
                    f"'current_fps': {n_processed / ((current_time - start_time).total_seconds() + 1):.2f}, "
                    f"'completion_percentage': {n_processed / frame_count:.2%}, "
                    f"'processing_time_seconds': {(current_time - start_time).total_seconds():.2f}"
                    f"}}")


    current_time = datetime.utcnow()

    # Keypoint mappings
    body_keypoints = {
        0: "Nose", 1: "LEye", 2: "REye", 3: "LEar", 4: "REar",
        5: "LShoulder", 6: "RShoulder", 7: "LElbow", 8: "RElbow",
        9: "LWrist", 10: "RWrist", 11: "LHip", 12: "RHip",
        13: "LKnee", 14: "RKnee", 15: "LAnkle", 16: "RAnkle",
        17: "Head", 18: "Neck", 19: "Hip", 20: "LBigToe", 21: "RBigToe",
        22: "LSmallToe", 23: "RSmallToe", 24: "LHeel", 25: "RHeel"
    }
    # Face keypoints
    face_keypoints = {i: f"FacePoint_{i-26}" for i in range(26, 94)}
    # Left hand keypoints
    left_hand_keypoints = {i: f"LHandPoint_{i-94}" for i in range(94, 115)}
    # Right hand keypoints
    right_hand_keypoints = {i: f"RHandPoint_{i-115}" for i in range(115, 136)}
    all_keypoints = {**body_keypoints, **face_keypoints, **left_hand_keypoints, **right_hand_keypoints}

    # parse data into joint mappings
    def parse_alphapose_data(alphapose_array):
        parsed_data = {}
        for i in range(0, len(alphapose_array), 3):
            keypoint_index = i // 3
            x, y, confidence = alphapose_array[i:i + 3]
            joint_name = all_keypoints.get(keypoint_index, f"UnknownPoint_{keypoint_index}")
            parsed_data[joint_name] = {"x": x, "y": y, "confidence": confidence}
        return parsed_data

    with open("output.json", 'w') as f:
        for i in range(frame_count):
            frame_data = {
                "features": {
                    "alphapose": (
                        parse_alphapose_data(all_ap[i].tolist()) if i < len(all_ap) else {}
                    ),
                    "yolov7": all_y7[i].tolist() if i < len(all_y7) else [],
                }
            }
            f.write(json.dumps(frame_data) + "\n")


    print(f"{observation} complete: {current_time - start_time}")

def export_model(model, filename):
    try:
        # Ensure the model is in evaluation mode
        model.eval()
        
        # Sample input tensor (batch size, channels, height, width)
        input_tensor = torch.rand(77, 3, 384, 384)  # This should match the model's input requirements
        
        # Export the model to ONNX format
        torch.onnx.export(model, input_tensor, filename)
        print(f"Model exported successfully to {filename}")
    except Exception as e:
        print(f"Error exporting model: {e}")

# export_model(yolo_pose_model, "yolov7_w6_pose.onnx")
export_model(yolo_human_model, "yolov7.onnx")
# export_model(detector.pose_model, "alpha_pose.onnx")
skele_exec(observation="observation")