# ----------------------------------------------------
# Written by JeongMin Kim(jm.kim@dankook.ac.kr)
# ----------------------------------------------------
import json
import numpy as np
from dataclasses import dataclass
from os import path as osp
from typing import List, Tuple
from pycocotools.coco import COCO as PyCOCO


@dataclass
class ImageInfo():
    license: int
    coco_url: str
    flickr_url: str
    file_name: str
    height: Tuple[int, float]
    width: Tuple[int, float]
    date_captured: None
    id: int


@dataclass
class Annotation:
    area: float
    iscrowd: int
    image_id: int
    bbox: List[float]
    category_id: int
    id: int
    keypoints: List[float]
    num_keypoints: int


class COCOEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (ImageInfo, Annotation)):
            return o.__dict__
        if isinstance(o, np.float32):
            return float(o)
        return super().default(o)


class COCO:
    def __init__(self, annotation_file) -> None:
        super.__init__(annotation_file)
    
    def get_keypoints(self, image_id=None):
        """ 구현 되지 않은 부분 필요하면 구현하면 됨. """
        pass 
    


class KeypointDB:
    def __init__(self, args, json_file=None, is_load_coco=False):
        self.output_dir = args.output_dir
        self.input_dir = args.input_dir
        self.is_load_coco = is_load_coco
        self.json_file_list = json_file
        self.db = dict(
            info=dict(
                description=None,
                url=None,
                version=None,
                year="",
                contributor="",
                date_created="",
            ),
            licenses=[
                dict(
                    url=None,
                    id=0,
                    name=None,
                )
            ],
            images=[
                # license, url, file_name, height, width, date_captured, id
            ],
            annotations=[
                # segmentation, area, iscrowd, image_id, bbox, category_id, id
            ],
            categories=[
                # supercategory, id, name
            ],
        )

    def load_coco_json(self):
        # If you load coco json file, json file is only one file
        if self.is_load_coco is False:
            return 
        with open(self.json_file_list, 'r') as file:
            data = json.load(file)
            # self.db = data
            flags = np.array([True, True, True, True])
            if 'annotations' in data:
                flags[0] = False
                self.db['annotations'] = data['annotations']
            if 'images' in data:
                flags[1] = False
                self.db['images'] = data['images']
            if 'categories' in data:
                flags[2] = False
                self.db['categories'] = data['categories']
            if 'info' in data:
                flags[3] = False
                self.db['info'] = data['info']
            if 'licenses' in data:
                self.db['licenses'] = data['lincenses']
            and_result = None
            for i in range(len(flags)-1):
                and_result = flags[i] and flags[i+1]
            if  and_result:
                self.db = data
                       
        return
    
    def add_annotations(self, annotations):
        self.db['annotations'].extend(annotations)

    def saver(self):
        with open(osp.join(self.output_dir, "annotations.json"), 'w') as f:
            json.dump(self.db, f, indent=4, cls=COCOEncoder)

        print(f"save json -> {osp.join(self.output_dir, 'annotations.json')}")

        
class KeypointDBAdapter(KeypointDB):
    def __init__(self, args, json_file=None, is_load_coco=False):
        super().__init__(args, json_file, is_load_coco)
        self.load_coco_json()
    def get_keypoints(self, image_id):
        annotations = self.db['annotations']
        keypoints = [ann['keypoints'] for ann in annotations if ann['image_id'] == image_id]
        return keypoints


class ResultJson2KeypointDB(KeypointDB):
    def __init__(self, args, result_json):
        super().__init__(args)
        self.result_json = result_json
        self.convert_and_add()
        
    def convert_and_add(self):
        annotations = []
        with open(self.result_json, 'r') as file:
            data = json.load(file)
            for i, result in enumerate(data):
                annotation = dict(
                    area=0,
                    iscrowd=0,
                    image_id=result['image_id'],
                    bbox=[0, 0, 0, 0],
                    category_id = 1,
                    id = i,
                    keypoints = result['keypoints'],
                    num_keypoints=len(result['keypoints']) // 3
                )
                annotations.append(annotation)
        self.add_annotations(annotations)


class KeypointManager:
    def __init__(self, coco_annotation_file, keypoint_db_args, keypoint_db_json_file, result_json, is_load_coco):
        self.coco = COCO(coco_annotation_file)
        self.keypoint_db = KeypointDB(keypoint_db_args, keypoint_db_json_file, is_load_coco)
        self.result_adapter = ResultJson2KeypointDB(keypoint_db_args, result_json)
        
    def get_keypoints(self, image_id, source='coco'):
        if source == 'coco':
            raise NotImplemented("COCO's get_keypoints method is not implemented.")
            return self.coco.get_keypoints(image_id)
        elif source == 'keypoint_db':
            return self.keypoint_db.get_keypoints(image_id)
        elif source == 'result_json':
            return self.result_adapter.get_keypoints(image_id)
        else:
            raise ValueError("Source must be either 'coco', 'keypoint_db', or 'result_json'")
    