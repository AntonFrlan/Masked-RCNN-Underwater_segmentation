import os
import numpy as np
import torch
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "slika"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "maska"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "slika", self.imgs[idx])
        mask_path = os.path.join(self.root, "maska", self.masks[idx])
        ime = self.imgs[idx].split(".")[0]

        annotations_path = os.path.join(self.root, "annotations", ime + ".txt")
        annote = open(annotations_path, "r")
        labels_num = annote.readline().split(" ")[:-1]
        annote.close()

        img = Image.open(img_path).convert("RGB")  # np.single(np.load(img_path)), zamjenio sam ih ali nezz ako sam to napravio i tokom SSIP-a
        mask = Image.open(mask_path)  # np.load(mask_path), zamjenio sam ih ali nezz ako sam to napravio i tokom SSIP-a
        # mask[mask > 0] = 1  # dodano, valjda ce sada raditi
        # mask = np.single(mask)  # dodano, valjda ce sada raditi
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:, None, None]

        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.from_numpy(np.array(labels_num, dtype=np.int64))
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
