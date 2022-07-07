# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
import cv2
import torch
from PIL import Image
# from torch.cuda.amp import autocast as autocast

from scene_graph_benchmark.scene_parser import SceneParser
from scene_graph_benchmark.AttrRCNN import AttrRCNN


def cv2Img_to_Image(input_img):
    cv2_img = input_img.copy()
    img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    return img


def detect_objects_on_single_image(model, cv2_imgs,sizes):
    # cv2_img is the original input, so we can get the height and 
    # width information to scale the output boxes.
    # if isinstance(cv2_imgs, list):
    #     img_input = []
    #     for cv2_img in cv2_imgs:
    #         cv2_img = cv2Img_to_Image(cv2_img)
    #         cv2_img, _ = transforms(cv2_img, target=None)
    #         cv2_img = cv2_img.to(model.device)
    #         img_input.append(cv2_img)   
    # else:
    #     cv2_img = cv2_imgs
    #     img_input = cv2Img_to_Image(cv2_imgs)
    #     img_input, _ = transforms(img_input, target=None)
    #     img_input = img_input.to(model.device)

    with torch.no_grad():
        # ys = model(cv2_imgs)
       
        ys = model(cv2_imgs)
        preds = []
        for p in ys:
            preds.append(p.to(torch.device("cpu")))
    assert  len(preds)==len(sizes)
    results = []
    for prediction,(img_width, img_height) in zip(preds,sizes):
        prediction = prediction.resize((img_width, img_height))
        boxes = prediction.bbox.tolist()

        classes = prediction.get_field("labels").tolist()
        scores = prediction.get_field("scores").tolist()
        scores_all = prediction.get_field("scores_all").tolist()
        box_features = prediction.get_field("box_features")
        attr_scores = prediction.get_field("attr_scores")
        attr_labels = prediction.get_field("attr_labels")
        results.append((box_features, [
            {"rect": box, 
            "class": cls, 
            "conf": score,
            'score_all':score_all,
            "attr": attr.tolist(),
            "attr_conf": attr_conf.tolist()}
            for box, cls, score, score_all, attr, attr_conf in
            zip(boxes, classes, scores, scores_all, attr_labels, attr_scores)
        ])
        )
    return results
