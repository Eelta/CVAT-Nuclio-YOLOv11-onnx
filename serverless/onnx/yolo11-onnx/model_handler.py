# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import cv2
import numpy as np
import onnxruntime as ort


class ModelHandler:
    def __init__(self, labels):
        self.model = None
        self.load_network(model="yolo11n.onnx")
        self.labels = labels

    def load_network(self, model):
        device = ort.get_device()
        cuda = True if device == 'GPU' else False
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            so = ort.SessionOptions()
            so.log_severity_level = 3

            self.model = ort.InferenceSession(model, providers=providers, sess_options=so)
            self.input_name = self.model.get_inputs()[0].name
            self.is_initiated = True
        except Exception as e:
            raise Exception(f"Cannot load model {model}: {e}")

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleup=True, stride=32):
        shape = im.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

        if auto:
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)

        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im, r, (dw, dh)

    def _nms(self, boxes, scores, iou_threshold=0.45):
        if len(boxes) == 0:
            return []
        boxes = boxes.astype(np.float32)
        scores = scores.astype(np.float32)
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=0.0, nms_threshold=iou_threshold)
        if len(indices) == 0:
            return []
        if isinstance(indices[0], list) or isinstance(indices[0], np.ndarray):
            indices = [i[0] for i in indices]
        return indices

    def _infer(self, inputs: np.ndarray):
        try:
            img = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB)
            image = img.copy()
            image, ratio, dwdh = self.letterbox(image, auto=False)
            image = image.transpose((2, 0, 1))
            image = np.expand_dims(image, 0)
            image = np.ascontiguousarray(image)

            im = image.astype(np.float32) / 255.0

            detections = self.model.run(None, {self.input_name: im})[0]

            predictions = np.squeeze(detections).T

            scores = np.max(predictions[:, 4:], axis=1)
            mask = scores >= 0.25
            predictions = predictions[mask]
            scores = scores[mask]

            if len(predictions) == 0:
                return [], [], []

            labels = np.argmax(predictions[:, 4:], axis=1)

            boxes = predictions[:, :4]

            boxes[:, 0] -= boxes[:, 2] / 2
            boxes[:, 1] -= boxes[:, 3] / 2
            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]

            dw, dh = dwdh
            boxes[:, [0, 2]] -= dw
            boxes[:, [1, 3]] -= dh
            boxes /= ratio

            h_orig, w_orig = inputs.shape[:2]
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w_orig)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h_orig)

            indices = self._nms(boxes, scores, iou_threshold=0.45)
            if len(indices) == 0:
                return [], [], []

            boxes = boxes[indices]
            labels = labels[indices]
            scores = scores[indices]

            return boxes.astype(np.int32), labels, scores

        except Exception as e:
            print(f"Inference error: {e}")
            return [], [], []

    def infer(self, image, threshold):
        image = np.array(image)
        image = image[:, :, ::-1].copy()
        h, w, _ = image.shape

        boxes, labels, scores = self._infer(image)

        results = []
        for box, label, score in zip(boxes, labels, scores):
            if score >= threshold:
                xtl, ytl, xbr, ybr = box
                results.append({
                    "confidence": str(float(score)),
                    "label": self.labels.get(int(label), "unknown"),
                    "points": [int(xtl), int(ytl), int(xbr), int(ybr)],
                    "type": "rectangle",
                })

        return results
