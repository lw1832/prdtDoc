from typing import TypedDict

import torch
from PIL.ImageDraw import ImageDraw
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import math

from torch.fft import Tensor

class Bbox:

    @property
    def x1(self):
        return self.layout_box[0]

    @property
    def y1(self):
        return self.layout_box[1]

    @property
    def x2(self):
        return self.layout_box[2]

    @property
    def y2(self):
        return self.layout_box[3]

    @property
    def xyxy(self):
        return [self.x1, self.y1, self.x2, self.y2]

    @property
    def text(self):
        self.sort_ocr()
        return join_plain_text([self.enhance_text(ocr) for ocr in self.ocr_results])

    @staticmethod
    def enhance_text(ocr):
        if ocr[2] == 0:
            return "**" + ocr[1] + "**"
        else:
            return ocr[1]

    def __init__(self, layout_box=None, layout_class=None, layout_score=None, ocr_results=None):
        if torch.is_tensor(layout_box):
            layout_box = [xy.item() for xy in layout_box]
        self.layout_box = layout_box
        if layout_class is None:
            # 默认分类为text
            layout_class = 1
        self.layout_class = layout_class
        if layout_score is None:
            # 默认检测分数为1.0
            layout_score = 1.0
        self.layout_score = layout_score
        self.ocr_results = []
        self.append_ocr(ocr_results)

    def __lt__(self, other):
        if self.y2 <= other.y1 or self.y1 >= other.y2:
            return self.y2 <= other.y1
        return self.x2 <= other.x1

    def merge_bbox(self, x1, y1, x2, y2):
        """
        用于将本bbox与另一个bbox做坐标合并
        (x1, y1) 取两者最小的坐标, (x2, y2) 取两者最大的坐标
        """
        if self.layout_box is None:
            self.layout_box = [x1, y1, x2, y2]
        else:
            self.layout_box = [min(x1, self.x1), min(y1, self.y1), max(x2, self.x2), max(y2, self.y2)]
        return self

    def append_ocr(self, data):
        """
                往此bbox中添加ocr结果
                Args:
                    data: [{'poly':[x1, y1, _, _, x2, y2, _, _], 'text':str}]
                """
        def append_ocr_single(ocr_result):
            bbox = poly_to_bbox(ocr_result['poly'])
            text = ocr_result['text']
            # 顺序将ocr文本添加到texts队列中(未排序)
            self.ocr_results.append([bbox, text, self.layout_class])
            # 更新此bbox的边界坐标
            self.merge_bbox(*bbox)
        if isinstance(data, list):
            for box in data:
                append_ocr_single(box)
        elif data is None:
            return
        else:
            append_ocr_single(data)

    def merge(self, other: 'Bbox', lock_bbox=False):
        if len(other.ocr_results):
            self.ocr_results = self.ocr_results + other.ocr_results
        if not lock_bbox:
            self.merge_bbox(other.x1, other.y1, other.x2, other.y2)
        return self

    def sort_ocr(self):
        if len(self.ocr_results):
            self.ocr_results = sorted(self.ocr_results, key=lambda ocr: ocr[0][1])


def join_plain_text(texts):
    return "".join(texts)

# 计算两个检测框的覆盖率
# IOU = S(A∩B)/S(A∪B)
# 如果两个检测框不相交，返回0.0
def bbox_iou(box1, box2):
    """
    计算两个边界框之间的交并比（IoU）。

    参数:
    - box1: 第一个边界框，格式为 (x1, y1, x2, y2)
    - box2: 第二个边界框，格式为 (x1, y1, x2, y2)

    返回:
    - iou: 交并比（IoU）值
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    inter_width = inter_x2 - inter_x1 + 1
    inter_height = inter_y2 - inter_y1 + 1

    if inter_width <= 0 or inter_height <= 0:
        return 0.0

    inter_area = inter_width * inter_height
    box1_area = (x2_1 - x1_1 + 1) * (y2_1 - y1_1 + 1)
    box2_area = (x2_2 - x1_2 + 1) * (y2_2 - y1_2 + 1)
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area

    return iou

def bbox_belong(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    inter_width = inter_x2 - inter_x1 + 1
    inter_height = inter_y2 - inter_y1 + 1

    if inter_width <= 0 or inter_height <= 0:
        return 0.0

    inter_area = inter_width * inter_height
    box1_area = (x2_1 - x1_1 + 1) * (y2_1 - y1_1 + 1)
    box2_area = (x2_2 - x1_2 + 1) * (y2_2 - y1_2 + 1)
    return inter_area / min(box1_area, box2_area)

# 计算两个检测框的距离，
def bbox_dis(box1, box2):
    # 两个框有交集，则距离为0
    if bbox_iou(box1, box2) > 0:
        return 0.0

    x1_1, y1_1, x1_2, y1_2 = box1
    x2_1, y2_1, x2_2, y2_2 = box2

    # 计算两个框的距离
    x12_min = min(x1_1, x2_1)
    x12_max = max(x1_2, x2_2)
    y12_min = min(y1_1, y2_1)
    y12_max = max(y1_2, y2_2)
    dx1 = x1_2 - x1_1
    dx2 = x2_2 - x2_1
    dy1 = y1_2 - y1_1
    dy2 = y2_2 - y2_1
    ddx = (abs((x1_1 + x1_2) - (x2_1 + x2_2)) - (dx1 + dx2)) / 2
    ddy = (abs((y1_1 + y1_2) - (y2_1 + y2_2)) - (dy1 + dy2)) / 2

    if dx1 + dx2 < x12_max - x12_min and dy1 + dy2 < y12_max - y12_min:
        return math.sqrt(ddx**2 + ddy**2)
    elif dx1 + dx2 < x12_max - x12_min and dy1 + dy2 >= y12_max - y12_min:
        return ddx
    else:
        return ddy

LAYOUT_DETECTION_CLASSES = {
    0: 'title',
    1: 'plain text',
    2: 'abandon',
    3: 'figure',
    4: 'figure_caption',
    5: 'table',
    6: 'table_caption',
    7: 'table_footnote',
    8: 'isolate_formula',
    9: 'formula_caption'
}

# {'category_type': 'text', 'poly': [197.0, 92.0, 522.0, 92.0, 522.0, 131.0, 197.0, 131.0], 'score': 1.0, 'text': '中国邮政储蓄银行'}
def poly_to_bbox(item):
    return [item[0], item[1], item[4], item[5]]

def title_text_dis(title, text):
    # 两个框有交集，则距离为0
    dis = bbox_dis(title, text)
    ti_x1, ti_y1, ti_x2, ti_y2 = title
    tx_x1, tx_y1, tx_x2, tx_y2 = text
    # title 在 text 的右方或者下方
    above = tx_x2 > ti_x1 and ti_y1 < tx_y2
    return dis, above

def layout_cluster(bboxes, eps):
    if len(bboxes) <= 1:
        return bboxes
    cluster_boxes = {}
    bboxes_xyxy = [bbox.xyxy for bbox in bboxes]
    dbscan = DBSCAN(eps=eps, min_samples=1, metric=lambda x, y: bbox_dis(x, y))
    dbscan.fit(bboxes_xyxy)
    dbscan_result = dbscan.labels_
    for bbox, dbscan in zip(bboxes, dbscan_result):
        cluster = cluster_boxes.get(dbscan)
        if cluster is None:
            cluster_boxes[dbscan] = bbox
        else:
            cluster.merge(bbox)
    return list(cluster_boxes.values())

def sort(bbox_list):
    return sorted(bbox_list, key=lambda x: (x.x1, x.y1))

def layout_title_merge_to_nearest_bbox(title_bboxes, text_bboxes, above=True, merge_lock_bbox=False):
    unmatched_title_bboxes = []
    for title in title_bboxes:
        min_dis = 999
        target_text = None
        for text in text_bboxes:
            dis, abv = title_text_dis(title.xyxy, text.xyxy)
            if above and not abv:
                continue
            if dis <= 50:
                target_text = text if min_dis >= dis else target_text
                min_dis = min(dis, min_dis)
        if target_text:
            target_text.merge(title, lock_bbox=merge_lock_bbox)
        else:
            unmatched_title_bboxes.append(title)
    return text_bboxes, unmatched_title_bboxes

def ocr_layout_merge(ocr_results, layout_boxes, layout_classes, layout_scores):
    """
        根据ocr和layout任务输出生成结果集
    Args:

        ocr_results: 当页image的ocr识别对象数组
        [{'category_type': 'text', 'poly': [197.0, 92.0, 522.0, 92.0, 522.0, 131.0, 197.0, 131.0], 'score': 1.0, 'text': '中国邮政储蓄银行'}, ...]
        layout_boxes: 当页image的layout识别对象数组（boxes）
        [[197.0, 92.0, 522.0, 131.0], ...]
        layout_classes: layout_boxes对应的分类数组
        [0, 1, 0, 2, 1, ...]
        layout_scores: 对应识别分数

    Returns:
        {
            'texts': ['xxxx', 'xxxx', ...],
            'image_boxes': [[197.0, 92.0, 522.0, 131.0], ...],
            (以下暂未实现)
            'formular_boxes': [[197.0, 92.0, 522.0, 131.0], ...],
            'table_boxes': [[197.0, 92.0, 522.0, 131.0], ...],
            ...
        }
    """
    # 根据分类为检测框分组
    boxes_title = []
    boxes_texts = []
    boxes_images = []
    for idx, (layout_box, layout_class, layout_score) in enumerate(zip(layout_boxes, layout_classes, layout_scores)):
        layout_class_int = int(layout_class)
        if layout_class_int == 1:
            boxes_texts.append(Bbox(layout_box=layout_box, layout_class=layout_class_int, layout_score=layout_score))
        elif layout_class_int == 0:
            boxes_title.append(Bbox(layout_box=layout_box, layout_class=layout_class_int, layout_score=layout_score))
        elif (layout_score > 0.75) and (layout_class_int == 3 or layout_class_int == 5):
            boxes_images.append(Bbox(layout_box=layout_box, layout_class=layout_class_int, layout_score=layout_score))

    # 对bbox进行聚类，合并重复的、有交集的或者距离很近的bbox框
    boxes_texts = layout_cluster(boxes_texts, 1)
    boxes_images = layout_cluster(boxes_images, 1)

    # 将ocr绑定到bbox中
    visited = set()
    for box in boxes_texts + boxes_title:
        for idx, ocr_result in enumerate(ocr_results):
            # 判断ocr_bbox是否属于该layout_plaint_text_bbox
            if idx in visited or bbox_belong(poly_to_bbox(ocr_result['poly']), box.xyxy) <= 0.75:
                continue
            box.append_ocr(ocr_result)
            visited.add(idx)

    boxes_texts, unmatched_title_bboxes = layout_title_merge_to_nearest_bbox(boxes_title, boxes_texts)
    boxes_texts = sorted(boxes_texts + unmatched_title_bboxes)
    boxes_images, _ = layout_title_merge_to_nearest_bbox(boxes_title, boxes_images, above=False, merge_lock_bbox=True)
    boxes_images = sorted(boxes_images)
    # 获取figure
    return {'image_boxes': boxes_images, 'text_boxes': boxes_texts}





