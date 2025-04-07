import os
from typing import Optional, Callable
from torch.utils.data import Dataset, DataLoader

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
import random


def find_contours(image):
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []

    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            centers.append((center_x, center_y))

    return centers


CLASS_NAMES = ['apple', 'corn', 'tomato',
               'grape', 'palm', 'peach', 'pepper',
               'potato', 'strawberry',
               'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
               'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4',
               'pipe_fryum']

MULTI_CLASS = [
    'candle', 'capsules', 'macaroni1', 'macaroni2'
]

Chinese_position = {
    'top': '上方',
    'top left': '左上方',
    'top right': '右上方',
    'bottom': '下方',
    'bottom left': '左下方',
    'bottom right': '右下方',
    'center': '中间',
    'left': '左侧',
    'right': '右侧'
}

Chinese_class_names = {'apple': ["苹果叶子", "苹果叶片"], 'corn': ["玉米叶子", "玉米叶片"],
                       'tomato': ['番茄叶子', '番茄叶片'],
                       'grape': ['葡萄叶子', '葡萄叶片'], 'palm': ['棕榈叶片', '棕榈叶子'], 'peach': ['桃子叶子'],
                       'pepper': ['辣椒叶子，辣椒叶片'],
                       'potato': ['马铃薯叶子'], 'strawberry': ['草莓叶子', '草莓叶片'],
                       'candle': ['蜡烛'], 'capsules': ["药丸", "胶囊"], 'cashew': ['腰果'], 'chewinggum': ['口香糖'],
                       'fryum': ['元件', '元器件', '样品', '样本'],
                       'macaroni1': ['元件', '元器件', '样品', '样本'], 'macaroni2': ['元件', '元器件', '样品', '样本'],
                       'pipe_fryum': ['元件', '元器件', '样品', '样本']}

class_questions = [
    'This is an image for anomaly detection. What is the content of the image?',
    "What's the object in the image?",
    "What's this in the image?",
    "Describe this image.",
    "Take a look at this image and describe what you notice.",
    "Please provide a description of the picture.",
    "Could you describe the contents of this image for me?",
    "Can you identify the elements present in the image?"
    "What can you observe in this picture?",
    "Describe the objects shown in the image.",
    "Could you list the items visible in this image?",
    "What do you see in the picture?",
    "Identify the various components of this image.",
    "What is depicted in the photograph?",
    "Provide a rundown of the contents of this image.",
    "What's the subject matter of this image?",
    "Enumerate the objects that can be spotted in this image.",
    "Describe the visual elements within the picture.",
    "What visual information can you extract from this image?",
    "What elements compose the scene in the image?",
    "Please give a verbal depiction of the image.",
    "From your perspective, what is shown in the image?",
    "Could you break down the objects present in the picture?",
    "Summarize the contents of the image in your own words.",
    "What details can you identify within the image?",
    "Provide a textual account of the image's contents.",
    "Based on the image, can you discern any notable features?"
]

class_questions_cn = [
    "这是一张用于异常检测的图像。图像内容是什么？",
    "图像中有什么物体？",
    "图像中是什么东西？",
    "描述一下这张图片。",
    "请看一下这张图片，描述你注意到的内容。",
    "请提供这张图片的描述。",
    "你能描述一下这张图片的内容吗？",
    "你能识别出图像中的元素吗？",
    "你能在这张图片中看到什么？",
    "描述图像中展示的物体。",
    "你能列举出这张图片中可见的物品吗？",
    "你在图片里看到了什么？",
    "识别出图像中的各个组成部分。",
    "照片中描绘了什么？",
    "简要介绍一下这张图片的内容。",
    "这张图片的主题是什么？",
    "列举出这张图片中可以看到的物体。",
    "描述图片中的视觉元素。",
    "你能从这张图片中提取出什么视觉信息？",
    "图像中有哪些元素构成了场景？",
    "请用口头方式描述这张图片。",
    "从你的角度来看，这张图片展示了什么？",
    "你能分解出图片中存在的物体吗？",
    "用你自己的话概括一下图片的内容。",
    "你能在图像中识别出哪些细节？",
    "用文字叙述一下图片的内容。",
    "基于这张图片，你能辨别出哪些显著的特征吗？"
]

single_answers = [
    'This in the image is {}.',
    'What you\'re seeing here is {}.',
    'In this image, the featured object is {}.',
    '{} is visible in this picture.',
    'The object captured in the image is {}.',
    'The highlighted item is {}.',
    'It appears to be {} in the image.',
    'You\'re looking at {} in this photograph.',
    'This is none other than {}.',
    'The image showcases {}.',
    'What\'s presented here is {}.',
    'The focus is on {} in this image.',
    '{} is what we have in the image.',
    'The photographed subject is {}.',
    'This image contains {}.',
    'The visible entity is {}.',
    'The image encapsulates {}.',
    'The main subject here is {}.',
    'The image portrays {}.',
    'The item captured is {}.'
]

single_answers_cn = [
    "在图像中，这个物体是{}。",
    "你看到的是{}。",
    "在这张图片里，焦点放在了{}上。",
    "这张照片中展现出一个{}。",
    "图中的物体就是一个{}。",
    "图中突出显示的是一个{}。",
    "图中似乎是一个{}。",
    "这是{}。",
    "这张图片展示了一个{}。",
    "这里展现的是一个{}。",
    "图中重点呈现的是一个{}。",
    "图中的{}是我们所关注的。",
    "照片中的主要内容是{}。",
    "这张图片呈现了一个{}。",
    "图中可见的实体是{}。",
    "这张图片包含了{}。",
    "这张图片主要展现了{}。",
    "图片中描绘了{}。",
    "图片中拍摄到的物品是{}。"
]

multi_answers = [
    'In the image, there are several {}.',
    'You can spot multiple instances of {}.',
    'What you\'re seeing here is a collection of {}.',
    'A variety of {} are visible in this picture.',
    'The image captures several {}.',
    'The highlighted objects are {}.',
    'You\'ll notice a group of {} in this image.',
    'This photograph features several {}.',
    'The scene is filled with {}.',
    'Multiple instances of {} are depicted here.',
    'The image showcases an assortment of {}.',
    'What\'s presented here is a multitude of {}.',
    'In this image, numerous {} can be observed.',
    'The photographed scene contains several {}.',
    'This image encapsulates a number of {}.',
    'The visible entities are {}.',
    'The image portrays a variety of {}.',
    'You\'re looking at multiple {} in this photograph.',
    'Several instances of {} are what we have in the image.',
    'The items captured are {}.',
]

multi_answers_cn = [
    "在图像中，有几个{}。",
    "你可以看到多个{}的实例。",
    "你在这里看到的是一组{}的集合。",
    "这张图片中可见多种类型的{}。",
    "图像捕捉到了几个{}。",
    "突出显示的物体是{}。",
    "你会注意到这张图片中有一组{}。",
    "这张照片中展示了几个{}。",
    "场景中充满了{}。",
    "这里描绘了多个{}的情景。",
    "这张图片展示了各种各样的{}。",
    "这里呈现的是多种{}的众多实例。",
    "在这张图片中，你可以观察到许多{}。",
    "所拍摄的场景包含了几个{}。",
    "这张图片涵盖了若干个{}。",
    "图中可见的实体是{}。",
    "这张图片描绘了多种{}。",
    "你在这张照片中看到了多个{}。",
    "图中展现了几个{}的实例。",
    "图中所拍摄到的物品是{}。"
]

anomaly_questions = [
    'Are there any anomalies in the image?',
    'Are there any defects in the image?',
    'Is there any defect in the image?',
    'Is there any anomaly in the image?',
    'Do you observe any irregularities in the image?',
    'Are there any discrepancies in the image?',
    'Can you identify any aberrations in the image?',
    'Do you notice any abnormalities in the image?',
    'Are there any inconsistencies in the image?',
    'Is there any deviance in the image?',
    'Are there any anomalies present in the image?',
    'Do you perceive any faults in the image?',
    'Can you spot any atypical elements in the image?',
    'Are there any variations from the norm in the image?',
    'Do you see any irregular occurrences in the image?',
    'Is there any departure from the standard in the image?',
    'Can you detect any nonconformities in the image?',
    'Are there any divergences in the image?',
    'Do you identify any incongruities in the image?',
    'Is there any departure from expectations in the image?',
    'Are there any aberrant features in the image?',
    'Can you pinpoint any anomalies in the image?',
    'Do you discern any atypical aspects in the image?',
    'Are there any unusual elements in the image?'
]

anomaly_questions_cn = [
    "图像中是否存在任何异常？",
    "图像中是否存在任何缺陷？",
    "图像中是否有任何缺陷？",
    "图像中是否存在任何异常？",
    "你是否观察到图像中的任何不规则之处？",
    "你能否识别出图像中的任何异常现象？",
    "你是否注意到图像中的任何异常情况？",
    "图像中是否存在任何不一致之处？",
    "图像中是否存在任何异常情况？",
    "你是否察觉到图像中的任何缺陷？",
    "你能否发现图像中的任何非典型元素？",
    "图像中是否存在与常规不同的地方？",
    "你是否在图像中看到任何不规则的事件？",
    "图像中是否存在与标准不符的地方？",
    "图像中是否存在任何分歧？",
    "你是否辨别出图像中的任何不一致之处？",
    "图像中是否存在与预期不符的地方？",
    "图像中是否存在任何异常特征？",
    "你能否准确定位图像中的任何异常？",
    "图像中是否存在任何不寻常的元素？"
]

normal_answers = [
    'No, there is no anomaly in the image.',
    'No, there is no defect in the image.',
    'No, there are no anomalies in the image.',
    'No, there are no defects in the image.',
    "No, this is a photo of {} without any anomalies.",
    "No, this is a photo of {} without any defects.",
    'No, there is no irregularity in the image.',
    'No, there is no imperfection in the image.',
    'No, there are no abnormalities in the image.',
    'No, there are no blemishes in the image.',
    'No, this is a photo of {} without any irregularities.',
    'No, this is a photo of {} without any imperfections.',
    'No, there are no irregularities present in the image.',
    'No, there are no flaws in the image.',
    'No, there are no anomalies detected in the image.',
    'No, there are no defects to be found in the image.',
    'No, this is a photo of {} with no irregularities.',
    'No, this is a photo of {} with no imperfections.',
    'No, the image is free from irregularities.',
    'No, the image does not exhibit any flaws.',
    'No, there are no abnormalities observed in the image.',
    'No, there are no blemishes spotted in the image.',
    'No, this image of {} shows no irregularities.',
    'No, this image of {} displays no imperfections.',
    'No, there are no irregularities visible in the image.',
    'No, there are no defects evident in the image.'
]

normal_answers_cn = [
    "不，图像中没有任何异常。",
    "不，图像中没有任何缺陷。",
    "不，图像中没有任何异常现象。",
    "不，图像中没有任何缺陷。",
    "不，这是一张没有任何异常的{}照片。",
    "不，这是一张没有任何缺陷的{}照片。",
    "不，图像中没有任何不规则之处。",
    "不，图像中没有任何瑕疵。",
    "不，图像中没有任何异常情况。",
    "不，图像中没有任何瑕疵。",
    "不，这是一张没有任何不规则之处的{}照片。",
    "不，这是一张没有任何瑕疵的{}照片。",
    "不，图像中没有任何不规则现象。",
    "不，图像中没有任何瑕疵。",
    "不，图像中没有任何异常被检测出。",
    "不，图像中没有任何缺陷可寻找。",
    "不，这是一张没有任何不规则之处的{}照片。",
    "不，这是一张没有任何瑕疵的{}照片。",
    "不，图像中没有任何不规则之处。",
    "不，图像中没有任何瑕疵。",
    "不，图像中没有任何异常现象。",
    "不，图像中没有任何瑕疵。",
    "不，这张{}的照片没有任何不规则之处。",
    "不，这张{}的照片没有任何瑕疵。",
    "不，图像中没有任何不规则之处可见。",
    "不，图像中没有任何可见瑕疵。"
]

detail_questions = [
    "What's the anomaly?",
    "What's the defect?",
    "What are the anomalies?",
    "What are the defects?",
    "Why you think so?",
    "What's the irregularity?"
    "What's the flaw?",
    "What are the irregularities?",
    "What are the flaws?",
    "Can you identify the anomaly?",
    "Could you point out the defect?",
    "Do you see any anomalies?",
    "Do you notice any defects?",
    "What's considered anomalous?",
    "What's deemed as a defect?",
    "Can you detect any anomalies?",
    "Can you spot any defects?",
    "What constitutes an anomaly?",
    "What falls under the category of defects?",
    "What's regarded as an anomaly?",
    "What's categorized as a defect?",
    "What anomalies are present?",
    "What defects have been identified?",
    "What kind of anomalies are we looking at?",
    "What types of defects are visible?",
]

detail_questions_cn = [
    "异常部分是什么？",
    "缺陷是什么？",
    "有哪些异常？",
    "有哪些缺陷？",
    "你为什么这么认为？",
    "有什么不规则之处吗？",
    "有什么缺陷吗？",
    "有哪些不规则之处？",
    "有哪些缺陷？",
    "你能识别出异常吗？",
    "你能指出缺陷吗？",
    "你看到了任何异常吗？",
    "你注意到了任何缺陷吗？",
    "什么被认为是异常的？",
    "什么被视为缺陷？",
    "你能检测出任何异常吗？",
    "你能发现任何缺陷吗？",
    "什么构成了异常？",
    "什么属于缺陷的范畴？",
    "什么被看作是异常？",
    "什么被归类为缺陷？",
    "有什么异常存在吗？",
    "有哪些缺陷被发现了？",
    "有哪些类型的缺陷是可见的？"
]

detail1_questions = [
    "Could you give me some advice?",
    "Could you give me some treatment for this kind of leaf disease?",
    "Could you give me something to prevent this leaf disease?",
    "Could you give me some ways to deal with such anomalies?",
    "Could you give me some advice on the disease of these leaves?",
    "Can you give me some good ways?",
    "Can you give some suggestions about the anomalies of this object?",
    "Can you give some prevention and treatment methods?",
    "Can you give some good methods for this type of defect?",
    "Can you give some good methods for these abnormal phenomena?",
    "Can you give some good suggestions to prevent this kind of abnormality?",
    "How can we reduce the probability of such anomalies?"
    "How to prevent the occurrence of such abnormalities?",
    "How to improve such anomalies?",
    "How to improve this disease?",
    "How to treat this kind of disease?",
    "How to manage the category of defects??",
    "What can be done to prevent such anomalies?",
    "What can be done to manage this type of anomaly?",
    "What can be done to control such diseases?",
    "What can be done to prevent such diseases?",
    "What can be done to reduce the occurrence of such diseases?",
    "What can be done to treat these abnormalities?",
    "What can be done to deal with such anomalies?",
    "What are some good ways to deal with this kind of anomaly?",
    "What are the ways to deal with these diseases?",
    "Is there any way to deal with the category of defects?",
]

detail1_questions_cn = [
    "能不能给我一些意见？",
    "能不能给我一些对这类叶子疾病的防治方法？",
    "能不能给我一些防治这类叶子疾病的方法？",
    "你能给我一些应对这类异常发生的方法吗？",
    "你能给我一些建议针对这类叶子的疾病吗？",
    "你能给我一些好的方法吗？",
    "能不能给出一些建议针对这个物体的异常？",
    "能不能给出一些防治的方法？",
    "能不能给出一些好的方法对于这类型的缺陷？",
    "你能给一些好的方法对于这些异常现象吗？",
    "你能给一些好的建议预防这类异常吗？",
    "怎么样可以降低这类异常发生的概率？",
    "怎么样去预防这类异常的发生？",
    "怎么样去改善这类异常？",
    "怎么样去改善这类病害？",
    "怎么样去治理这类病害？",
    "怎么样去治理这类缺陷？",
    "怎么样做可以预防这类异常？",
    "做什么可以治理这类异常？",
    "怎么样做可以治理这类病害？",
    "怎么样做可以预防这类病害？",
    "怎么样做可以降低这类病害的发生？",
    "做什么可以治疗这类异常？",
    "有什么可以应对这类异常？",
    "有什么好的办法可以应对这类异常？",
    "有哪些方法可以应对这类病害？",
    "有什么方法可以应对这类缺陷？",
]
PCB_names = [
    'printed wiring board',
    'circuit card',
    'electronic board',
    'PCB assembly',
    'circuitry panel',
    'circuit substrate',
    'wiring substrate',
    'circuit laminate',
    'electronic substrate',
    'board with printed circuits',
    'PCB layout',
    'circuit interconnect board',
    'electrical board',
    'integrated circuit board',
    'printed wiring assembly',
    'PCB design',
    'printed electronic board',
    'conductor board',
    'printed circuitry card',
    'electronics motherboard'
]

PCB_names_cn = [
    "印刷线路板",
    "电路板",
    "PCB组件",
    "电路板面",
    "电路基板",
    "布线基板",
    "电路层压板",
    "电子基板",
    "带印刷电路的板子",
    "PCB",
    "电路互连板",
    "电气板",
    "集成电路板",
    "印刷布线组件",
    "印刷电路板",
    "导体板",
    "印刷电路卡",
    "电子主板"
]

Road_names = [
    'pavement',
    'concrete',
    'road',
    'sideroad',
    'concrete road',
    'roadway',
    'surface',
    'street',
    'wall',
    'concrete surfacce',
    'concrete wall'
]

Road_names_cn = [
    "人行道",
    "混凝土",
    "道路",
    "小路",
    "混凝土路",
    "道路",
    "路面",
    "水泥路表面",
    "墙面"
]


def get_class_name(name):
    global PCB_names
    if name == 'candle':
        return 'candles'
    elif 'macaroni' in name:
        return 'macaronis'
    elif 'pcb' in name:
        return random.choice(PCB_names)
    elif name == 'road':
        return random.choice(Road_names)
    else:
        return name.replace('_', " ")


# TODO: Finish This
def get_class_name_cn(name):
    global PCB_names
    if name in Chinese_class_names.keys():
        return random.choice(Chinese_class_names[name])
    elif 'pcb' in name:
        return random.choice(PCB_names_cn)
    elif name == 'road':
        return random.choice(Road_names_cn)
    else:
        return random.choice(['元件', '元器件', '样品', '样本'])


def format_position(position):
    ret = ""
    for i in range(len(position)):
        if i == 0:
            ret += position[i]
        else:
            if i != len(position) - 1:
                ret += ", "
                ret += position[i]
            else:
                ret += " and " + position[i]

    return ret


def format_position_cn(position):
    ret = ""
    for i in range(len(position)):
        if i == 0:
            ret += Chinese_position[position[i]]
        else:
            if i != len(position) - 1:
                ret += "，"
                ret += Chinese_position[position[i]]
            else:
                ret += "和" + Chinese_position[position[i]]

    return ret


class SupervisedDataset(Dataset):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.resize = transforms.Resize(
            (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
        )

        self.norm_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

        self.paths = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if ('masks' not in file_path and 'ground_truth' not in file_path) and (
                        'png' in file_path or 'JPG' in file_path or 'JPEG' in file_path or 'jpg' in file_path):
                    self.paths.append(file_path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        img_path = self.paths[index]
        img = self.resize(Image.open(img_path).convert('RGB'))
        if 'mvtec_anomaly_detection' in img_path or 'visa' in img_path or 'mvtec_loco_anomaly_detection' in img_path:
            class_name = img_path.split('/')[-4]
        elif 'road' in img_path:
            class_name = 'road'

        centers = []

        if 'good' not in img_path:
            if 'mvtec_anomaly_detection' in img_path:
                mask_path = img_path.replace('test', 'ground_truth')
                mask_path = mask_path.replace('.png', '_mask.png')
            elif 'visa' in img_path:
                mask_path = img_path.replace('test', 'ground_truth')
                mask_path = mask_path.replace('.JPG', '.png')
            elif 'mvtec_loco_anomaly_detection' in img_path:
                mask_path = img_path.replace('test', 'ground_truth')
                mask_path = mask_path.replace('.png', '/000.png')
            elif 'crack_road' in img_path:
                mask_path = img_path.replace('images', 'masks')
                mask_path = mask_path.replace('.jpg', '.png')
            elif 'iva_road' in img_path:
                mask_path = img_path.replace('images', 'masks')
                mask_path = mask_path.replace('.jpg', '.png')
            elif 'Magnetic-Tile-Defect' in img_path:
                mask_path = img_path.replace('Imgs', 'masks')
                mask_path = mask_path.replace('.jpg', '.png')
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (224, 224))
            centers = find_contours(mask)
            mask = transforms.ToTensor()(mask)
        else:
            mask = torch.zeros((1, 224, 224))

        img = self.norm_transform(img)

        position = []
        if len(centers) > 0:
            for center in centers:
                center_y = center[0] / 224
                center_x = center[1] / 224

                if center_x <= 1 / 3 and center_y <= 1 / 3:
                    position.append('top left')
                elif center_x <= 1 / 3 and center_y > 1 / 3 and center_y <= 2 / 3:
                    position.append('top')
                elif center_x <= 1 / 3 and center_y > 2 / 3:
                    position.append('top right')

                elif center_x <= 2 / 3 and center_y <= 1 / 3:
                    position.append('left')
                elif center_x <= 2 / 3 and center_y > 1 / 3 and center_y <= 2 / 3:
                    position.append('center')
                elif center_x <= 2 / 3 and center_y > 2 / 3:
                    position.append('right')

                elif center_y <= 1 / 3:
                    position.append('bottom left')
                elif center_y > 1 / 3 and center_y <= 2 / 3:
                    position.append('bottom')
                elif center_y > 2 / 3:
                    position.append('bottom right')

            position = list(set(position))

        conversation = []

        Use_chinese = random.randint(0, 1) == 0

        r = random.randint(0, 2)
        if not Use_chinese:
            if r == 0 and 'mvtec_loco_anomaly_detection' not in img_path:
                conversation.append({"from": "human", "value": random.choice(class_questions)})
                if class_name not in MULTI_CLASS:
                    conversation.append(
                        {"from": "gpt", "value": random.choice(single_answers).format(get_class_name(class_name))})
                else:
                    conversation.append(
                        {"from": "gpt", "value": random.choice(multi_answers).format(get_class_name(class_name))})
        else:
            if r == 0 and 'mvtec_loco_anomaly_detection' not in img_path:
                conversation.append({"from": "human", "value": random.choice(class_questions_cn)})
                if class_name not in MULTI_CLASS:
                    conversation.append({"from": "gpt", "value": random.choice(single_answers_cn).format(
                        get_class_name_cn(class_name))})
                else:
                    conversation.append(
                        {"from": "gpt", "value": random.choice(multi_answers_cn).format(get_class_name_cn(class_name))})

        if not Use_chinese:
            conversation.append({"from": "human", "value": random.choice(anomaly_questions)})
            if len(centers) == 0:
                conversation.append(
                    {"from": "gpt", "value": random.choice(normal_answers).format(get_class_name(class_name))})
            if len(centers) == 1:
                abnormal_describe = "Yes, there is {} in the image, at the {} of the image.".format(
                    random.choice(['an anomaly', 'a defect']), position[0])
                conversation.append({"from": "gpt", "value": abnormal_describe})
            elif len(centers) > 1:
                if class_name != 'road':
                    abnormal_describe = "Yes, there are {} anomalies in the image, they are at the {} of the image.".format(
                        str(len(centers)), format_position(position))
                else:
                    abnormal_describe = "Yes, there is {} in the image.".format(
                        random.choice(['an anomaly', 'a defect']))
                conversation.append({"from": "gpt", "value": abnormal_describe})
        else:
            conversation.append({"from": "human", "value": random.choice(anomaly_questions_cn)})
            if len(centers) == 0:
                conversation.append(
                    {"from": "gpt", "value": random.choice(normal_answers_cn).format(get_class_name_cn(class_name))})
            if len(centers) == 1:
                abnormal_describe = "是的，图中有1个{}， 在图像的{}。".format(random.choice(['异常', '缺陷']),
                                                                           format_position_cn(position))
                conversation.append({"from": "gpt", "value": abnormal_describe})
            elif len(centers) > 1:
                if class_name != 'road':
                    abnormal_describe = "是的，图中有{}个异常, 在图像的{}.".format(str(len(centers)),
                                                                                  format_position_cn(position))
                else:
                    abnormal_describe = "是的，图中有1个异常。"
                conversation.append({"from": "gpt", "value": abnormal_describe})

        if 'good' not in img_path and 'mvtec_anomaly_detection' in img_path:
            anomaly_detail = img_path.split('/')[-2]
            if not Use_chinese:
                detail_question = random.choice(detail_questions)
                detail1_question = random.choice(detail1_questions)  # 假设有 detail1_questions 列表
            else:
                detail_question = random.choice(detail_questions_cn)
                detail1_question = random.choice(detail1_questions_cn)  # 假设有 detail1_questions_cn 列表

                # 随机选择一个问题
            selected_question = random.choice([detail_question, detail1_question])
            conversation.append({"from": "human", "value": selected_question})

            detail_answer = ''
            detail_answer_cn = ''
            detail1_answer = ''
            detail1_answer_cn = ''
            be = 'is' if len(centers) == 1 else 'are'
            num = 'a' if len(centers) == 1 else str(len(centers))
            p = format_position(position)
            p_cn = format_position_cn(position)
            s = '' if len(centers) == 1 else 's'
            es = '' if len(centers) == 1 else 'es'

            flag = 1

            if class_name == 'apple':
                if anomaly_detail == 'apple_black_rot':
                    detail_answer = 'The anomaly in this image belongs to apple black rot and is located to the {} of this leaf.'.format(
                        p)
                    detail_answer_cn = '图中的异常属于苹果黑腐病，异常位置在这个叶子的{}。'.format(p_cn)
                    detail1_answer = 'When cleaning the orchard, pay attention to the pruning branches, diseased fallen leaves, and diseased dead fruit out of the orchard to burn.'
                    detail1_answer_cn = '清理果园时注意将剪除枝、病落叶、带病僵果等带出果园集中烧毁。'
                elif anomaly_detail == 'apple_cedar_rust':
                    detail_answer = 'The anomaly in this image belongs to apple cedar rust and is located to the {} of this leaf.'.format(
                        p)
                    detail_answer_cn = '图中的异常属于苹果雪松锈病，异常位置在这个叶子的{}。'.format(p_cn)
                    detail1_answer = 'Before large-scale precipitation, spray a fungicide to prevent rust bacteria from infecting during rainfall.'
                    detail1_answer_cn = '在大范围降水前，喷一次杀菌剂，防止锈病菌在降雨过程中侵染。'
                elif anomaly_detail == 'apple_scab':
                    detail_answer = 'The anomaly in this image belongs to apple black scab and is located to the {} of this leaf.'.format(
                        p)
                    detail_answer_cn = '图中的异常属于苹果黑星病，异常位置在这个叶子的{}。'.format(p_cn)
                    detail1_answer = 'After the symptoms of the disease appear, spray in time with rain resistant, long-lasting bactericide.'
                    detail1_answer_cn = '在病害症状出现后，及时喷施耐雨水冲刷，药效持久的杀菌剂。'
                else:
                    flag = 0
                    conversation = conversation[:-1]
            elif class_name == 'corn':
                if anomaly_detail == 'corn_northern_leaf_blight':
                    detail_answer = 'The anomaly in this image belongs to corn northern leaf blight and is located to the {} of the {}.'.format(
                        p, get_class_name(class_name))
                    detail_answer_cn = '图中的异常属于玉米北叶枯病，异常位置在这个{}的{}。'.format(
                        get_class_name_cn(class_name), p_cn)
                    detail1_answer = 'After the disease occurs, the disease residues in the field should be removed in time to reduce the overwintering and spread of germs.'
                    detail1_answer_cn = '病害发生后，及时清除田间的病残体，减少病菌的越冬和传播。'
                elif anomaly_detail == 'corn_gray_leaf_spot':
                    detail_answer = 'The anomaly in this image belongs to corn gray leaf spot and is located to the {} of the {}.'.format(
                        p, get_class_name(class_name))
                    detail_answer_cn = '图中的异常属于玉米灰斑病，异常位置在这个{}的{}。'.format(
                        get_class_name_cn(class_name), p_cn)
                    detail1_answer = 'Strengthen field management, timely drainage after rain to prevent moisture retention.'
                    detail1_answer_cn = '加强田间管理，雨后及时排水，防止湿气滞留。'
                else:
                    flag = 0
                    conversation = conversation[:-1]
            elif class_name == 'tomato':
                if anomaly_detail == 'tomato_septoria_leaf_spot':
                    detail_answer = 'The anomaly in this image belongs to tomato septoria leaf spot and is located to the {} of the {}.'.format(
                        p, get_class_name(class_name))
                    detail_answer_cn = '图中的异常属于番茄斑枯病，异常位置在这个{}的{}。'.format(
                        get_class_name_cn(class_name), p_cn)
                    detail1_answer = 'Strengthen cultivation management, increase the application of phosphorus and potassium fertilizer, improve disease resistance.'
                    detail1_answer_cn = '加强栽培管理，增施磷、钾肥，提高抗病性。'
                else:
                    flag = 0
                    conversation = conversation[:-1]
            elif class_name == 'grape':
                if anomaly_detail == 'grape_black_measles':
                    detail_answer = 'The anomaly in this image belongs to grape black measles and is located to the {} of the {}.'.format(
                        p, get_class_name(class_name))
                    detail_answer_cn = '图中的异常属于葡萄轮斑病，异常位置在这个{}的{}。'.format(
                        get_class_name_cn(class_name), p_cn)
                    detail1_answer = ' Apply more organic fertilizer and control the amount of nitrogen fertilizer.'
                    detail1_answer_cn = '多施有机肥，控制氮肥用量。'
                elif anomaly_detail == 'grape_black_rot':
                    detail_answer = 'The anomaly in this image belongs to grape black rot and is located to the {} of the {}.'.format(
                        p, get_class_name(class_name))
                    detail_answer_cn = '图中的异常属于葡萄黑腐病，异常位置在这个{}的{}。'.format(
                        get_class_name_cn(class_name), p_cn)
                    detail1_answer = 'Timely drainage, increase the application of organic fertilizer.'
                    detail1_answer_cn = '及时排水，增施有机肥。'
                else:
                    flag = 0
                    conversation = conversation[:-1]
            elif class_name == 'palm':
                if anomaly_detail == 'palm_bug':
                    detail_answer = 'The anomaly in this image belongs to palm bug and is located to the {} of the {}.'.format(
                        p, get_class_name(class_name))
                    detail_answer_cn = '图中的异常属于棕榈病虫害，异常位置在这个{}的{}。'.format(
                        get_class_name_cn(class_name), p_cn)
                    detail1_answer = 'Remove diseased plants, clean up diseased leaves in time, and reduce the breeding environment of pests and diseases.'
                    detail1_answer_cn = '铲除病株，及时清理病叶，减少病虫害的滋生环境。'
                else:
                    flag = 0
                    conversation = conversation[:-1]
            elif class_name == 'peach':
                if anomaly_detail == 'peach_bacterial_spot':
                    detail_answer = 'The anomaly in this image belongs to peach bacterial spot and is located to the {} of the {}.'.format(
                        p, get_class_name(class_name))
                    detail_answer_cn = '图中的异常属于桃子细菌性斑病，异常位置在这个{}的{}。'.format(
                        get_class_name_cn(class_name), p_cn)
                    detail1_answer = 'Peach orchard pay attention to drainage, increase the application of biological organic fertilizer.'
                    detail1_answer_cn = '桃园注意排水，增施生物有机肥。'
                else:
                    flag = 0
                    conversation = conversation[:-1]
            elif class_name == 'pepper':
                if anomaly_detail == 'pepper_bacterial_spot':
                    detail_answer = 'The anomaly in this image belongs to pepper bacterial spot and is located to the {} of the {}.'.format(
                        p, get_class_name(class_name))
                    detail_answer_cn = '图中的异常属于辣椒细菌性叶斑病，异常位置在这个{}的{}。'.format(
                        get_class_name_cn(class_name), p_cn)
                    detail1_answer = 'Pay attention to clean the field, remove the sick remains in time after harvest or deep dive in time.'
                    detail1_answer_cn = '注意清洁田园，收获后及时清除病残体或及时深翻。'
                else:
                    flag = 0
                    conversation = conversation[:-1]
            elif class_name == 'potato':
                if anomaly_detail == 'potato_early_blight':
                    detail_answer = 'The anomaly in this image belongs to potato early blight and is located to the {} of the {}.'.format(
                        p, get_class_name(class_name))
                    detail_answer_cn = '图中的异常属于马铃薯早疫病，异常位置在这个{}的{}。'.format(
                        get_class_name_cn(class_name), p_cn)
                    detail1_answer = 'Early and disease-resistant varieties should be selected for timely irrigation.'
                    detail1_answer_cn = '选用早熟抗病品种，及时灌溉。'
                else:
                    flag = 0
                    conversation = conversation[:-1]
            elif class_name == 'strawberry':
                if anomaly_detail == 'strawberry_leaf_scorch':
                    detail_answer = 'The anomaly in this image belongs to strawberry leaf scorch and is located to the {} of the {}.'.format(
                        p, get_class_name(class_name))
                    detail_answer_cn = '图中的异常属于草莓叶枯病，异常位置在这个{}的{}。'.format(
                        get_class_name_cn(class_name), p_cn)
                    detail1_answer = 'Do not apply nitrogen fertilizer, increase the application of phosphorus, potassium fertilizer, appropriate irrigation.'
                    detail1_answer_cn = '不偏施氮肥，增施磷、钾肥，适量灌水。'
                else:
                    flag = 0
                    conversation = conversation[:-1]

            if flag:
                if not Use_chinese:
                    if selected_question in detail_questions:
                        conversation.append({"from": "gpt", "value": detail_answer})
                    else:
                        conversation.append({"from": "gpt", "value": detail1_answer})
                else:
                    if selected_question in detail_questions_cn:
                        conversation.append({"from": "gpt", "value": detail_answer_cn})
                    else:
                        conversation.append({"from": "gpt", "value": detail1_answer_cn})

        if not Use_chinese:
            if r == 1 and 'mvtec_loco_anomaly_detection' not in img_path:
                conversation.append({"from": "human", "value": random.choice(class_questions)})
                if class_name not in MULTI_CLASS:
                    conversation.append(
                        {"from": "gpt", "value": random.choice(single_answers).format(get_class_name(class_name))})
                else:
                    conversation.append(
                        {"from": "gpt", "value": random.choice(multi_answers).format(get_class_name(class_name))})
        else:
            if r == 1 and 'mvtec_loco_anomaly_detection' not in img_path:
                conversation.append({"from": "human", "value": random.choice(class_questions_cn)})
                if class_name not in MULTI_CLASS:
                    conversation.append({"from": "gpt", "value": random.choice(single_answers_cn).format(
                        get_class_name_cn(class_name))})
                else:
                    conversation.append(
                        {"from": "gpt", "value": random.choice(multi_answers_cn).format(get_class_name_cn(class_name))})

        print(img_path, conversation)

        return img, conversation, class_name, mask, img_path

    def collate(self, instances):
        images = []
        texts = []
        class_names = []
        masks = []
        img_paths = []
        for instance in instances:
            images.append(instance[0])
            texts.append(instance[1])
            class_names.append(instance[2])
            masks.append(instance[3])
            if 'mvtec_anomaly_detection' in instance[4] or 'visa' in instance[4] or 'mvtec_loco_anomaly_detection' in \
                    instance[4]:
                img_paths.append(instance[4])

        return dict(
            images=images,
            texts=texts,
            class_names=class_names,
            masks=masks,
            img_paths=img_paths
        )