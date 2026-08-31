import re

import cv2
import numpy as np

REG_MAX = 16
PROJECT = np.arange(REG_MAX, dtype=np.float32)
LICENSE_CLASS = 8
PLATE_SIZE = (94, 24)
CLASS_NAMES = [
    'car',
    'blue truck',
    'yellow truck',
    'dump truck',
    'wuxiao',
    'wheel',
    'cleaning table',
    'manual',
    'license',
]
CLASS_COLORS = {
    'vehicle': (0, 220, 0),
    'plate': (0, 255, 255),
    'wheel': (255, 255, 0),
    'water': (0, 160, 255),
}
VEHICLE_LABEL_CN = {
    'car': '小汽车',
    'blue truck': '蓝色卡车',
    'yellow truck': '黄色卡车',
    'dump truck': '渣土车',
    'wuxiao': '五小工程车',
}
CLEANING_LABEL_CN = {
    'cleaning table': '清洗台清洗',
    'manual': '人工清洗',
}
CLASS_NAME_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
CLASS_ALIAS_TO_ID = {name.lower(): idx for idx, name in enumerate(CLASS_NAMES)}
VEHICLE_CLASS_IDS = {0, 1, 2, 3, 4}
WATER_CLASS_IDS = {6, 7}
CLASS_THRESH = {
    0: 0.40,
    1: 0.40,
    2: 0.40,
    3: 0.40,
    4: 0.40,
    5: 0.25,
    6: 0.25,
    7: 0.25,
    8: 0.25
}
PLATE_CAR_LINK_IOU = 0.02
CAR_PLATE_CACHE_TTL = 60
REPORT_MIN_FRAMES = 3
DIRECTION_MAP = {
    (1, 1): (5, '正向前出'),
    (1, -1): (6, '正向后出'),
    (-1, 1): (7, '反向前出'),
    (-1, -1): (8, '反向后出'),
}
PLATE_DECODE_CHARS = (
    "#"
    "京沪津渝冀晋蒙辽吉黑"
    "苏浙皖闽赣鲁豫鄂湘粤"
    "桂琼川贵云藏陕甘青宁新"
    "学警港澳挂使领民航危"
    "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
    "险品"
)
LPR_CHARS = list(PLATE_DECODE_CHARS)
LPR_BLANK = 0
PLATE_COLOR_NAMES = ['黑色', '蓝色', '绿色', '白色', '黄色']
PROVINCE_CHARS = ''.join([
    '京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
    '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
    '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁', '新',
])
PLATE_SUFFIX_CHARS = '学警港澳挂使领民航危险品'
PLATE_ALLOWED_CHARS = set(PLATE_DECODE_CHARS[1:])
PLATE_LETTERS = set('ABCDEFGHJKLMNPQRSTUVWXYZ')
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
PLATE_REGEX = re.compile(rf'^[{PROVINCE_CHARS}][A-Z][A-Z0-9]{{5}}$')
PLATE_REGEX_NE = re.compile(rf'^[{PROVINCE_CHARS}][A-Z][A-Z0-9]{{6}}$')
ALNUM = set('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
FFMPEG_PIX_BYTES = {
    'bgr24': 3,
    'rgb24': 3,
}
WHEEL_CLASS_NAME_TO_CLEAN_VALUE = {
    '0-25': 1,
    '25-50': 2,
    '50-75': 3,
    '75-100': 4,
}
WHEEL_SIDE_TO_PHOTO_TYPE = {
    'left': '4',
    'right': '5',
}


def select_box_color(label_name):
    if label_name in VEHICLE_LABEL_CN:
        return CLASS_COLORS['vehicle']
    if label_name == 'license':
        return CLASS_COLORS['plate']
    if label_name == 'wheel':
        return CLASS_COLORS['wheel']
    if label_name in ('cleaning table', 'manual'):
        return CLASS_COLORS['water']
    return (0, 255, 0)


def localize_vehicle(label):
    return VEHICLE_LABEL_CN.get(label, label)


def localize_cleaning(label):
    return CLEANING_LABEL_CN.get(label, label)
