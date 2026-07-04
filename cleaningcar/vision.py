def _is_normalized(points):
    if not points:
        return False
    return all(0.0 <= p[0] <= 1.0 and 0.0 <= p[1] <= 1.0 for p in points)


def scale_polygon(points, width, height):
    if not points:
        return []
    if _is_normalized(points):
        return [(float(x) * width, float(y) * height) for x, y in points]
    return [(float(x), float(y)) for x, y in points]


def scale_point(point, width, height):
    if not point:
        return (0.0, 0.0)
    x, y = point
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        return float(x) * width, float(y) * height
    return float(x), float(y)


def get_anchor_point(box, offset_ratio=0.0):
    if not box:
        return None
    x1, y1, x2, y2 = box
    cx = 0.5 * (x1 + x2)
    height = max(1.0, (y2 - y1))
    offset = float(offset_ratio)
    if offset < 0.0:
        offset = 0.0
    elif offset > 0.95:
        offset = 0.95
    cy = y2 - offset * height
    return (float(cx), float(cy))


def box_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    inter = interW * interH
    if inter <= 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]) + 1e-6
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]) + 1e-6
    return inter / (areaA + areaB - inter)


def point_in_box(pt, box):
    if box is None or pt is None:
        return False
    x, y = pt
    return (box[0] <= x <= box[2]) and (box[1] <= y <= box[3])
