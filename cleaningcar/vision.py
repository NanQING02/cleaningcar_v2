NORMALIZED_COORD_TOLERANCE = 1e-3


def _clip_normalized(value):
    return max(0.0, min(1.0, float(value)))


def _is_normalized(points, tolerance=NORMALIZED_COORD_TOLERANCE):
    if not points:
        return False
    return all(
        -tolerance <= float(p[0]) <= 1.0 + tolerance
        and -tolerance <= float(p[1]) <= 1.0 + tolerance
        for p in points
    )


def scale_polygon(points, width, height):
    if not points:
        return []
    if _is_normalized(points):
        return [(_clip_normalized(x) * width, _clip_normalized(y) * height) for x, y in points]
    return [(float(x), float(y)) for x, y in points]


def scale_point(point, width, height):
    if not point:
        return (0.0, 0.0)
    x, y = point
    tolerance = NORMALIZED_COORD_TOLERANCE
    if -tolerance <= float(x) <= 1.0 + tolerance and -tolerance <= float(y) <= 1.0 + tolerance:
        return _clip_normalized(x) * width, _clip_normalized(y) * height
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


def _point_on_segment(point, start, end, tolerance=1e-6):
    px, py = map(float, point)
    x1, y1 = map(float, start)
    x2, y2 = map(float, end)
    cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
    if abs(cross) > tolerance:
        return False
    return (
        min(x1, x2) - tolerance <= px <= max(x1, x2) + tolerance
        and min(y1, y2) - tolerance <= py <= max(y1, y2) + tolerance
    )


def point_in_polygon(point, polygon):
    if point is None or polygon is None or len(polygon) < 3:
        return False
    px, py = map(float, point)
    inside = False
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        if _point_on_segment((px, py), start, end):
            return True
        x1, y1 = map(float, start)
        x2, y2 = map(float, end)
        if (y1 > py) == (y2 > py):
            continue
        intersection_x = (x2 - x1) * (py - y1) / (y2 - y1) + x1
        if px < intersection_x:
            inside = not inside
    return inside


def _segments_intersect(a, b, c, d):
    def orientation(p, q, r):
        value = (
            (float(q[1]) - float(p[1])) * (float(r[0]) - float(q[0]))
            - (float(q[0]) - float(p[0])) * (float(r[1]) - float(q[1]))
        )
        if abs(value) <= 1e-6:
            return 0
        return 1 if value > 0 else 2

    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)
    if o1 != o2 and o3 != o4:
        return True
    return (
        (o1 == 0 and _point_on_segment(c, a, b))
        or (o2 == 0 and _point_on_segment(d, a, b))
        or (o3 == 0 and _point_on_segment(a, c, d))
        or (o4 == 0 and _point_on_segment(b, c, d))
    )


def box_intersects_polygon(box, polygon):
    if box is None or polygon is None or len(polygon) < 3:
        return False
    x1, y1, x2, y2 = map(float, box)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    corners = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
    center = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
    if any(point_in_polygon(point, polygon) for point in (*corners, center)):
        return True
    if any(point_in_box(point, (x1, y1, x2, y2)) for point in polygon):
        return True
    box_edges = tuple((corners[index], corners[(index + 1) % 4]) for index in range(4))
    polygon_edges = tuple(
        (polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    )
    return any(
        _segments_intersect(box_start, box_end, polygon_start, polygon_end)
        for box_start, box_end in box_edges
        for polygon_start, polygon_end in polygon_edges
    )
