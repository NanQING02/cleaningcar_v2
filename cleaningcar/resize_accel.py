import cv2

_rga_resize = None
_RGA_READY = False


def resize_backend_name():
    return "cv2"


def resize_bgr(image, target_size, interpolation=cv2.INTER_LINEAR):
    if image is None:
        return None
    tw, th = int(target_size[0]), int(target_size[1])
    if tw <= 0 or th <= 0:
        raise ValueError(f"invalid target_size: {target_size}")
    h, w = image.shape[:2]
    if w == tw and h == th:
        return image
    return cv2.resize(image, (tw, th), interpolation=interpolation)
