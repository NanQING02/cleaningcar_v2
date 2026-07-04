import numpy as np

# Backup snapshot from 2026-03-29 (legacy IoU greedy tracker).
# Quick rollback: set logic.vehicle_tracker_impl = "legacy".
class ByteTrackTracker:
    def __init__(self, iou_thresh=0.3, max_age=60, center_gate_ratio=0.0):
        self.iou_thresh = float(iou_thresh)
        self.max_age = int(max_age)
        self.center_gate_ratio = max(0.0, float(center_gate_ratio))
        self.tracks = {}
        self.next_id = 1

    def update(self, frame_idx, detections):
        to_prune = []
        for tid, tr in self.tracks.items():
            last_seen = tr.get('last_seen', frame_idx)
            if frame_idx - last_seen > self.max_age:
                to_prune.append(tid)
        for tid in to_prune:
            self.tracks.pop(tid, None)
        track_ids = list(self.tracks.keys())
        det_boxes = [np.array(det['box'], dtype=float) for det in detections]
        assigned_tracks = {}
        assigned_dets = set()
        if track_ids and det_boxes:
            iou_matrix = np.zeros((len(track_ids), len(det_boxes)), dtype=np.float32)
            for ti, tid in enumerate(track_ids):
                tbox = np.array(self.tracks[tid]['box'], dtype=float)
                xb1, yb1, xb2, yb2 = tbox
                for di, dbox in enumerate(det_boxes):
                    x1, y1, x2, y2 = dbox
                    xA = max(xb1, x1)
                    yA = max(yb1, y1)
                    xB = min(xb2, x2)
                    yB = min(yb2, y2)
                    interW = max(0.0, xB - xA)
                    interH = max(0.0, yB - yA)
                    inter = interW * interH
                    if inter <= 0.0:
                        iou_matrix[ti, di] = 0.0
                    else:
                        areaA = max(1.0, (xb2 - xb1) * (yb2 - yb1))
                        areaB = max(1.0, (x2 - x1) * (y2 - y1))
                        iou_matrix[ti, di] = inter / (areaA + areaB - inter)
            while True:
                ti, di = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                max_iou = iou_matrix[ti, di]
                if max_iou < self.iou_thresh:
                    if self.center_gate_ratio <= 0.0:
                        break
                    tid = track_ids[ti]
                    tbox = np.array(self.tracks[tid]['box'], dtype=float)
                    dbox = det_boxes[di]
                    tcx = 0.5 * (tbox[0] + tbox[2])
                    tcy = 0.5 * (tbox[1] + tbox[3])
                    dcx = 0.5 * (dbox[0] + dbox[2])
                    dcy = 0.5 * (dbox[1] + dbox[3])
                    diag = ((tbox[2] - tbox[0]) ** 2 + (tbox[3] - tbox[1]) ** 2) ** 0.5
                    if diag <= 0.0:
                        break
                    center_dist = ((tcx - dcx) ** 2 + (tcy - dcy) ** 2) ** 0.5
                    if center_dist > self.center_gate_ratio * diag:
                        break
                tid = track_ids[ti]
                assigned_tracks[tid] = di
                assigned_dets.add(di)
                iou_matrix[ti, :] = -1.0
                iou_matrix[:, di] = -1.0
                if np.max(iou_matrix) <= 0.0:
                    break
        for tid, det_idx in assigned_tracks.items():
            det = detections[det_idx]
            tr = self.tracks.get(tid)
            if not tr:
                continue
            tr['box'] = det['box']
            tr['cls'] = det['cls']
            tr['score'] = float(det.get('score', 0.0))
            tr['last_seen'] = frame_idx
            tr['age'] = 0
            tr['hits'] = tr.get('hits', 0) + 1
        for di, det in enumerate(detections):
            if di in assigned_dets:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {
                'box': det['box'],
                'cls': det['cls'],
                'score': float(det.get('score', 0.0)),
                'last_seen': frame_idx,
                'age': 0,
                'hits': 1,
            }
            assigned_tracks[tid] = di
        to_delete = []
        for tid, track in self.tracks.items():
            if tid in assigned_tracks:
                continue
            track['age'] = track.get('age', 0) + 1
            if track['age'] > self.max_age:
                to_delete.append(tid)
        for tid in to_delete:
            self.tracks.pop(tid, None)
        assignments = [-1] * len(detections)
        for tid, det_idx in assigned_tracks.items():
            if 0 <= det_idx < len(assignments):
                assignments[det_idx] = tid
        return assignments

    def get_active_tracks(self):
        return {tid: tr for tid, tr in self.tracks.items() if tr.get('age', 0) == 0}


VehicleTracker = ByteTrackTracker
