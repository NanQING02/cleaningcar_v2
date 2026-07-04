import numpy as np

_STATE_NEW = 0
_STATE_TRACKED = 1
_STATE_LOST = 2
_STATE_REMOVED = 3


def _clip01(value):
    return float(np.clip(float(value), 0.0, 1.0))


def _tlbr_iou(box_a, box_b):
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(1e-6, (float(box_a[2]) - float(box_a[0])) * (float(box_a[3]) - float(box_a[1])))
    area_b = max(1e-6, (float(box_b[2]) - float(box_b[0])) * (float(box_b[3]) - float(box_b[1])))
    return float(inter / max(1e-6, area_a + area_b - inter))


class _KalmanFilter:
    def __init__(self):
        self._std_weight_position = 1.0 / 20.0
        self._std_weight_velocity = 1.0 / 160.0
        self._motion_mat = np.eye(8, dtype=np.float32)
        for i in range(4):
            self._motion_mat[i, i + 4] = 1.0
        self._update_mat = np.eye(4, 8, dtype=np.float32)

    def initiate(self, measurement_xyah):
        mean_pos = np.asarray(measurement_xyah, dtype=np.float32)
        mean_vel = np.zeros(4, dtype=np.float32)
        mean = np.concatenate([mean_pos, mean_vel], axis=0)
        h = max(1.0, float(measurement_xyah[3]))
        std = np.asarray(
            [
                2.0 * self._std_weight_position * h,
                2.0 * self._std_weight_position * h,
                1e-2,
                2.0 * self._std_weight_position * h,
                10.0 * self._std_weight_velocity * h,
                10.0 * self._std_weight_velocity * h,
                1e-5,
                10.0 * self._std_weight_velocity * h,
            ],
            dtype=np.float32,
        )
        covariance = np.diag(std * std).astype(np.float32)
        return mean, covariance

    def predict(self, mean, covariance):
        h = max(1.0, float(mean[3]))
        std_pos = np.asarray(
            [
                self._std_weight_position * h,
                self._std_weight_position * h,
                1e-2,
                self._std_weight_position * h,
            ],
            dtype=np.float32,
        )
        std_vel = np.asarray(
            [
                self._std_weight_velocity * h,
                self._std_weight_velocity * h,
                1e-5,
                self._std_weight_velocity * h,
            ],
            dtype=np.float32,
        )
        motion_cov = np.diag(np.concatenate([std_pos, std_vel]) ** 2).astype(np.float32)
        mean = self._motion_mat.dot(mean)
        covariance = self._motion_mat.dot(covariance).dot(self._motion_mat.T) + motion_cov
        return mean.astype(np.float32), covariance.astype(np.float32)

    def project(self, mean, covariance):
        h = max(1.0, float(mean[3]))
        std = np.asarray(
            [
                self._std_weight_position * h,
                self._std_weight_position * h,
                1e-1,
                self._std_weight_position * h,
            ],
            dtype=np.float32,
        )
        innovation_cov = np.diag(std * std).astype(np.float32)
        mean_proj = self._update_mat.dot(mean)
        cov_proj = self._update_mat.dot(covariance).dot(self._update_mat.T) + innovation_cov
        return mean_proj.astype(np.float32), cov_proj.astype(np.float32)

    def update(self, mean, covariance, measurement_xyah):
        projected_mean, projected_cov = self.project(mean, covariance)
        cross_cov = covariance.dot(self._update_mat.T)
        try:
            k_gain = np.linalg.solve(projected_cov, cross_cov.T).T
        except np.linalg.LinAlgError:
            k_gain = cross_cov.dot(np.linalg.pinv(projected_cov))
        innovation = np.asarray(measurement_xyah, dtype=np.float32) - projected_mean
        new_mean = mean + k_gain.dot(innovation)
        new_cov = covariance - k_gain.dot(projected_cov).dot(k_gain.T)
        return new_mean.astype(np.float32), new_cov.astype(np.float32)


class _STrack:
    def __init__(self, tlwh, score, cls_id=-1, det_index=-1):
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.score = float(score)
        self.cls_id = cls_id
        self.det_index = int(det_index)
        self.track_id = 0
        self.state = _STATE_NEW
        self.is_activated = False
        self.mean = None
        self.covariance = None
        self.frame_id = 0
        self.start_frame = 0
        self.tracklet_len = 0

    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] = max(1.0, float(ret[2] * ret[3]))
        ret[3] = max(1.0, float(ret[3]))
        ret[0] = ret[0] - ret[2] * 0.5
        ret[1] = ret[1] - ret[3] * 0.5
        return ret.astype(np.float32)

    @property
    def tlbr(self):
        tlwh = self.tlwh
        return np.asarray([tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]], dtype=np.float32)

    def to_xyah(self):
        tlwh = self.tlwh
        h = max(1.0, float(tlwh[3]))
        cx = float(tlwh[0] + tlwh[2] * 0.5)
        cy = float(tlwh[1] + tlwh[3] * 0.5)
        a = float(tlwh[2] / h)
        return np.asarray([cx, cy, a, h], dtype=np.float32)

    def activate(self, kalman_filter, frame_id, track_id):
        self.track_id = int(track_id)
        self.mean, self.covariance = kalman_filter.initiate(self.to_xyah())
        self.state = _STATE_TRACKED
        self.is_activated = True
        self.frame_id = int(frame_id)
        self.start_frame = int(frame_id)
        self.tracklet_len = 0

    def re_activate(self, kalman_filter, new_track, frame_id, new_id=False, track_id=None):
        self.mean, self.covariance = kalman_filter.update(self.mean, self.covariance, new_track.to_xyah())
        self._tlwh = new_track._tlwh.copy()
        self.score = new_track.score
        self.cls_id = new_track.cls_id
        self.state = _STATE_TRACKED
        self.is_activated = True
        self.frame_id = int(frame_id)
        self.tracklet_len = 0
        if new_id and track_id is not None:
            self.track_id = int(track_id)

    def update(self, kalman_filter, new_track, frame_id):
        self.frame_id = int(frame_id)
        self.tracklet_len += 1
        self.mean, self.covariance = kalman_filter.update(self.mean, self.covariance, new_track.to_xyah())
        self._tlwh = new_track._tlwh.copy()
        self.score = new_track.score
        self.cls_id = new_track.cls_id
        self.state = _STATE_TRACKED
        self.is_activated = True

    def predict(self, kalman_filter):
        if self.mean is None or self.covariance is None:
            return
        mean = self.mean.copy()
        if self.state != _STATE_TRACKED:
            mean[7] = 0.0
        self.mean, self.covariance = kalman_filter.predict(mean, self.covariance)

    def mark_lost(self):
        self.state = _STATE_LOST

    def mark_removed(self):
        self.state = _STATE_REMOVED

    def end_frame(self):
        return int(self.frame_id)


class ByteTrackTracker:
    def __init__(
        self,
        iou_thresh=0.3,
        max_age=60,
        center_gate_ratio=0.0,
        track_thresh=0.5,
        high_thresh=0.6,
        low_thresh=0.1,
    ):
        self.iou_thresh = _clip01(iou_thresh)
        self.max_age = max(1, int(max_age))
        self.center_gate_ratio = max(0.0, float(center_gate_ratio))
        self.track_thresh = _clip01(track_thresh)
        self.high_thresh = max(self.track_thresh, _clip01(high_thresh))
        self.low_thresh = min(self.track_thresh, _clip01(low_thresh))
        primary_iou_thresh = min(self.iou_thresh, 0.20)
        self.primary_cost_thresh = float(1.0 - max(0.05, min(0.95, primary_iou_thresh)))
        self.secondary_cost_thresh = float(1.0 - max(self.iou_thresh, 0.50))
        self.unconfirmed_cost_thresh = float(1.0 - max(self.iou_thresh, 0.30))

        self.frame_id = 0
        self.next_id = 1
        self.kalman_filter = _KalmanFilter()
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []

    def _normalize_frame_id(self, frame_idx):
        try:
            idx = int(frame_idx)
        except Exception:
            idx = self.frame_id + 1
        if idx <= self.frame_id:
            idx = self.frame_id + 1
        self.frame_id = idx

    @staticmethod
    def _tlbr_to_tlwh(tlbr):
        x1, y1, x2, y2 = [float(v) for v in tlbr]
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        return np.asarray([x1, y1, w, h], dtype=np.float32)

    def _build_detection_track(self, det, det_index):
        box = det.get("box", [0, 0, 1, 1])
        score = float(det.get("score", 0.0))
        cls_id = det.get("cls", -1)
        tlwh = self._tlbr_to_tlwh(box)
        return _STrack(tlwh, score, cls_id=cls_id, det_index=det_index)

    @staticmethod
    def _joint_stracks(strack_a, strack_b):
        exists = set()
        out = []
        for trk in list(strack_a) + list(strack_b):
            tid = int(getattr(trk, "track_id", 0))
            if tid <= 0 or tid in exists:
                continue
            exists.add(tid)
            out.append(trk)
        return out

    @staticmethod
    def _sub_stracks(strack_a, strack_b):
        b_ids = {int(getattr(t, "track_id", 0)) for t in strack_b}
        return [t for t in strack_a if int(getattr(t, "track_id", 0)) not in b_ids]

    def _iou_distance(self, tracks, detections, use_center_gate=True):
        n = len(tracks)
        m = len(detections)
        if n == 0 or m == 0:
            return np.zeros((n, m), dtype=np.float32)
        dists = np.zeros((n, m), dtype=np.float32)
        for i, trk in enumerate(tracks):
            t_box = trk.tlbr
            t_cx = 0.5 * float(t_box[0] + t_box[2])
            t_cy = 0.5 * float(t_box[1] + t_box[3])
            diag = max(1e-6, float(np.hypot(t_box[2] - t_box[0], t_box[3] - t_box[1])))
            for j, det in enumerate(detections):
                d_box = det.tlbr
                if use_center_gate and self.center_gate_ratio > 0.0:
                    d_cx = 0.5 * float(d_box[0] + d_box[2])
                    d_cy = 0.5 * float(d_box[1] + d_box[3])
                    center_dist = float(np.hypot(t_cx - d_cx, t_cy - d_cy))
                    if center_dist > self.center_gate_ratio * diag:
                        dists[i, j] = 1e5
                        continue
                iou = _tlbr_iou(t_box, d_box)
                dists[i, j] = 1.0 - float(iou)
        return dists

    @staticmethod
    def _hungarian(cost_matrix):
        cost = np.asarray(cost_matrix, dtype=np.float64)
        n_rows, n_cols = cost.shape
        if n_rows == 0 or n_cols == 0:
            return np.asarray([], dtype=int), np.asarray([], dtype=int)

        transposed = False
        if n_rows > n_cols:
            cost = cost.T
            n_rows, n_cols = cost.shape
            transposed = True

        u = np.zeros(n_rows + 1, dtype=np.float64)
        v = np.zeros(n_cols + 1, dtype=np.float64)
        p = np.zeros(n_cols + 1, dtype=np.int32)
        way = np.zeros(n_cols + 1, dtype=np.int32)

        for i in range(1, n_rows + 1):
            p[0] = i
            j0 = 0
            minv = np.full(n_cols + 1, np.inf, dtype=np.float64)
            used = np.zeros(n_cols + 1, dtype=bool)
            while True:
                used[j0] = True
                i0 = p[j0]
                delta = np.inf
                j1 = 0
                for j in range(1, n_cols + 1):
                    if used[j]:
                        continue
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
                if not np.isfinite(delta):
                    break
                for j in range(0, n_cols + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta
                j0 = j1
                if p[j0] == 0:
                    break
            while True:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break

        row_ind = []
        col_ind = []
        for j in range(1, n_cols + 1):
            if p[j] == 0:
                continue
            row_ind.append(p[j] - 1)
            col_ind.append(j - 1)

        row_ind = np.asarray(row_ind, dtype=int)
        col_ind = np.asarray(col_ind, dtype=int)
        if transposed:
            return col_ind, row_ind
        return row_ind, col_ind

    def _linear_assignment(self, cost_matrix, thresh):
        if cost_matrix.size == 0:
            rows = list(range(cost_matrix.shape[0]))
            cols = list(range(cost_matrix.shape[1]))
            return [], rows, cols

        big = 1e6
        cost = np.asarray(cost_matrix, dtype=np.float64).copy()
        cost[~np.isfinite(cost)] = big
        row_ind, col_ind = self._hungarian(cost)

        matched_rows = set()
        matched_cols = set()
        matches = []
        for r, c in zip(row_ind.tolist(), col_ind.tolist()):
            if cost[r, c] <= thresh:
                matches.append((r, c))
                matched_rows.add(r)
                matched_cols.add(c)

        unmatched_rows = [i for i in range(cost.shape[0]) if i not in matched_rows]
        unmatched_cols = [j for j in range(cost.shape[1]) if j not in matched_cols]
        return matches, unmatched_rows, unmatched_cols

    def _remove_duplicate_stracks(self, strack_a, strack_b):
        if not strack_a or not strack_b:
            return strack_a, strack_b
        dists = self._iou_distance(strack_a, strack_b, use_center_gate=False)
        pairs = np.where(dists < 0.15)
        dup_a = set()
        dup_b = set()
        for ia, ib in zip(pairs[0].tolist(), pairs[1].tolist()):
            life_a = strack_a[ia].frame_id - strack_a[ia].start_frame
            life_b = strack_b[ib].frame_id - strack_b[ib].start_frame
            if life_a > life_b:
                dup_b.add(ib)
            else:
                dup_a.add(ia)
        filtered_a = [t for i, t in enumerate(strack_a) if i not in dup_a]
        filtered_b = [t for i, t in enumerate(strack_b) if i not in dup_b]
        return filtered_a, filtered_b

    def _split_detections(self, detections):
        high = []
        low = []
        for det_index, det in enumerate(detections):
            score = float(det.get("score", 0.0))
            track = self._build_detection_track(det, det_index)
            if score >= self.track_thresh:
                high.append(track)
            elif score >= self.low_thresh:
                low.append(track)
        return high, low

    def update(self, frame_idx, detections):
        if detections is None:
            detections = []
        self._normalize_frame_id(frame_idx)
        assignments = [-1] * len(detections)

        detections_high, detections_low = self._split_detections(detections)
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        unconfirmed = []
        tracked_stracks = []
        for trk in self.tracked_stracks:
            if not trk.is_activated:
                unconfirmed.append(trk)
            else:
                tracked_stracks.append(trk)

        strack_pool = self._joint_stracks(tracked_stracks, self.lost_stracks)
        for trk in strack_pool:
            trk.predict(self.kalman_filter)

        dists = self._iou_distance(strack_pool, detections_high)
        matches, u_track, u_detection_high = self._linear_assignment(dists, self.primary_cost_thresh)
        for i_track, i_det in matches:
            track = strack_pool[i_track]
            det = detections_high[i_det]
            if track.state == _STATE_TRACKED:
                track.update(self.kalman_filter, det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(self.kalman_filter, det, self.frame_id, new_id=False)
                refind_stracks.append(track)
            if 0 <= det.det_index < len(assignments):
                assignments[det.det_index] = track.track_id

        unmatched_high = [detections_high[i] for i in u_detection_high]
        remained_tracks = []
        for idx in u_track:
            trk = strack_pool[idx]
            if trk.state == _STATE_TRACKED:
                remained_tracks.append(trk)

        dists_low = self._iou_distance(remained_tracks, detections_low)
        matches_low, u_track_low, u_detection_low = self._linear_assignment(dists_low, self.secondary_cost_thresh)
        for i_track, i_det in matches_low:
            track = remained_tracks[i_track]
            det = detections_low[i_det]
            if track.state == _STATE_TRACKED:
                track.update(self.kalman_filter, det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(self.kalman_filter, det, self.frame_id, new_id=False)
                refind_stracks.append(track)
            if 0 <= det.det_index < len(assignments):
                assignments[det.det_index] = track.track_id

        for idx in u_track_low:
            track = remained_tracks[idx]
            if track.state != _STATE_LOST:
                track.mark_lost()
                lost_stracks.append(track)

        dists_unconfirmed = self._iou_distance(unconfirmed, unmatched_high)
        matches_u, u_unconfirmed, u_remaining_high = self._linear_assignment(
            dists_unconfirmed, self.unconfirmed_cost_thresh
        )
        for i_track, i_det in matches_u:
            track = unconfirmed[i_track]
            det = unmatched_high[i_det]
            track.update(self.kalman_filter, det, self.frame_id)
            activated_stracks.append(track)
            if 0 <= det.det_index < len(assignments):
                assignments[det.det_index] = track.track_id

        for idx in u_unconfirmed:
            track = unconfirmed[idx]
            track.mark_removed()
            removed_stracks.append(track)

        for idx in u_remaining_high:
            det = unmatched_high[idx]
            if det.score < self.high_thresh:
                continue
            det.activate(self.kalman_filter, self.frame_id, self.next_id)
            self.next_id += 1
            activated_stracks.append(det)
            if 0 <= det.det_index < len(assignments):
                assignments[det.det_index] = det.track_id

        for track in self.lost_stracks:
            if self.frame_id - track.end_frame() > self.max_age:
                track.mark_removed()
                removed_stracks.append(track)

        tracked_now = [t for t in self.tracked_stracks if t.state == _STATE_TRACKED]
        self.tracked_stracks = self._joint_stracks(tracked_now, activated_stracks)
        self.tracked_stracks = self._joint_stracks(self.tracked_stracks, refind_stracks)

        self.lost_stracks = self._sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks = self._joint_stracks(self.lost_stracks, lost_stracks)
        self.lost_stracks = self._sub_stracks(self.lost_stracks, removed_stracks)

        self.removed_stracks = self._joint_stracks(self.removed_stracks, removed_stracks)
        self.tracked_stracks, self.lost_stracks = self._remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )

        return assignments

    def get_active_tracks(self):
        active = {}
        for track in self.tracked_stracks:
            if track.state != _STATE_TRACKED or not track.is_activated:
                continue
            active[track.track_id] = {
                "box": [float(v) for v in track.tlbr.tolist()],
                "cls": track.cls_id,
                "score": float(track.score),
            }
        return active


class LegacyIoUTracker:
    def __init__(self, iou_thresh=0.3, max_age=60, center_gate_ratio=0.0):
        self.iou_thresh = float(iou_thresh)
        self.max_age = int(max_age)
        self.center_gate_ratio = max(0.0, float(center_gate_ratio))
        self.tracks = {}
        self.next_id = 1

    def update(self, frame_idx, detections):
        to_prune = []
        for tid, tr in self.tracks.items():
            last_seen = tr.get("last_seen", frame_idx)
            if frame_idx - last_seen > self.max_age:
                to_prune.append(tid)
        for tid in to_prune:
            self.tracks.pop(tid, None)
        track_ids = list(self.tracks.keys())
        det_boxes = [np.array(det["box"], dtype=float) for det in detections]
        assigned_tracks = {}
        assigned_dets = set()
        if track_ids and det_boxes:
            iou_matrix = np.zeros((len(track_ids), len(det_boxes)), dtype=np.float32)
            for ti, tid in enumerate(track_ids):
                tbox = np.array(self.tracks[tid]["box"], dtype=float)
                xb1, yb1, xb2, yb2 = tbox
                for di, dbox in enumerate(det_boxes):
                    x1, y1, x2, y2 = dbox
                    x_a = max(xb1, x1)
                    y_a = max(yb1, y1)
                    x_b = min(xb2, x2)
                    y_b = min(yb2, y2)
                    inter_w = max(0.0, x_b - x_a)
                    inter_h = max(0.0, y_b - y_a)
                    inter = inter_w * inter_h
                    if inter <= 0.0:
                        iou_matrix[ti, di] = 0.0
                    else:
                        area_a = max(1.0, (xb2 - xb1) * (yb2 - yb1))
                        area_b = max(1.0, (x2 - x1) * (y2 - y1))
                        iou_matrix[ti, di] = inter / (area_a + area_b - inter)
            while True:
                ti, di = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                max_iou = iou_matrix[ti, di]
                if max_iou < self.iou_thresh:
                    if self.center_gate_ratio <= 0.0:
                        break
                    tid = track_ids[ti]
                    tbox = np.array(self.tracks[tid]["box"], dtype=float)
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
            tr["box"] = det["box"]
            tr["cls"] = det["cls"]
            tr["score"] = float(det.get("score", 0.0))
            tr["last_seen"] = frame_idx
            tr["age"] = 0
            tr["hits"] = tr.get("hits", 0) + 1
        for di, det in enumerate(detections):
            if di in assigned_dets:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {
                "box": det["box"],
                "cls": det["cls"],
                "score": float(det.get("score", 0.0)),
                "last_seen": frame_idx,
                "age": 0,
                "hits": 1,
            }
            assigned_tracks[tid] = di
        to_delete = []
        for tid, track in self.tracks.items():
            if tid in assigned_tracks:
                continue
            track["age"] = track.get("age", 0) + 1
            if track["age"] > self.max_age:
                to_delete.append(tid)
        for tid in to_delete:
            self.tracks.pop(tid, None)
        assignments = [-1] * len(detections)
        for tid, det_idx in assigned_tracks.items():
            if 0 <= det_idx < len(assignments):
                assignments[det_idx] = tid
        return assignments

    def get_active_tracks(self):
        return {tid: tr for tid, tr in self.tracks.items() if tr.get("age", 0) == 0}


class VehicleTracker:
    def __init__(self, iou_thresh=0.3, max_age=60, center_gate_ratio=0.0, tracker_impl="bytetrack"):
        impl = str(tracker_impl or "bytetrack").strip().lower()
        if impl in {"legacy", "legacy_iou", "simple_iou", "iou"}:
            self.impl = "legacy"
            self._tracker = LegacyIoUTracker(
                iou_thresh=iou_thresh,
                max_age=max_age,
                center_gate_ratio=center_gate_ratio,
            )
        else:
            self.impl = "bytetrack"
            self._tracker = ByteTrackTracker(
                iou_thresh=iou_thresh,
                max_age=max_age,
                center_gate_ratio=center_gate_ratio,
            )

    def update(self, frame_idx, detections):
        return self._tracker.update(frame_idx, detections)

    def get_active_tracks(self):
        return self._tracker.get_active_tracks()
