"""aligner.py
MediaPipe Face Mesh based face alignment helper.

Provides align_face_mediapipe(image, bbox, output_size=160)
Returns an RGB numpy array of shape (output_size, output_size, 3)
or None if alignment fails (caller should fallback to center-crop).
"""
import math
import cv2
import numpy as np
try:
    import mediapipe as mp
except Exception:
    mp = None

mp_face_mesh = mp.solutions.face_mesh if mp is not None else None


def _get_eye_landmarks(landmarks):
    # MediaPipe FaceMesh common indexes for eye outer landmarks
    # left eye outer ~ 33, right eye outer ~ 263 (approx)
    try:
        left = landmarks[33]
        right = landmarks[263]
        return (left.x, left.y), (right.x, right.y)
    except Exception:
        return None, None


def align_face_mediapipe(image, bbox, output_size=160, pad_ratio=0.25):
    """
    Align face in `image` according to landmarks detected by MediaPipe Face Mesh.

    Args:
        image: BGR or RGB numpy array (we'll convert to RGB internally if needed).
        bbox: tuple (x1, y1, x2, y2) in image coordinates.
        output_size: final square output size (pixels).
        pad_ratio: how much extra margin to include around bbox (fraction of max(w,h)).

    Returns:
        aligned_rgb: numpy array shape (output_size, output_size, 3), dtype=uint8
        or None if MP is not available or landmarks not found.
    """
    if mp_face_mesh is None:
        return None

    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))

    if x2 <= x1 or y2 <= y1:
        return None

    # expand bbox a bit to include context
    bw = x2 - x1
    bh = y2 - y1
    pad = int(max(bw, bh) * pad_ratio)
    xa = max(0, x1 - pad)
    ya = max(0, y1 - pad)
    xb = min(w, x2 + pad)
    yb = min(h, y2 + pad)

    face = image[ya:yb, xa:xb].copy()
    if face.size == 0:
        return None

    # convert to RGB for mediapipe
    if face.shape[2] == 3:
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    else:
        face_rgb = face

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm:
        res = fm.process(face_rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        left_rel, right_rel = _get_eye_landmarks(lm)
        if left_rel is None or right_rel is None:
            return None

        fh, fw = face.shape[:2]
        left_pt = np.array([left_rel[0] * fw, left_rel[1] * fh])
        right_pt = np.array([right_rel[0] * fw, right_rel[1] * fh])

        # compute the angle to rotate so the eyes are horizontal
        dy = right_pt[1] - left_pt[1]
        dx = right_pt[0] - left_pt[0]
        angle = math.degrees(math.atan2(dy, dx))

        # rotate around center of face crop
        center = (fw // 2, fh // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(face_rgb, M, (fw, fh), flags=cv2.INTER_CUBIC)

        # after rotation, re-compute eye center
        # approximate center between the two eye points
        left_pt_r = np.dot(M, np.array([left_pt[0], left_pt[1], 1.0]))
        right_pt_r = np.dot(M, np.array([right_pt[0], right_pt[1], 1.0]))
        eyes_center = ((left_pt_r[0] + right_pt_r[0]) / 2.0, (left_pt_r[1] + right_pt_r[1]) / 2.0)

        # place a square crop centered at eyes_center with size = max(face dim) * 1.2
        crop_size = int(max(fw, fh) * 1.0)
        half = crop_size // 2
        cx = int(eyes_center[0])
        cy = int(eyes_center[1])
        x0 = max(0, cx - half)
        y0 = max(0, cy - half)
        x1c = min(fw, cx + half)
        y1c = min(fh, cy + half)
        crop = rotated[y0:y1c, x0:x1c]
        if crop.size == 0:
            return None

        aligned = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_CUBIC)
        # aligned is currently RGB
        return aligned


def center_crop_resize(image, bbox, output_size=160, pad_ratio=0.25):
    # fallback: crop bbox (with pad) and resize to output
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    bw = x2 - x1
    bh = y2 - y1
    pad = int(max(bw, bh) * pad_ratio)
    xa = max(0, int(x1 - pad))
    ya = max(0, int(y1 - pad))
    xb = min(w, int(x2 + pad))
    yb = min(h, int(y2 + pad))
    crop = image[ya:yb, xa:xb]
    if crop.size == 0:
        return None
    # ensure RGB
    if crop.shape[2] == 3:
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    else:
        crop_rgb = crop
    return cv2.resize(crop_rgb, (output_size, output_size), interpolation=cv2.INTER_CUBIC)
