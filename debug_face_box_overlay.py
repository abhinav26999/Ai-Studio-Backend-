import cv2
import numpy as np
from huggingface_hub import hf_hub_download
import sys
import os

def debug_face_overlay(image_path: str, output_path: str = "debug_face_box.jpg"):
    """
    Renders anatomical face bounding box and all 5 YuNet facial landmarks onto the image.
    Prints geometric diagnostic metrics: D_em, Center_x, Forehead Top, Chin Bottom, Left/Right Bounds.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot load image {image_path}")
        return

    h, w = img.shape[:2]
    yunet_model = hf_hub_download(repo_id="opencv/face_detection_yunet", filename="face_detection_yunet_2023mar.onnx")
    detector = cv2.FaceDetectorYN.create(yunet_model, "", (w, h), score_threshold=0.65)
    
    faces = detector.detect(img)[1]
    if faces is None or len(faces) == 0:
        print(f"No face detected in {image_path} with score >= 0.65")
        return

    face = max(faces, key=lambda f: f[2] * f[3])
    score = float(face[-1])
    raw_box = face[0:4].astype(int)
    
    # 5 landmarks in OpenCV YuNet order:
    # 0: Right Eye (subject's right, viewer's left)
    # 1: Left Eye (subject's left, viewer's right)
    # 2: Nose Tip
    # 3: Right Mouth Corner
    # 4: Left Mouth Corner
    landmarks = face[4:14].reshape(5, 2).astype(float)
    re, le, nose, rm, lm = landmarks

    eye_center = (re + le) / 2.0
    mouth_center = (rm + lm) / 2.0
    d_em = float(np.linalg.norm(mouth_center - eye_center))
    face_center_x = float((eye_center[0] + mouth_center[0]) / 2.0)

    top = max(0.0, float(eye_center[1] - 0.70 * d_em))
    bottom = min(float(h), float(mouth_center[1] + 0.50 * d_em))
    left = max(0.0, float(face_center_x - 0.80 * d_em))
    right = min(float(w), float(face_center_x + 0.80 * d_em))

    bx, by, bw, bh = left, top, right - left, bottom - top

    print("=" * 60)
    print(" YuNet Landmark & Anatomical Box Diagnostics")
    print("=" * 60)
    print(f"Image Resolution    : {w}x{h} px")
    print(f"Detection Score     : {score:.4f}")
    print(f"Right Eye (Subject) : ({re[0]:.1f}, {re[1]:.1f})")
    print(f"Left Eye (Subject)  : ({le[0]:.1f}, {le[1]:.1f})")
    print(f"Nose Tip            : ({nose[0]:.1f}, {nose[1]:.1f})")
    print(f"Right Mouth Corner  : ({rm[0]:.1f}, {rm[1]:.1f})")
    print(f"Left Mouth Corner   : ({lm[0]:.1f}, {lm[1]:.1f})")
    print("-" * 60)
    print(f"D_em (Eye-Mouth Dist): {d_em:.2f} px")
    print(f"Center_x            : {face_center_x:.2f} px")
    print(f"Forehead Top (y1)   : {top:.2f} px")
    print(f"Chin Bottom (y2)    : {bottom:.2f} px")
    print(f"Left Bound (x1)     : {left:.2f} px")
    print(f"Right Bound (x2)    : {right:.2f} px")
    print(f"Anatomical (w, h)   : {bw:.2f} x {bh:.2f} px (Aspect: {bw/bh:.2f})")
    print("=" * 60)

    # Draw overlay visual
    vis = img.copy()
    
    # Draw raw YuNet detection in light gray
    cv2.rectangle(vis, (raw_box[0], raw_box[1]), (raw_box[0] + raw_box[2], raw_box[1] + raw_box[3]), (180, 180, 180), 1)
    
    # Draw Anatomical Box in Bright Green
    cv2.rectangle(vis, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
    
    # Draw 1:1 Square Box in Cyan
    side = int(round(max(bw, bh)))
    cx = bx + bw / 2.0
    cy = by + bh / 2.0
    sq_x1 = int(round(cx - side / 2.0))
    sq_y1 = int(round(cy - side / 2.0))
    cv2.rectangle(vis, (sq_x1, sq_y1), (sq_x1 + side, sq_y1 + side), (255, 255, 0), 2)

    # Draw landmarks with color-coded dots
    colors = [
        (0, 0, 255),    # Red - Right Eye
        (0, 165, 255),  # Orange - Left Eye
        (0, 255, 255),  # Yellow - Nose
        (255, 0, 255),  # Magenta - Right Mouth
        (255, 0, 0)     # Blue - Left Mouth
    ]
    labels = ["R-Eye", "L-Eye", "Nose", "R-Mouth", "L-Mouth"]
    for pt, c, lbl in zip(landmarks, colors, labels):
        ix, iy = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(vis, (ix, iy), 5, c, -1)
        cv2.putText(vis, lbl, (ix + 6, iy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw legend
    cv2.putText(vis, f"D_em={d_em:.1f}px Score={score:.2f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(vis, "Green: Anatomical Box | Cyan: 1:1 Wav2Lip Square", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)

    cv2.imwrite(output_path, vis)
    print(f"Debug overlay saved to: {output_path}")

if __name__ == "__main__":
    test_img = sys.argv[1] if len(sys.argv) > 1 else "/Users/aeologic/.gemini/antigravity-ide/brain/f6c467dc-d7d9-45ee-b58f-0a0c99abc2d1/scratch/latest_input_photo.jpg"
    out_img = sys.argv[2] if len(sys.argv) > 2 else "/Users/aeologic/.gemini/antigravity-ide/brain/f6c467dc-d7d9-45ee-b58f-0a0c99abc2d1/scratch/debug_face_box_overlay.jpg"
    debug_face_overlay(test_img, out_img)
