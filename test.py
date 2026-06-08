import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


# ============================================================
# 1. 路径配置
# ============================================================

VISIBLE_DIR = Path(
    r"F:\BaiduNetdiskDownload\MMFW-UAV-suc\images\train"
)

INFRARED_DIR = Path(
    r"F:\BaiduNetdiskDownload\MMFW-UAV-suc\image\train"
)

LABEL_DIR = Path(
    r"F:\BaiduNetdiskDownload\MMFW-UAV-suc\labels\train"
)

OUTPUT_CSV = Path(
    r"F:\BaiduNetdiskDownload\MMFW-UAV-suc\uav_roi_registration_residual_results_500.csv"
)

OUTPUT_SUMMARY = Path(
    r"F:\BaiduNetdiskDownload\MMFW-UAV-suc\uav_roi_registration_residual_summary_500.csv"
)

SAVE_VIS = True

VIS_DIR = Path(
    r"F:\BaiduNetdiskDownload\MMFW-UAV-suc\uav_roi_registration_full_image_vis_500"
)

MAX_VIS_SAVE = 100

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]


# ============================================================
# 2. 评估参数
# ============================================================

# 只评估前 500 张配对图片
NUM_EVAL_IMAGES = 500

# 在原始 UAV bbox 基础上扩大 ROI
ROI_EXPAND_RATIO = 2.5

# 最小 ROI 尺寸，避免目标太小导致边缘过少
MIN_ROI_SIZE = 96

# 最大 ROI 尺寸，避免引入过多背景
MAX_ROI_SIZE = 256

# 如果一张图有多个目标，是否全部评估
EVALUATE_ALL_BOXES = True


# ============================================================
# 3. 中文路径图像读取与保存
# ============================================================

def imread_gray_unicode(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Failed to decode image, file may be corrupted: {path}")

    return img


def imwrite_unicode(path: Path, img):
    path.parent.mkdir(parents=True, exist_ok=True)

    ext = path.suffix.lower()
    if ext == "":
        ext = ".jpg"
        path = path.with_suffix(ext)

    success, encoded_img = cv2.imencode(ext, img)
    if not success:
        raise ValueError(f"Failed to encode image: {path}")

    encoded_img.tofile(str(path))


# ============================================================
# 4. 文件匹配
# ============================================================

def collect_images(folder: Path):
    image_dict = {}

    for ext in IMG_EXTS:
        for p in folder.glob(f"*{ext}"):
            image_dict[p.stem] = p
        for p in folder.glob(f"*{ext.upper()}"):
            image_dict[p.stem] = p

    return image_dict


def safe_sort_key(x):
    x = str(x)
    if x.isdigit():
        return (0, int(x))
    return (1, x)


def check_folders():
    if not VISIBLE_DIR.exists():
        raise FileNotFoundError(f"Visible image directory not found: {VISIBLE_DIR}")

    if not INFRARED_DIR.exists():
        raise FileNotFoundError(f"Infrared image directory not found: {INFRARED_DIR}")

    if not LABEL_DIR.exists():
        raise FileNotFoundError(f"Label directory not found: {LABEL_DIR}")

    visible_dict = collect_images(VISIBLE_DIR)
    infrared_dict = collect_images(INFRARED_DIR)
    label_files = list(LABEL_DIR.glob("*.txt"))

    print("========== Dataset Check ==========")
    print(f"Visible dir:  {VISIBLE_DIR}")
    print(f"Infrared dir: {INFRARED_DIR}")
    print(f"Label dir:    {LABEL_DIR}")
    print(f"Visible image count:  {len(visible_dict)}")
    print(f"Infrared image count: {len(infrared_dict)}")
    print(f"Label count:          {len(label_files)}")
    print(f"Will evaluate first:  {NUM_EVAL_IMAGES} paired images")
    print("===================================")

    return visible_dict, infrared_dict


# ============================================================
# 5. YOLO 标签读取与 ROI 生成
# ============================================================

def read_yolo_label(label_path: Path):
    boxes = []

    if not label_path.exists():
        return boxes

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            continue

        cls_id = int(float(parts[0]))
        xc = float(parts[1])
        yc = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])

        boxes.append({
            "class_id": cls_id,
            "xc": xc,
            "yc": yc,
            "w": w,
            "h": h
        })

    return boxes


def yolo_box_to_pixel_roi(
    box,
    image_width,
    image_height,
    expand_ratio=2.5,
    min_size=96,
    max_size=256
):
    xc = box["xc"] * image_width
    yc = box["yc"] * image_height
    bw = box["w"] * image_width
    bh = box["h"] * image_height

    roi_w = max(bw * expand_ratio, min_size)
    roi_h = max(bh * expand_ratio, min_size)

    roi_w = min(roi_w, max_size)
    roi_h = min(roi_h, max_size)

    x1 = int(round(xc - roi_w / 2))
    y1 = int(round(yc - roi_h / 2))
    x2 = int(round(xc + roi_w / 2))
    y2 = int(round(yc + roi_h / 2))

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_width, x2)
    y2 = min(image_height, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def crop_roi(img, roi):
    x1, y1, x2, y2 = roi
    return img[y1:y2, x1:x2]


# ============================================================
# 6. ROI 内配准误差估计
# ============================================================

def resize_to_same_size(vis_gray, ir_gray):
    h_ir, w_ir = ir_gray.shape[:2]
    h_vis, w_vis = vis_gray.shape[:2]

    if (h_vis, w_vis) != (h_ir, w_ir):
        vis_gray = cv2.resize(vis_gray, (w_ir, h_ir), interpolation=cv2.INTER_LINEAR)

    return vis_gray, ir_gray


def preprocess_for_cross_modal(gray):
    gray = gray.astype(np.uint8)

    h, w = gray.shape[:2]
    if min(h, w) < 64:
        scale = 64 / min(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    med = np.median(blurred)
    lower = int(max(0, 0.66 * med))
    upper = int(min(255, 1.33 * med))

    if upper <= lower:
        lower = 30
        upper = 120

    edges = cv2.Canny(blurred, lower, upper)

    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    return edges


def estimate_residual_shift_phase_correlation(vis_edge, ir_edge):
    if vis_edge.shape != ir_edge.shape:
        vis_edge = cv2.resize(vis_edge, (ir_edge.shape[1], ir_edge.shape[0]))

    vis_f = vis_edge.astype(np.float32)
    ir_f = ir_edge.astype(np.float32)

    h, w = ir_f.shape[:2]
    if h < 8 or w < 8:
        return np.nan, np.nan, np.nan

    hann = cv2.createHanningWindow((w, h), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(vis_f, ir_f, hann)

    dx, dy = shift
    return float(dx), float(dy), float(response)


def estimate_residual_shift_ecc(vis_edge, ir_edge, max_iter=100, eps=1e-5):
    if vis_edge.shape != ir_edge.shape:
        vis_edge = cv2.resize(vis_edge, (ir_edge.shape[1], ir_edge.shape[0]))

    vis_f = vis_edge.astype(np.float32) / 255.0
    ir_f = ir_edge.astype(np.float32) / 255.0

    h, w = ir_f.shape[:2]
    if h < 8 or w < 8:
        return None

    warp_matrix = np.eye(2, 3, dtype=np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        max_iter,
        eps
    )

    try:
        cc, warp_matrix = cv2.findTransformECC(
            templateImage=ir_f,
            inputImage=vis_f,
            warpMatrix=warp_matrix,
            motionType=cv2.MOTION_TRANSLATION,
            criteria=criteria,
            inputMask=None,
            gaussFiltSize=3
        )

        dx = float(warp_matrix[0, 2])
        dy = float(warp_matrix[1, 2])

        return dx, dy, float(cc)

    except cv2.error:
        return None


def edge_overlap_score(vis_edge, ir_edge):
    if vis_edge.shape != ir_edge.shape:
        vis_edge = cv2.resize(vis_edge, (ir_edge.shape[1], ir_edge.shape[0]))

    vis_bin = vis_edge > 0
    ir_bin = ir_edge > 0

    intersection = np.logical_and(vis_bin, ir_bin).sum()
    union = np.logical_or(vis_bin, ir_bin).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


# ============================================================
# 7. 整图可视化
# ============================================================

def make_full_edge_overlay(vis_gray, ir_gray, roi):
    """
    生成整图边缘叠加图，但只突出 UAV ROI 区域。
    绿色：可见光边缘
    红色：红外边缘
    蓝框：评估用 UAV ROI
    """
    vis_full_edge = preprocess_for_cross_modal(vis_gray)
    ir_full_edge = preprocess_for_cross_modal(ir_gray)

    if vis_full_edge.shape != ir_full_edge.shape:
        vis_full_edge = cv2.resize(
            vis_full_edge,
            (ir_full_edge.shape[1], ir_full_edge.shape[0])
        )

    h, w = ir_gray.shape[:2]

    vis_edge_color = np.zeros((h, w, 3), dtype=np.uint8)
    ir_edge_color = np.zeros((h, w, 3), dtype=np.uint8)

    vis_edge_color[vis_full_edge > 0] = (0, 255, 0)
    ir_edge_color[ir_full_edge > 0] = (0, 0, 255)

    overlay = cv2.addWeighted(vis_edge_color, 0.6, ir_edge_color, 0.6, 0)

    x1, y1, x2, y2 = roi
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return overlay


def draw_roi_on_gray(gray, roi, color=(0, 255, 255)):
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    x1, y1, x2, y2 = roi
    cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
    return bgr


def save_full_image_visualization(
    vis_gray,
    ir_gray,
    roi,
    stem,
    box_id,
    dx,
    dy,
    residual_error,
    save_dir
):
    """
    保存整图可视化：
    左：整张可见光图 + UAV ROI 框
    中：整张红外图 + UAV ROI 框
    右：整张边缘叠加图 + UAV ROI 框
    但评估误差只来自 UAV ROI 内。
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    h, w = ir_gray.shape[:2]

    vis_show = draw_roi_on_gray(vis_gray, roi, color=(0, 255, 255))
    ir_show = draw_roi_on_gray(ir_gray, roi, color=(0, 255, 255))
    overlay = make_full_edge_overlay(vis_gray, ir_gray, roi)

    text = f"ROI-only residual={residual_error:.2f}px, dx={dx:.2f}, dy={dy:.2f}"
    cv2.putText(
        overlay,
        text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # 为了保存图片大小适中，统一缩放宽度
    target_h = 360
    scale = target_h / h
    target_w = int(w * scale)

    vis_show = cv2.resize(vis_show, (target_w, target_h), interpolation=cv2.INTER_AREA)
    ir_show = cv2.resize(ir_show, (target_w, target_h), interpolation=cv2.INTER_AREA)
    overlay = cv2.resize(overlay, (target_w, target_h), interpolation=cv2.INTER_AREA)

    canvas = np.hstack([vis_show, ir_show, overlay])

    out_path = save_dir / f"{stem}_box{box_id}_full_vis_roi_eval.jpg"
    imwrite_unicode(out_path, canvas)


# ============================================================
# 8. 主评估流程
# ============================================================

def evaluate_uav_roi_registration():
    visible_dict, infrared_dict = check_folders()

    visible_stems = set(visible_dict.keys())
    infrared_stems = set(infrared_dict.keys())

    common_stems_all = sorted(
        visible_stems.intersection(infrared_stems),
        key=safe_sort_key
    )

    common_stems = common_stems_all[:NUM_EVAL_IMAGES]

    print("\n========== Pair Matching ==========")
    print(f"Total matched visible-infrared image pairs: {len(common_stems_all)}")
    print(f"Actually evaluated image pairs:            {len(common_stems)}")
    print("===================================")

    results = []
    failed_list = []
    no_label_list = []
    saved_vis_count = 0

    for stem in tqdm(common_stems, desc="Evaluating UAV ROI registration"):
        vis_path = visible_dict[stem]
        ir_path = infrared_dict[stem]
        label_path = LABEL_DIR / f"{stem}.txt"

        try:
            boxes = read_yolo_label(label_path)

            if len(boxes) == 0:
                no_label_list.append(stem)
                continue

            if not EVALUATE_ALL_BOXES:
                boxes = boxes[:1]

            vis_gray = imread_gray_unicode(vis_path)
            ir_gray = imread_gray_unicode(ir_path)

            vis_gray, ir_gray = resize_to_same_size(vis_gray, ir_gray)

            h, w = ir_gray.shape[:2]

            for box_id, box in enumerate(boxes):
                roi = yolo_box_to_pixel_roi(
                    box,
                    image_width=w,
                    image_height=h,
                    expand_ratio=ROI_EXPAND_RATIO,
                    min_size=MIN_ROI_SIZE,
                    max_size=MAX_ROI_SIZE
                )

                if roi is None:
                    continue

                # 注意：下面只裁剪 UAV 区域参与评估
                vis_roi = crop_roi(vis_gray, roi)
                ir_roi = crop_roi(ir_gray, roi)

                if vis_roi.size == 0 or ir_roi.size == 0:
                    continue

                vis_edge = preprocess_for_cross_modal(vis_roi)
                ir_edge = preprocess_for_cross_modal(ir_roi)

                dx_phase, dy_phase, response_phase = estimate_residual_shift_phase_correlation(
                    vis_edge,
                    ir_edge
                )

                if np.isnan(dx_phase) or np.isnan(dy_phase):
                    residual_phase = np.nan
                else:
                    residual_phase = float(np.sqrt(dx_phase ** 2 + dy_phase ** 2))

                ecc_result = estimate_residual_shift_ecc(vis_edge, ir_edge)

                if ecc_result is not None:
                    dx_ecc, dy_ecc, ecc_score = ecc_result
                    residual_ecc = float(np.sqrt(dx_ecc ** 2 + dy_ecc ** 2))
                else:
                    dx_ecc = np.nan
                    dy_ecc = np.nan
                    ecc_score = np.nan
                    residual_ecc = np.nan

                overlap = edge_overlap_score(vis_edge, ir_edge)

                x1, y1, x2, y2 = roi

                results.append({
                    "image_id": stem,
                    "box_id": box_id,
                    "visible_path": str(vis_path),
                    "infrared_path": str(ir_path),
                    "label_path": str(label_path),

                    "roi_x1": x1,
                    "roi_y1": y1,
                    "roi_x2": x2,
                    "roi_y2": y2,
                    "roi_width": x2 - x1,
                    "roi_height": y2 - y1,

                    "bbox_xc_norm": box["xc"],
                    "bbox_yc_norm": box["yc"],
                    "bbox_w_norm": box["w"],
                    "bbox_h_norm": box["h"],

                    "dx_phase": dx_phase,
                    "dy_phase": dy_phase,
                    "residual_error_phase_px": residual_phase,
                    "phase_response": response_phase,

                    "dx_ecc": dx_ecc,
                    "dy_ecc": dy_ecc,
                    "residual_error_ecc_px": residual_ecc,
                    "ecc_score": ecc_score,

                    "roi_edge_overlap": overlap
                })

                if SAVE_VIS and saved_vis_count < MAX_VIS_SAVE:
                    # 注意：可视化是整图，但图上标出了 UAV ROI；
                    # 评估指标仍然只由 ROI 内计算得到。
                    save_full_image_visualization(
                        vis_gray=vis_gray,
                        ir_gray=ir_gray,
                        roi=roi,
                        stem=stem,
                        box_id=box_id,
                        dx=dx_ecc if not np.isnan(dx_ecc) else dx_phase,
                        dy=dy_ecc if not np.isnan(dy_ecc) else dy_phase,
                        residual_error=residual_ecc if not np.isnan(residual_ecc) else residual_phase,
                        save_dir=VIS_DIR
                    )
                    saved_vis_count += 1

        except Exception as e:
            failed_list.append((stem, str(e)))

    df = pd.DataFrame(results)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n========== UAV ROI Evaluation Finished ==========")
    print(f"Evaluated image pairs: {len(common_stems)}")
    print(f"Valid evaluated UAV ROIs: {len(df)}")
    print(f"No-label images: {len(no_label_list)}")
    print(f"Failed images: {len(failed_list)}")
    print(f"Detailed results saved to: {OUTPUT_CSV}")
    print("=================================================")

    if no_label_list:
        no_label_path = OUTPUT_CSV.with_name("uav_roi_no_label_images_500.txt")
        with open(no_label_path, "w", encoding="utf-8") as f:
            for item in no_label_list:
                f.write(item + "\n")
        print(f"No-label image list saved to: {no_label_path}")

    if failed_list:
        failed_path = OUTPUT_CSV.with_name("uav_roi_failed_images_500.txt")
        with open(failed_path, "w", encoding="utf-8") as f:
            for stem, err in failed_list:
                f.write(f"{stem}: {err}\n")
        print(f"Failed image list saved to: {failed_path}")

    if len(df) == 0:
        print("No valid UAV ROI was evaluated. Please check label path and YOLO label format.")
        return

    summarize_roi_results(df)


# ============================================================
# 9. 详细统计汇总
# ============================================================

def summarize_roi_results(df: pd.DataFrame):
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 300)

    summary_rows = []

    def add_error_summary(method_name, column_name):
        errors = df[column_name].dropna().to_numpy()

        if len(errors) == 0:
            return

        summary_rows.append({
            "Method": method_name,
            "Evaluated UAV ROIs": len(errors),
            "Mean residual error (px)": np.mean(errors),
            "Median residual error (px)": np.median(errors),
            "Std residual error (px)": np.std(errors),
            "Min residual error (px)": np.min(errors),
            "Max residual error (px)": np.max(errors),
            "P25 residual error (px)": np.percentile(errors, 25),
            "P75 residual error (px)": np.percentile(errors, 75),
            "P90 residual error (px)": np.percentile(errors, 90),
            "P95 residual error (px)": np.percentile(errors, 95),
            "Error < 1px (%)": np.mean(errors < 1) * 100,
            "Error < 2px (%)": np.mean(errors < 2) * 100,
            "Error < 3px (%)": np.mean(errors < 3) * 100,
            "Error < 5px (%)": np.mean(errors < 5) * 100,
            "Error < 10px (%)": np.mean(errors < 10) * 100,
        })

    add_error_summary("Phase correlation on UAV ROI edge maps", "residual_error_phase_px")
    add_error_summary("ECC translation on UAV ROI edge maps", "residual_error_ecc_px")

    summary_df = pd.DataFrame(summary_rows)

    OUTPUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8-sig")

    print("\n===== UAV ROI Residual Alignment Error Summary =====")
    print(summary_df.round(4).to_string(index=False))

    overlaps = df["roi_edge_overlap"].dropna().to_numpy()

    if len(overlaps) > 0:
        overlap_summary = pd.DataFrame([{
            "Metric": "UAV ROI edge overlap",
            "Evaluated UAV ROIs": len(overlaps),
            "Mean edge overlap": np.mean(overlaps),
            "Median edge overlap": np.median(overlaps),
            "Std edge overlap": np.std(overlaps),
            "Min edge overlap": np.min(overlaps),
            "Max edge overlap": np.max(overlaps),
            "P25 edge overlap": np.percentile(overlaps, 25),
            "P75 edge overlap": np.percentile(overlaps, 75),
            "P90 edge overlap": np.percentile(overlaps, 90),
            "P95 edge overlap": np.percentile(overlaps, 95),
        }])

        overlap_summary_path = OUTPUT_SUMMARY.with_name("uav_roi_edge_overlap_summary_500.csv")
        overlap_summary.to_csv(overlap_summary_path, index=False, encoding="utf-8-sig")

        print("\n===== UAV ROI Edge Overlap Summary =====")
        print(overlap_summary.round(4).to_string(index=False))
        print(f"\nROI edge overlap summary saved to: {overlap_summary_path}")

    # 额外输出 ECC 重点指标，方便论文使用
    if "residual_error_ecc_px" in df.columns:
        ecc_errors = df["residual_error_ecc_px"].dropna().to_numpy()

        if len(ecc_errors) > 0:
            print("\n===== Recommended Metrics for Paper: ECC-based UAV ROI Alignment =====")
            print(f"Evaluated UAV ROIs: {len(ecc_errors)}")
            print(f"Mean residual alignment error:   {np.mean(ecc_errors):.4f} px")
            print(f"Median residual alignment error: {np.median(ecc_errors):.4f} px")
            print(f"Std residual alignment error:    {np.std(ecc_errors):.4f} px")
            print(f"Error < 2 px:  {np.mean(ecc_errors < 2) * 100:.2f}%")
            print(f"Error < 5 px:  {np.mean(ecc_errors < 5) * 100:.2f}%")
            print(f"Error < 10 px: {np.mean(ecc_errors < 10) * 100:.2f}%")

    print(f"\nSummary saved to: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    evaluate_uav_roi_registration()
