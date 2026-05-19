import argparse
import numpy as np
import cv2
import os

# ==========================================
# 你的論文演算法核心邏輯 (完全保留)
# ==========================================
def get_rgba(img):
    if img.shape[2] == 4:
        img_rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        R = img_rgba[:, :, 0]
        G = img_rgba[:, :, 1]
        B = img_rgba[:, :, 2]
        A = img_rgba[:, :, 3]
        img_rgb = img_rgba[:, :, :3]
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        R = img_rgb[:, :, 0]
        G = img_rgb[:, :, 1]
        B = img_rgb[:, :, 2]
        A = np.full((img_rgb.shape[0], img_rgb.shape[1]), 255, dtype=np.uint8)

    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_ycbcr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    return R, G, B, A, img_hsv, img_ycbcr

def base_requirement(R, G, B, A):
    R_int = R.astype(int)
    G_int = G.astype(int)
    return (R > 95) & (G > 40) & (B > 20) & (R > G) & (R > B) & (np.abs(R_int - G_int) > 15) & (A > 15)

def hsv_requirement(img_hsv):
    H = img_hsv[:, :, 0].astype(float) * 2.0
    S = img_hsv[:, :, 1].astype(float) / 255.0
    return (H >= 0.0) & (H <= 50.0) & (S >= 0.23) & (S <= 0.68)

def ycbcr_requirement(img_ycbcr):
    Y = img_ycbcr[:, :, 0]
    Cr = img_ycbcr[:, :, 1]
    Cb = img_ycbcr[:, :, 2]
    return (Cr > 135) & (Cb > 85) & (Y > 80) & \
           (Cr <= (1.5862 * Cb) + 20) & \
           (Cr >= (0.3448 * Cb) + 76.2069) & \
           (Cr >= (-4.5652 * Cb) + 234.5652) & \
           (Cr <= (-1.15 * Cb) + 301.75) & \
           (Cr <= (-2.2857 * Cb) + 432.85)

# ==========================================
# 新增：包裝你的演算法，方便給每一張圖呼叫
# ==========================================
def detect_skin_paper_method(img):
    R, G, B, A, img_hsv, img_ycbcr = get_rgba(img)
    mask_base = base_requirement(R, G, B, A)
    mask_hsv = hsv_requirement(img_hsv)
    mask_ycbcr = ycbcr_requirement(img_ycbcr)

    final_skin_mask = mask_base & (mask_hsv | mask_ycbcr)

    binary_result = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    binary_result[final_skin_mask] = 255
    return binary_result

# ==========================================
# 新增：IoU 計算邏輯
# ==========================================
def calculate_iou(mask1, mask2):
    m1 = (mask1 > 0).astype(np.float32)
    m2 = (mask2 > 0).astype(np.float32)
    
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    
    if union == 0:
        return 1.0
        
    return intersection / union

# ==========================================
# 新增：批次處理與評估迴圈
# ==========================================
def evaluate_dataset():
    # 請根據你實際的檔案名稱修改這裡
    IMAGE_GT_PAIRS = [
        ("pic1.jpg", "gt1.png"),
        ("pic2.jpg", "gt2.png"),
        ("pic3.jpg", "gt3.png"),
        ("pic4.jpg", "gt4.png"),
        ("pic5.jpg", "gt5.png"),
        ("pic6.jpg", "gt6.png")
    ]
    
    IMAGE_DIR = "images"
    GT_DIR = "ground_truth"
    
    if not os.path.exists(IMAGE_DIR) or not os.path.exists(GT_DIR):
        print(f"請先建立 {IMAGE_DIR} 和 {GT_DIR} 資料夾，並放入圖片！")
        return

    print("="*45)
    print("論文演算法膚色偵測 IOU 評估")
    print("="*45)
    print(f"{'Image ID':<15} | {'IoU Score':<10}")
    print("-" * 30)

    iou_scores = []

    for img_name, gt_name in IMAGE_GT_PAIRS:
        img_path = os.path.join(IMAGE_DIR, img_name)
        gt_path = os.path.join(GT_DIR, gt_name)
        
        # 讀取圖片 (保留你的 IMREAD_UNCHANGED 以支援 Alpha 通道)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        # GT 以灰階讀取
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) 
        
        if img is None or gt is None:
            print(f"Warning: 找不到檔案 {img_name} 或 {gt_name}")
            continue
            
        # 確保 GT 是二元影像
        _, gt_mask = cv2.threshold(gt, 1, 255, cv2.THRESH_BINARY)
        
        # 執行你的演算法取得預測遮罩
        pred_mask = detect_skin_paper_method(img)
        
        # 避免大小不一致的問題
        if pred_mask.shape != gt_mask.shape:
            gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 計算 IoU 並記錄
        iou = calculate_iou(pred_mask, gt_mask)
        iou_scores.append(iou)
        
        print(f"{img_name.split('.')[0]:<15} | {iou:.4f}")

    print("-" * 30)
    if iou_scores:
        avg_iou = sum(iou_scores) / len(iou_scores)
        print(f"{'Average IoU':<15} | {avg_iou:.4f}")
    print("="*45)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skin Detection Evaluation")
    parser.add_argument("--mode", type=str, default="eval", choices=["single", "eval"], 
                        help="使用 'single' 測試單張圖片，或 'eval' 評估整個資料夾")
    parser.add_argument("--input", type=str, help="單張圖片測試的路徑 (配合 --mode single 使用)")
    args = parser.parse_args()

    # 保留你原本測單張圖的功能，並加入評估模式切換
    if args.mode == "single":
        if not args.input:
            print("請提供 --input 圖片路徑")
        else:
            img = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
            result = detect_skin_paper_method(img)
            cv2.imshow("Original", img)
            cv2.imshow("Skin Mask", result)
            cv2.imwrite("output_mask.png", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        # 預設執行資料集評估
        evaluate_dataset()