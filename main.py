import argparse
import numpy as np
import cv2
import os

def get_rgba(img):
    if img.shape[2] == 4:
        # 將 BGRA 轉換為 RGBA，以符合論文的 RGB 邏輯
        img_bgr_blurred = cv2.bilateralFilter(img[:,:,:3], 9, 75, 75)
        # img_bgr_blurred = cv2.GaussianBlur(img[:,:,:3], (5, 5), 0)
        img_to_process = np.dstack((img_bgr_blurred, img[:,:,3]))
        img_rgba = cv2.cvtColor(img_to_process, cv2.COLOR_BGRA2RGBA)
        
        # 直接提取整張圖的 R, G, B, A 矩陣！
        # 這裡的寫法代表：取出所有高度、所有寬度，以及對應的第幾個通道
        R = img_rgba[:, :, 0]
        G = img_rgba[:, :, 1]
        B = img_rgba[:, :, 2]
        A = img_rgba[:, :, 3]
        img_rgb = img_rgba[:, :, :3]
    else:
        # 如果圖片只有 3 個通道 (例如 JPG 沒有透明度)
        # 將 BGR 轉換為 RGB
        # img_to_process = cv2.GaussianBlur(img, (5, 5), 0)
        img_to_process = cv2.bilateralFilter(img[:,:,:3], 9, 75, 75)
        img_rgb = cv2.cvtColor(img_to_process, cv2.COLOR_BGR2RGB)
        
        R = img_rgb[:, :, 0]
        G = img_rgb[:, :, 1]
        B = img_rgb[:, :, 2]
        # 因為沒有 Alpha 通道，我們可以手動建立一個全為 255 (完全不透明) 的矩陣
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

def filter_small_components(binary_mask, min_area_ratio=0.005):
    """
    找出所有獨立的白色區塊，面積小於整張圖面積 N% 的直接塗黑。
    這對去除背景的衣服反光、木頭反光非常有效。
    """
    total_area = binary_mask.shape[0] * binary_mask.shape[1]
    min_area = total_area * min_area_ratio
    
    # 尋找連通區域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    clean_mask = np.zeros_like(binary_mask)
    # label 0 是背景，所以從 1 開始跑
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean_mask[labels == i] = 255
            
    return clean_mask

def calculate_iou(mask1, mask2):
    m1 = (mask1 > 0).astype(np.float32)
    m2 = (mask2 > 0).astype(np.float32)
    
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    
    if union == 0:
        return 1.0
        
    return intersection / union

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input Image")
    parser.add_argument("--input", type=str, required=True, help="Path to the input image")
    parser.add_argument("--groundtruth", type=str, required=True, help="Path to the groundtruth image")
    args = parser.parse_args()
    basename = os.path.basename(args.input)

    img = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    groundtruth = cv2.imread(args.groundtruth, cv2.IMREAD_GRAYSCALE)
    R, G, B, A, img_hsv, img_ycbcr = get_rgba(img)

    mask_base = base_requirement(R, G, B, A)
    mask_hsv = hsv_requirement(img_hsv)
    mask_ycbcr = ycbcr_requirement(img_ycbcr)

    final_skin_mask = mask_base & (mask_hsv | mask_ycbcr)

    binary_result = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    binary_result[final_skin_mask] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    # 開運算：先侵蝕再膨脹 (去雜訊)
    binary_result = cv2.morphologyEx(binary_result, cv2.MORPH_OPEN, kernel)
    # 閉運算：先膨脹再侵蝕 (補破洞，例如眼睛、嘴巴)
    binary_result = cv2.morphologyEx(binary_result, cv2.MORPH_CLOSE, kernel)

    binary_result = filter_small_components(binary_result, min_area_ratio=0.01)

    iou = calculate_iou(binary_result, groundtruth)
    print(f"{'Image ID':<15} | {'IoU Score':<10}")
    print(f"{basename:<15} | {iou:.4f}")

    # cv2.imshow("Original", img)
    # cv2.imshow("Skin Detection (Binary Mask)", binary_result)
    cv2.imwrite(f"{basename}_output.png", binary_result)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    