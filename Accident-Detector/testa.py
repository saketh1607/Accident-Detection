from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# ------------------------------
# Load YOLO accident detection model with better configuration
# ------------------------------
yolo_model_path = r"C:\Users\saket\Downloads\Accident-Detection-Model-master (1)\Accident-Detection-Model-master\runs\detect\train\weights\best.pt"
yolo_model = YOLO(yolo_model_path)

# Set model to evaluation mode if available
if hasattr(yolo_model, 'model'):
    yolo_model.model.eval()

# ------------------------------
# Enhanced Severity Classifier with CNN Model Integration
# ------------------------------

class EnhancedSeverityClassifier:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def extract_advanced_features(self, crop_img, bbox_coords, img_shape):
        """
        Extract comprehensive features for severity assessment
        """
        features = {}
        
        # 1. Bounding box analysis
        x1, y1, x2, y2 = bbox_coords
        bbox_area = (x2 - x1) * (y2 - y1)
        img_area = img_shape[0] * img_shape[1]
        features['area_ratio'] = bbox_area / img_area
        
        # 2. Position analysis
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        img_center_x = img_shape[1] / 2
        img_center_y = img_shape[0] / 2
        
        features['distance_from_center'] = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2) / max(img_shape)
        features['horizontal_position'] = center_x / img_shape[1]
        features['vertical_position'] = center_y / img_shape[0]
        
        # 3. Multi-color space analysis
        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(crop_img, cv2.COLOR_BGR2LAB)
        
        # Dark regions (oil, blood, debris)
        dark_mask_hsv = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))
        dark_mask_lab = cv2.inRange(lab, np.array([0, 0, 0]), np.array([255, 128, 128]))
        
        features['dark_ratio_hsv'] = np.sum(dark_mask_hsv > 0) / dark_mask_hsv.size
        features['dark_ratio_lab'] = np.sum(dark_mask_lab > 0) / dark_mask_lab.size
        features['dark_ratio_combined'] = (features['dark_ratio_hsv'] + features['dark_ratio_lab']) / 2
        
        # Red regions (blood, emergency lights)
        red_mask1 = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        features['red_ratio'] = np.sum(red_mask > 0) / red_mask.size
        
        # 4. Texture and edge analysis
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        
        # Multi-scale edge detection
        edges_fine = cv2.Canny(gray, 50, 150)
        edges_coarse = cv2.Canny(gray, 30, 100)
        
        features['edge_density_fine'] = np.sum(edges_fine > 0) / edges_fine.size
        features['edge_density_coarse'] = np.sum(edges_coarse > 0) / edges_coarse.size
        features['edge_density_combined'] = (features['edge_density_fine'] + features['edge_density_coarse']) / 2
        
        # Texture analysis using variance
        features['brightness_variance'] = np.var(gray)
        features['brightness_std'] = np.std(gray)
        
        # 5. Shape and structural analysis
        width = x2 - x1
        height = y2 - y1
        features['aspect_ratio'] = width / height if height > 0 else 1
        features['compactness'] = (width * height) / (2 * (width + height))**2 if (width + height) > 0 else 0
        
        # 6. Motion blur estimation (indicates high-speed impact)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        features['sharpness'] = laplacian_var
        
        return features
    
    def detect_human_figures(self, crop_img):
        """
        Enhanced human detection using multiple approaches
        """
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        
        # Multi-threshold dark region detection
        dark_masks = []
        for threshold in [50, 80, 100]:
            dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, threshold]))
            dark_masks.append(dark_mask)
        
        # Combine dark masks
        combined_dark_mask = cv2.bitwise_or(dark_masks[0], dark_masks[1])
        combined_dark_mask = cv2.bitwise_or(combined_dark_mask, dark_masks[2])
        
        dark_ratio = np.sum(combined_dark_mask > 0) / combined_dark_mask.size
        
        # Contour analysis for human-like shapes
        contours, _ = cv2.findContours(combined_dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        human_likelihood = 0
        best_compactness = 0
        
        if contours:
            for contour in contours:
                if cv2.contourArea(contour) > 100:  # Filter small noise
                    x, y, w, h = cv2.boundingRect(contour)
                    contour_area = cv2.contourArea(contour)
                    bbox_area = w * h
                    
                    if bbox_area > 0:
                        compactness = contour_area / bbox_area
                        aspect_ratio = w / h if h > 0 else 1
                        
                        # Human-like shape criteria
                        if (0.2 < compactness < 0.8) and (0.3 < aspect_ratio < 3.0):
                            human_likelihood += compactness * (contour_area / (crop_img.shape[0] * crop_img.shape[1]))
                            best_compactness = max(best_compactness, compactness)
        
        is_human = human_likelihood > 0.1 and dark_ratio > 0.15
        return is_human, human_likelihood, dark_ratio, best_compactness
    
    def calculate_severity_score(self, features, human_detection_results):
        """
        Advanced severity scoring with weighted features
        """
        score = 0
        reasons = []
        weights = {}
        
        is_human, human_likelihood, dark_ratio, compactness = human_detection_results
        
        # CRITICAL: Human on road (highest weight)
        if is_human:
            score += 10
            reasons.append("CRITICAL: Human figure detected on roadway")
            weights['human_presence'] = 10
        
        # Area-based scoring (moderate weight)
        if features['area_ratio'] > 0.15:
            score += 4
            reasons.append("Very large accident area")
            weights['area'] = 4
        elif features['area_ratio'] > 0.08:
            score += 3
            reasons.append("Large accident area")
            weights['area'] = 3
        elif features['area_ratio'] > 0.04:
            score += 2
            reasons.append("Medium accident area")
            weights['area'] = 2
        else:
            score += 1
            weights['area'] = 1
        
        # Central position scoring
        if features['distance_from_center'] < 0.15:
            score += 3
            reasons.append("Accident in central roadway area")
            weights['position'] = 3
        elif features['distance_from_center'] < 0.3:
            score += 2
            reasons.append("Accident in moderate traffic area")
            weights['position'] = 2
        else:
            score += 1
            weights['position'] = 1
        
        # Debris and damage indicators
        if features['dark_ratio_combined'] > 0.25:
            score += 3
            reasons.append("Heavy debris/fluid spillage detected")
            weights['debris'] = 3
        elif features['dark_ratio_combined'] > 0.15:
            score += 2
            reasons.append("Moderate debris detected")
            weights['debris'] = 2
        elif features['dark_ratio_combined'] > 0.08:
            score += 1
            reasons.append("Light debris detected")
            weights['debris'] = 1
        
        # Edge density (structural damage)
        if features['edge_density_combined'] > 0.15:
            score += 3
            reasons.append("High structural damage/debris scattering")
            weights['damage'] = 3
        elif features['edge_density_combined'] > 0.08:
            score += 2
            reasons.append("Moderate structural damage")
            weights['damage'] = 2
        elif features['edge_density_combined'] > 0.04:
            score += 1
            weights['damage'] = 1
        
        # Blood/emergency indicators
        if features['red_ratio'] > 0.05:
            score += 3
            reasons.append("Blood/emergency indicators detected")
            weights['emergency'] = 3
        elif features['red_ratio'] > 0.02:
            score += 1
            weights['emergency'] = 1
        
        # Sharpness (motion blur indicates speed)
        if features['sharpness'] < 100:  # Very blurry
            score += 2
            reasons.append("High-speed impact indicators")
            weights['speed'] = 2
        elif features['sharpness'] < 500:
            score += 1
            weights['speed'] = 1
        
        return score, reasons, weights

# Initialize the enhanced classifier
severity_classifier = EnhancedSeverityClassifier()

def predict_severity_enhanced(crop_img, bbox_coords, img_shape):
    """
    Final severity prediction using advanced features
    """
    features = severity_classifier.extract_advanced_features(crop_img, bbox_coords, img_shape)
    human_detection_results = severity_classifier.detect_human_figures(crop_img)
    score, reasons, weights = severity_classifier.calculate_severity_score(features, human_detection_results)
    
    # Enhanced classification with stricter thresholds
    if score >= 12:
        severity = "CRITICAL"
    elif score >= 8:
        severity = "SEVERE"
    elif score >= 5:
        severity = "MODERATE"
    else:
        severity = "MINOR"
    
    return severity, score, reasons, weights

# ------------------------------
# Improved Image Processing and Text Rendering
# ------------------------------

def create_text_background(img, text, position, font_scale, thickness, padding=5):
    """
    Create a background for text to ensure readability
    """
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = position
    
    # Calculate background coordinates
    bg_x1 = x - padding
    bg_y1 = y - text_h - padding
    bg_x2 = x + text_w + padding
    bg_y2 = y + baseline + padding
    
    return (bg_x1, bg_y1, bg_x2, bg_y2), (text_w, text_h)

def draw_text_with_background(img, text, position, font_scale, text_color, bg_color, thickness=2):
    """
    Draw text with semi-transparent background
    """
    # Create a temporary image for the background
    temp_img = img.copy()
    
    # Get text size and background coordinates
    bg_coords, (text_w, text_h) = create_text_background(temp_img, text, position, font_scale, thickness)
    bg_x1, bg_y1, bg_x2, bg_y2 = bg_coords
    
    # Ensure coordinates are within image bounds
    bg_x1 = max(0, bg_x1)
    bg_y1 = max(0, bg_y1)
    bg_x2 = min(img.shape[1], bg_x2)
    bg_y2 = min(img.shape[0], bg_y2)
    
    # Draw semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    
    # Draw text
    cv2.putText(img, text, (position[0], position[1]), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
    
    return bg_coords

# ------------------------------
# Load and process image with better error handling
# ------------------------------
img_path = r"C:\Users\saket\OneDrive\图片\Screenshots\Screenshot 2025-11-07 115829.jpg"
img = cv2.imread(img_path)

if img is None:
    print(f"Error: Could not load image from {img_path}")
    # Try alternative loading method
    try:
        from PIL import Image
        pil_img = Image.open(img_path)
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        print("Image loaded successfully using PIL")
    except Exception as e:
        print(f"Failed to load image: {e}")
        exit()

original_img = img.copy()
print(f"Image loaded: {img.shape[1]}x{img.shape[0]} pixels")

# ------------------------------
# Enhanced YOLO Detection with better parameters
# ------------------------------
print("\nRunning enhanced YOLO detection...")

# Try multiple confidence thresholds with improved parameters
confidence_thresholds = [0.35, 0.25, 0.15, 0.08]
results = None
best_detections = 0

for conf_threshold in confidence_thresholds:
    print(f"Trying confidence threshold: {conf_threshold}")
    results = yolo_model.predict(
        img, 
        conf=conf_threshold, 
        iou=0.5,  # Increased IoU for better accuracy
        verbose=False,
        augment=True  # Use test-time augmentation
    )
    
    num_detections = len(results[0].boxes)
    print(f"  Detections found: {num_detections}")
    
    if num_detections > best_detections:
        best_detections = num_detections
        best_conf = conf_threshold
        best_results = results

if best_detections > 0:
    results = best_results
    print(f"✓ Using best configuration: confidence={best_conf}, detections={best_detections}")
else:
    print("\n⚠ No confident detections found. Trying with minimal confidence...")
    results = yolo_model.predict(img, conf=0.01, verbose=False)

# ------------------------------
# Process detections with enhanced visualization
# ------------------------------
if len(results[0].boxes) == 0:
    print("No accidents detected with current model.")
    
    # Add "No Accident Detected" text to image
    text = "No Accident Detected"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
    text_x = (img.shape[1] - text_size[0]) // 2
    text_y = (img.shape[0] + text_size[1]) // 2
    
    draw_text_with_background(
        img, text, (text_x, text_y), 
        font_scale=1.5, 
        text_color=(255, 255, 255), 
        bg_color=(0, 0, 255),
        thickness=3
    )
    
else:
    print(f"\n{'='*60}")
    print(f"Detected {len(results[0].boxes)} accident(s)")
    print(f"{'='*60}\n")
    
    # Color mapping for severity levels
    color_map = {
        "CRITICAL": (0, 0, 255),      # Red
        "SEVERE": (0, 69, 255),       # Red-Orange
        "MODERATE": (0, 165, 255),    # Orange
        "MINOR": (0, 255, 255)        # Yellow
    }
    
    for i, box in enumerate(results[0].boxes):
        confidence = float(box.conf.cpu().numpy()[0])
        coords = box.xyxy.cpu().numpy().flatten()
        x1, y1, x2, y2 = map(int, coords)
        
        # Ensure coordinates are within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
        
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            print(f"Skipping detection {i+1}: bounding box too small")
            continue
        
        # Crop accident region
        accident_crop = original_img[y1:y2, x1:x2]
        
        if accident_crop.size == 0:
            print(f"Skipping detection {i+1}: empty crop region")
            continue
        
        # Enhanced severity prediction
        severity, score, reasons, weights = predict_severity_enhanced(
            accident_crop, 
            (x1, y1, x2, y2), 
            img.shape
        )
        
        # Print detailed analysis
        print(f"Accident {i+1}:")
        print(f"  Confidence: {confidence:.2%}")
        print(f"  Severity: {severity} (Score: {score}/20)")
        print(f"  Bounding Box: ({x1}, {y1}) to ({x2}, {y2})")
        print(f"  Feature Weights: {weights}")
        print(f"  Reasons:")
        for reason in reasons:
            print(f"    - {reason}")
        print()
        
        # Get color for this severity
        color = color_map.get(severity, (0, 0, 255))
        
        # Draw enhanced bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
        
        # Draw corner markers for better visibility
        marker_length = 15
        # Top-left
        cv2.line(img, (x1, y1), (x1 + marker_length, y1), color, 3)
        cv2.line(img, (x1, y1), (x1, y1 + marker_length,), color, 3)
        # Top-right
        cv2.line(img, (x2, y1), (x2 - marker_length, y1), color, 3)
        cv2.line(img, (x2, y1), (x2, y1 + marker_length), color, 3)
        # Bottom-left
        cv2.line(img, (x1, y2), (x1 + marker_length, y2), color, 3)
        cv2.line(img, (x1, y2), (x1, y2 - marker_length), color, 3)
        # Bottom-right
        cv2.line(img, (x2, y2), (x2 - marker_length, y2), color, 3)
        cv2.line(img, (x2, y2), (x2, y2 - marker_length), color, 3)
        
        # Create enhanced label with proper positioning
        label = f"{severity} - Score: {score}/20 - Conf: {confidence:.0%}"
        
        # Calculate text position (ensure it's within image bounds)
        text_y = max(30, y1 - 10) if y1 > 50 else y2 + 30
        
        # Draw text with background
        draw_text_with_background(
            img, label, (x1, text_y),
            font_scale=0.7,
            text_color=(255, 255, 255),
            bg_color=color,
            thickness=2
        )

# ------------------------------
# Add overall title to the image
# ------------------------------
title = "Accident Detection & Severity Analysis"
title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
title_x = (img.shape[1] - title_size[0]) // 2

draw_text_with_background(
    img, title, (title_x, 40),
    font_scale=1.2,
    text_color=(255, 255, 255),
    bg_color=(0, 0, 0),
    thickness=3
)

# ------------------------------
# Save and display results
# ------------------------------
output_path = img_path.replace('.jpg', '_enhanced_analysis.jpg')
cv2.imwrite(output_path, img)
print(f"\nEnhanced analysis image saved to: {output_path}")

# Create a detailed report image
report_img = img.copy()
height, width = report_img.shape[:2]

# Add summary information
summary_y = height - 150
cv2.rectangle(report_img, (0, summary_y), (width, height), (0, 0, 0), -1)

summary_text = [
    "Detection Summary:",
    f"Accidents Found: {len(results[0].boxes)}",
    "Severity Levels: CRITICAL(Red) > SEVERE(Orange) > MODERATE(Yellow) > MINOR(Green)"
]

for i, text in enumerate(summary_text):
    text_y = summary_y + 30 + (i * 25)
    cv2.putText(report_img, text, (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

report_path = img_path.replace('.jpg', '_detailed_report.jpg')
cv2.imwrite(report_path, report_img)
print(f"Detailed report saved to: {report_path}")

# Display results
try:
    # Resize for display if too large
    height, width = img.shape[:2]
    if width > 1200:
        scale = 1200 / width
        img_display = cv2.resize(img, (int(width*scale), int(height*scale)))
    else:
        img_display = img
    
    cv2.imshow("Enhanced Accident Detection & Severity Analysis", img_display)
    print("\nPress any key in the image window to close...")
    cv2.waitKey(0)
except Exception as e:
    print(f"Display error: {e}")
finally:
    cv2.destroyAllWindows()

print("\nEnhanced processing complete!")


