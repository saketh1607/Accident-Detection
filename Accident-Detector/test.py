from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter

# ------------------------------
# Load YOLO accident detection model
# ------------------------------
yolo_model_path = r"C:\Users\saket\Downloads\Accident-Detection-Model-master (1)\Accident-Detection-Model-master\runs\detect\train\weights\best.pt"
yolo_model = YOLO(yolo_model_path)

# ------------------------------
# Enhanced Severity Classifier with Multiple Features
# ------------------------------
def extract_features(crop_img, bbox_coords, img_shape):
    """
    Extract multiple features for better severity classification
    """
    features = {}
    
    # 1. Bounding box area (larger accidents tend to be more severe)
    x1, y1, x2, y2 = bbox_coords
    bbox_area = (x2 - x1) * (y2 - y1)
    img_area = img_shape[0] * img_shape[1]
    features['area_ratio'] = bbox_area / img_area
    
    # 2. Position (accidents in middle of road are often more severe)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    img_center_x = img_shape[1] / 2
    features['distance_from_center'] = abs(center_x - img_center_x) / img_shape[1]
    features['vertical_position'] = center_y / img_shape[0]
    
    # 3. Color analysis (debris, fluids indicate severity)
    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    
    # Check for dark spots (oil, blood, debris)
    dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 50]))
    features['dark_ratio'] = np.sum(dark_mask > 0) / dark_mask.size
    
    # Check for scattered debris (high variance in brightness)
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    features['brightness_variance'] = np.var(gray)
    
    # 4. Edge density (more edges = more debris/damage)
    edges = cv2.Canny(gray, 50, 150)
    features['edge_density'] = np.sum(edges > 0) / edges.size
    
    # 5. Shape aspect ratio (fallen person vs vehicle collision)
    width = x2 - x1
    height = y2 - y1
    features['aspect_ratio'] = width / height if height > 0 else 1
    
    return features

def detect_person_on_road(crop_img):
    """
    Detect if there's a person lying on the road using shape and color analysis
    """
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    
    # Check for dark human-like shape (clothing)
    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    
    # Detect dark clothing or body
    dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))
    dark_ratio = np.sum(dark_mask > 0) / dark_mask.size
    
    # Check compactness of dark region (person vs scattered debris)
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        compactness = (w * h) / (crop_img.shape[0] * crop_img.shape[1])
        
        # Person lying down: significant dark area (15%+) in relatively compact shape
        if dark_ratio > 0.15 and compactness > 0.3:
            return True, dark_ratio, compactness
    
    return False, dark_ratio, 0

def rule_based_severity(features, crop_img):
    """
    Rule-based severity classification using extracted features
    Replace this with your trained CNN model for production
    """
    score = 0
    reasons = []
    
    # CRITICAL: Check for person on road FIRST
    is_person, dark_ratio, compactness = detect_person_on_road(crop_img)
    if is_person:
        score += 8  # Automatically high severity
        reasons.append("CRITICAL: Person detected on road")
    
    # Large accident area
    if features['area_ratio'] > 0.12:
        score += 3
        reasons.append("Large accident area")
    elif features['area_ratio'] > 0.06:
        score += 2
        reasons.append("Medium accident area")
    else:
        score += 1
    
    # Central position (more dangerous)
    if features['distance_from_center'] < 0.2:
        score += 3
        reasons.append("Accident in center of road")
    elif features['distance_from_center'] < 0.35:
        score += 1
    
    # Dark spots indicate fluids/debris/person
    if features['dark_ratio'] > 0.2:
        score += 2
        reasons.append("Significant dark debris/fluids/body detected")
    elif features['dark_ratio'] > 0.1:
        score += 1
        reasons.append("Dark debris detected")
    
    # High edge density = scattered debris or vehicle damage
    if features['edge_density'] > 0.12:
        score += 2
        reasons.append("Scattered debris detected")
    
    # Horizontal aspect ratio (fallen person/motorcycle)
    if 1.3 < features['aspect_ratio'] < 3.0:
        score += 3
        reasons.append("Fallen person/motorcycle orientation detected")
    elif features['aspect_ratio'] > 3.0:
        score += 1
        reasons.append("Wide accident area")
    
    # High brightness variance indicates damage/debris
    if features['brightness_variance'] > 2000:
        score += 2
        reasons.append("High contrast indicating damage/debris")
    elif features['brightness_variance'] > 1200:
        score += 1
    
    # Lower road position (middle/bottom of frame = on roadway)
    if features['vertical_position'] > 0.5:
        score += 2
        reasons.append("Accident on main roadway")
    
    # Classify based on score with STRICT thresholds
    if score >= 8:
        severity = "Severe"
    elif score >= 5:
        severity = "Moderate"
    else:
        severity = "Minor"
    
    return severity, score, reasons

def predict_severity_enhanced(crop_img, bbox_coords, img_shape):
    """
    Enhanced severity prediction using multiple features
    """
    features = extract_features(crop_img, bbox_coords, img_shape)
    severity, score, reasons = rule_based_severity(features, crop_img)
    
    return severity, score, reasons

# ------------------------------
# Load and process image
# ------------------------------
img_path = r"C:\Users\saket\Downloads\OIP (1).jpg"
img = cv2.imread(img_path)
original_img = img.copy()

# ------------------------------
# Step 1: Detect accidents
# ------------------------------
results = yolo_model.predict(img, conf=0.4)

# ------------------------------
# Step 2: Process each detected accident
# ------------------------------
if len(results[0].boxes) == 0:
    print("No accidents detected.")
else:
    print(f"\nDetected {len(results[0].boxes)} accident(s):\n")
    
    for i, box in enumerate(results[0].boxes):
        # Convert tensor to numpy array
        coords = box.xyxy.cpu().numpy().flatten()  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, coords)
        
        # Crop accident region
        accident_crop = original_img[y1:y2, x1:x2]
        
        # Predict severity with enhanced method
        severity, score, reasons = predict_severity_enhanced(
            accident_crop, 
            (x1, y1, x2, y2), 
            img.shape
        )
        
        # Print detailed analysis
        print(f"Accident {i+1}:")
        print(f"  Severity: {severity} (Score: {score}/12)")
        print(f"  Reasons:")
        for reason in reasons:
            print(f"    - {reason}")
        print()
        
        # Choose color based on severity
        color_map = {
            "Severe": (0, 0, 255),    # Red
            "Moderate": (0, 165, 255),  # Orange
            "Minor": (0, 255, 0)      # Green
        }
        color = color_map.get(severity, (0, 0, 255))
        
        # Annotate with thicker lines for visibility
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        
        # Add severity label with background
        label = f"{severity} ({score}/12)"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(img, (x1, y1-label_h-10), (x1+label_w, y1), color, -1)
        cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

# ------------------------------
# Display and save results
# ------------------------------
# Save annotated image first
output_path = img_path.replace('.jpg', '_annotated.jpg')
cv2.imwrite(output_path, img)
print(f"\nAnnotated image saved to: {output_path}")

# Resize if image is too large for display
height, width = img.shape[:2]
if width > 1200:
    scale = 1200 / width
    img_display = cv2.resize(img, (int(width*scale), int(height*scale)))
else:
    img_display = img

# Try to display with timeout
try:
    cv2.imshow("Accident Detection & Severity", img_display)
    print("\nPress any key in the image window to close (or wait 30 seconds)...")
    cv2.waitKey(30000)  # 30 second timeout instead of infinite wait
except:
    print("Could not display window (this is normal in some environments)")
# finally:
    # cv2.destroyAllWindows()
    
print("\nProcessing complete!")