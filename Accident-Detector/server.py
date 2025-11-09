import cv2
import numpy as np
from collections import Counter
import torch
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO
import uvicorn
from fastapi import FastAPI, File, UploadFile
import os
import shutil
import asyncio

# ------------------------------
# Initialize FastAPI App
# ------------------------------
app = FastAPI(title="Accident Analysis API")

# ------------------------------
# Load YOLO Model (Load once at startup)
# ------------------------------
# --- UPDATE THIS PATH ---
yolo_model_path = r"C:\Users\saket\Downloads\Accident-Detection-Model-master (1)\Accident-Detection-Model-master\runs\detect\train\weights\best.pt"

try:
    yolo_model = YOLO(yolo_model_path)
    if hasattr(yolo_model, 'model'):
        yolo_model.model.eval()
    print("✓ YOLO model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

# ------------------------------
# Enhanced Severity Classifier (Your code, unchanged)
# ------------------------------
class EnhancedSeverityClassifier:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def extract_advanced_features(self, crop_img, bbox_coords, img_shape):
        features = {}
        x1, y1, x2, y2 = bbox_coords
        bbox_area = (x2 - x1) * (y2 - y1)
        img_area = img_shape[0] * img_shape[1]
        features['area_ratio'] = bbox_area / img_area
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        img_center_x = img_shape[1] / 2
        img_center_y = img_shape[0] / 2
        features['distance_from_center'] = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2) / max(img_shape)
        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        dark_mask_hsv = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))
        features['dark_ratio_combined'] = np.sum(dark_mask_hsv > 0) / dark_mask_hsv.size
        red_mask1 = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        features['red_ratio'] = np.sum(red_mask > 0) / red_mask.size
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        edges_fine = cv2.Canny(gray, 50, 150)
        features['edge_density_combined'] = np.sum(edges_fine > 0) / edges_fine.size
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        features['sharpness'] = laplacian_var
        return features
    
    def detect_human_figures(self, crop_img):
        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        combined_dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))
        dark_ratio = np.sum(combined_dark_mask > 0) / combined_dark_mask.size
        contours, _ = cv2.findContours(combined_dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        human_likelihood = 0
        if contours:
            for contour in contours:
                if cv2.contourArea(contour) > 100:
                    x, y, w, h = cv2.boundingRect(contour)
                    contour_area = cv2.contourArea(contour)
                    bbox_area = w * h
                    if bbox_area > 0:
                        compactness = contour_area / bbox_area
                        aspect_ratio = w / h if h > 0 else 1
                        if (0.2 < compactness < 0.8) and (0.3 < aspect_ratio < 3.0):
                            human_likelihood += compactness * (contour_area / (crop_img.shape[0] * crop_img.shape[1]))
        is_human = human_likelihood > 0.1 and dark_ratio > 0.15
        return is_human, human_likelihood, dark_ratio, 0
    
    def calculate_severity_score(self, features, human_detection_results):
        score = 0
        reasons = []
        is_human, _, _, _ = human_detection_results
        if is_human:
            score += 10
            reasons.append("CRITICAL: Human figure detected on roadway")
        if features['area_ratio'] > 0.08:
            score += 3
            reasons.append("Large accident area")
        elif features['area_ratio'] > 0.04:
            score += 2
            reasons.append("Medium accident area")
        if features['distance_from_center'] < 0.15:
            score += 3
            reasons.append("Accident in central roadway area")
        if features['dark_ratio_combined'] > 0.25:
            score += 3
            reasons.append("Heavy debris/fluid spillage detected")
        if features['edge_density_combined'] > 0.15:
            score += 3
            reasons.append("High structural damage/debris scattering")
        if features['red_ratio'] > 0.05:
            score += 3
            reasons.append("Blood/emergency indicators detected")
        if features['sharpness'] < 100:
            score += 2
            reasons.append("High-speed impact indicators")
        
        # Ensure score has a base
        if score == 0 and features['area_ratio'] > 0:
            score = 1
            reasons.append("Minor incident detected")

        return score, reasons

# ------------------------------
# Severity Prediction Logic (Your code, refactored)
# ------------------------------
severity_classifier = EnhancedSeverityClassifier()

def predict_severity_enhanced(crop_img, bbox_coords, img_shape):
    features = severity_classifier.extract_advanced_features(crop_img, bbox_coords, img_shape)
    human_detection_results = severity_classifier.detect_human_figures(crop_img)
    score, reasons = severity_classifier.calculate_severity_score(features, human_detection_results)
    
    if score >= 12:
        severity = "CRITICAL"
    elif score >= 8:
        severity = "SEVERE"
    elif score >= 5:
        severity = "MODERATE"
    else:
        severity = "MINOR"
    
    return severity, score, reasons

# ------------------------------
# API Endpoint for Video Analysis
# ------------------------------

# --- Configuration ---
# Process 1 frame every N frames.
# Increase N for faster processing (e.g., 30 = ~1 frame/sec for 30fps video)
# Decrease N for more thorough analysis (e.g., 10)
FRAME_SAMPLE_RATE = 15

def get_overall_severity(severity_counts):
    """Determine the single highest severity for the whole video."""
    if severity_counts.get("CRITICAL", 0) > 0:
        return "CRITICAL"
    if severity_counts.get("SEVERE", 0) > 0:
        return "SEVERE"
    if severity_counts.get("MODERATE", 0) > 0:
        return "MODERATE"
    if severity_counts.get("MINOR", 0) > 0:
        return "MINOR"
    return "NONE"

@app.post("/analyze_video/")
async def analyze_video(video: UploadFile = File(...)):
    """
    Accepts a video file, samples frames, runs detection,
    and returns a combined JSON report.
    """
    
    # Save video temporarily
    temp_dir = "temp_videos"
    os.makedirs(temp_dir, exist_ok=True)
    temp_video_path = os.path.join(temp_dir, video.filename)
    
    try:
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
            
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            return {"error": "Could not open video file."}

        # --- Report Aggregation Variables ---
        total_detections = 0
        severity_counts = Counter()
        highest_severity_score = 0
        highest_severity_level = "NONE"
        critical_incident_details = [] # Store info for most severe frames
        
        frame_index = 0
        frames_processed = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # --- Frame Sampling Logic ---
            if frame_index % FRAME_SAMPLE_RATE == 0:
                frames_processed += 1
                original_frame = frame.copy()
                
                # Run YOLO detection
                results = yolo_model.predict(frame, conf=0.25, iou=0.5, verbose=False)
                
                if len(results[0].boxes) > 0:
                    for box in results[0].boxes:
                        total_detections += 1
                        coords = box.xyxy.cpu().numpy().flatten()
                        x1, y1, x2, y2 = map(int, coords)
                        
                        # Ensure coordinates are valid
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                        
                        if (x2 - x1) < 10 or (y2 - y1) < 10:
                            continue
                            
                        # Crop and analyze
                        accident_crop = original_frame[y1:y2, x1:x2]
                        if accident_crop.size == 0:
                            continue
                        
                        severity, score, reasons = predict_severity_enhanced(
                            accident_crop, (x1, y1, x2, y2), frame.shape
                        )
                        
                        # --- Aggregate Results ---
                        severity_counts[severity] += 1
                        
                        if score > highest_severity_score:
                            highest_severity_score = score
                            highest_severity_level = severity
                        
                        # Log details for severe incidents
                        if severity in ["CRITICAL", "SEVERE"]:
                            critical_incident_details.append({
                                "frame": frame_index,
                                "severity": severity,
                                "score": score,
                                "reasons": reasons,
                                "box": [x1, y1, x2, y2]
                            })
            
            frame_index += 1
            
        cap.release()
        
        # --- Generate Final Report ---
        overall_severity = get_overall_severity(severity_counts)
        
        report = {
            "filename": video.filename,
            "total_frames_in_video": frame_index,
            "total_frames_processed": frames_processed,
            "frame_sample_rate": FRAME_SAMPLE_RATE,
            "overall_video_severity": overall_severity,
            "total_detections_found": total_detections,
            "severity_breakdown": dict(severity_counts),
            "most_severe_incident": {
                "level": highest_severity_level,
                "score": highest_severity_score
            },
            "severe_incident_log": critical_incident_details[:20] # Limit to top 20 logs
        }
        
        return report

    except Exception as e:
        return {"error": f"An error occurred during processing: {str(e)}"}
    finally:
        # Clean up temporary file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)


# ------------------------------
# Main entry point to run the server
# ------------------------------
if __name__ == "__main__":
    print("Starting FastAPI server at http://0.0.0.0:8000")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)