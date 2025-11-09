import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from collections import deque
import requests
from datetime import datetime
from urllib.parse import quote
import folium
from streamlit_folium import st_folium
import time
import threading
# Page Configuration
st.set_page_config(
    page_title="🚨 Ultra-Fast Accident Detection",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal CSS
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .block-container { padding-top: 1rem; }
    .alert-response {
        background: #f0f8ff;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
        margin: 10px 0;
    }
    .history-item {
        background: #f8f9fa;
        padding: 10px;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
YOLO_MODEL_PATH = r"C:\Users\saket\Downloads\Accident-Detection-Model-master (1)\Accident-Detection-Model-master\runs\detect\train\weights\best.pt"
DEFAULT_LAT = 17.3850
DEFAULT_LON = 78.4867
API_BASE = 'http://localhost:3000'

# Session state initialization
if 'streaming' not in st.session_state:
    st.session_state.streaming = False
if 'cap' not in st.session_state:
    st.session_state.cap = None
if 'detections' not in st.session_state:
    st.session_state.detections = deque(maxlen=100)
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'detection_count' not in st.session_state:
    st.session_state.detection_count = 0
if 'stop_streaming' not in st.session_state:
    st.session_state.stop_streaming = False
if 'alert_sent' not in st.session_state:
    st.session_state.alert_sent = False
if 'alert_response' not in st.session_state:
    st.session_state.alert_response = None
if 'current_lat' not in st.session_state:
    st.session_state.current_lat = DEFAULT_LAT
if 'current_lon' not in st.session_state:
    st.session_state.current_lon = DEFAULT_LON
if 'hospitals' not in st.session_state:
    st.session_state.hospitals = []
if 'fps' not in st.session_state:
    st.session_state.fps = 0
if 'last_detection' not in st.session_state:
    st.session_state.last_detection = None
if 'video_ended' not in st.session_state:
    st.session_state.video_ended = False
if 'is_camera' not in st.session_state:
    st.session_state.is_camera = True
if 'server_response' not in st.session_state:
    st.session_state.server_response = None
if 'alert_history' not in st.session_state:
    st.session_state.alert_history = deque(maxlen=20)
if 'video_max_detection' not in st.session_state:
    st.session_state.video_max_detection = None
if 'camera_last_alert_time' not in st.session_state:
    st.session_state.camera_last_alert_time = 0

# ==========================================
# ULTRA-FAST DETECTOR - OPTIMIZED
# ==========================================
class UltraFastDetector:
    """Extremely optimized detector for maximum speed"""
    
    def __init__(self):
        try:
            self.yolo_model = YOLO(YOLO_MODEL_PATH)
            
            if torch.cuda.is_available():
                self.yolo_model.to('cuda')
                self.device = 'cuda'
                torch.backends.cudnn.benchmark = True
                self.use_half = True
            else:
                self.device = 'cpu'
                self.use_half = False
            
            if hasattr(self.yolo_model, 'model'):
                self.yolo_model.model.eval()
                for param in self.yolo_model.model.parameters():
                    param.requires_grad = False
            
            self.fire_lower1 = np.array([0, 120, 70], dtype=np.uint8)
            self.fire_upper1 = np.array([10, 255, 255], dtype=np.uint8)
            self.fire_lower2 = np.array([170, 120, 70], dtype=np.uint8)
            self.fire_upper2 = np.array([180, 255, 255], dtype=np.uint8)
            
            self.last_fire_check = 0
            self.has_fire_cached = False
            
            self.process_size = 256
            self.tiny_size = (40, 30)
            
            st.success("🚀 Ultra-fast mode activated with optimizations!")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            self.yolo_model = None
    
    def quick_severity(self, bbox_coords, img_shape, crop_img):
        """Lightning-fast severity calculation"""
        x1, y1, x2, y2 = bbox_coords
        
        bbox_area = (x2 - x1) * (y2 - y1)
        img_area = img_shape[0] * img_shape[1]
        area_ratio = bbox_area / img_area
        
        center_x = (x1 + x2) / 2
        distance_from_center = abs(center_x - img_shape[1]/2) / img_shape[1]
        
        score = 0
        
        if area_ratio > 0.15:
            score += 4
        elif area_ratio > 0.08:
            score += 2
        elif area_ratio > 0.03:
            score += 1
        
        if distance_from_center < 0.15:
            score += 3
        elif distance_from_center < 0.3:
            score += 1
        
        if x2 - x1 > 0:
            aspect_ratio = (x2 - x1) / max((y2 - y1), 1)
            if 1.2 < aspect_ratio < 2.8:
                score += 2
        
        if crop_img.size > 0 and score < 6:
            tiny_dark = cv2.resize(crop_img, (16, 16))
            if len(tiny_dark.shape) == 3:
                gray = cv2.cvtColor(tiny_dark, cv2.COLOR_BGR2GRAY)
                dark_ratio = np.sum(gray < 40) / gray.size
                if dark_ratio > 0.2:
                    score += 2
        
        if score >= 7:
            return "Severe", score
        elif score >= 4:
            return "Moderate", score
        else:
            return "Minor", score
    
    def check_fire_fast(self, frame):
        """Ultra-optimized fire detection"""
        current_time = time.time()
        
        if current_time - self.last_fire_check < 1.5:
            return self.has_fire_cached
        
        try:
            tiny = cv2.resize(frame, self.tiny_size)
            hsv = cv2.cvtColor(tiny, cv2.COLOR_BGR2HSV)
            
            mask1 = cv2.inRange(hsv, self.fire_lower1, self.fire_upper1)
            mask2 = cv2.inRange(hsv, self.fire_lower2, self.fire_upper2)
            
            fire_pixels = np.count_nonzero(mask1) + np.count_nonzero(mask2)
            total_pixels = mask1.size
            
            self.has_fire_cached = (fire_pixels / total_pixels) > 0.04
            self.last_fire_check = current_time
            
        except:
            self.has_fire_cached = False
        
        return self.has_fire_cached
    
    def process_frame_ultra_fast(self, frame):
        """Ultra-optimized processing"""
        if self.yolo_model is None:
            return [], False
        
        detections = []
        has_fire = False
        
        try:
            fire_thread = threading.Thread(target=lambda: self.check_fire_fast(frame))
            fire_thread.start()
            
            h, w = frame.shape[:2]
            small_frame = cv2.resize(frame, (self.process_size, self.process_size))
            
            scale_x = w / self.process_size
            scale_y = h / self.process_size
            
            with torch.no_grad():
                results = self.yolo_model.predict(
                    small_frame,
                    conf=0.25,
                    iou=0.45,
                    verbose=False,
                    augment=False,
                    half=self.use_half,
                    imgsz=self.process_size,
                    device=self.device,
                    max_det=6
                )
            
            fire_thread.join()
            has_fire = self.has_fire_cached
            
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    conf = float(box.conf[0])
                    if conf < 0.3:
                        continue
                        
                    coords = box.xyxy[0].cpu().numpy()
                    
                    x1 = max(0, int(coords[0] * scale_x))
                    y1 = max(0, int(coords[1] * scale_y))
                    x2 = min(w, int(coords[2] * scale_x))
                    y2 = min(h, int(coords[3] * scale_y))
                    
                    if (x2 - x1) < 20 or (y2 - y1) < 20:
                        continue
                    
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    
                    severity, score = self.quick_severity((x1, y1, x2, y2), frame.shape, crop)
                    
                    detections.append({
                        'confidence': conf,
                        'bbox': (x1, y1, x2, y2),
                        'severity': severity,
                        'score': score,
                        'has_fire': has_fire,
                        'timestamp': time.time()
                    })
            
        except Exception as e:
            pass
        
        return detections, has_fire

# ==========================================
# OPTIMIZED UTILITIES - FIXED VERSION
# ==========================================
def send_alert_fast(lat, lon, description, has_fire=False):
    """Fast alert sending - FIXED VERSION"""
    try:
        # Use the exact parameter names that work with curl
        payload = {
            "lat": float(lat),
            "lon": float(lon),
            "description": str(description),
            "fire_incident": bool(has_fire)  # Changed from "fire" to "fire_incident"
        }
        
        response = requests.post(
            f"{API_BASE}/api/accident-alert",
            json=payload,
            timeout=10,
            headers={"Content-Type": "application/json"}  # Explicit header
        )
        
        # IMPROVED RESPONSE HANDLING
        if response.status_code == 200:
            try:
                response_data = response.json()
                
                # Check if the server indicates success in the JSON
                if response_data.get('success') == True:
                    return True, {
                        "status": "success",
                        "status_code": response.status_code,
                        "server_response": response_data,
                        "request_sent": payload,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    # Server returned 200 but success=false
                    return False, {
                        "status": "server_error",
                        "status_code": response.status_code,
                        "server_response": response_data,
                        "request_sent": payload,
                        "timestamp": datetime.now().isoformat(),
                        "error": f"Server returned success=false: {response_data.get('message', 'Unknown error')}"
                    }
                    
            except ValueError as e:
                # JSON parsing failed
                return False, {
                    "status": "parse_error",
                    "status_code": response.status_code,
                    "server_response": response.text,
                    "request_sent": payload,
                    "timestamp": datetime.now().isoformat(),
                    "error": f"Failed to parse JSON response: {str(e)}"
                }
        else:
            # Non-200 status code
            error_text = response.text
            try:
                error_json = response.json()
                return False, {
                    "status": "http_error",
                    "status_code": response.status_code,
                    "server_response": error_json,
                    "request_sent": payload,
                    "timestamp": datetime.now().isoformat(),
                    "error": f"HTTP {response.status_code}: {error_json.get('error', error_json.get('message', 'Unknown error'))}"
                }
            except ValueError:
                return False, {
                    "status": "http_error",
                    "status_code": response.status_code,
                    "server_response": error_text,
                    "request_sent": payload,
                    "timestamp": datetime.now().isoformat(),
                    "error": f"HTTP {response.status_code}: {error_text}"
                }
            
    except requests.exceptions.Timeout:
        return False, {
            "status": "timeout",
            "error": "Request timeout - server not responding within 10 seconds",
            "request_sent": payload,
            "timestamp": datetime.now().isoformat()
        }
    except requests.exceptions.ConnectionError:
        return False, {
            "status": "connection_error", 
            "error": f"Connection failed - cannot reach server at {API_BASE}",
            "request_sent": payload,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return False, {
            "status": "exception",
            "error": str(e),
            "error_type": type(e).__name__,
            "request_sent": payload,
            "timestamp": datetime.now().isoformat()
        }

def test_server_connection():
    """Test if server is reachable - FIXED VERSION"""
    try:
        # Test with the same payload structure that works
        test_payload = {
            "lat": 17.3850,
            "lon": 78.4867, 
            "description": "Connection test",
            "fire_incident": False
        }
        
        response = requests.post(
            f"{API_BASE}/api/accident-alert",
            json=test_payload,
            timeout=5,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            try:
                response_data = response.json()
                if response_data.get('success') == True:
                    return True, response_data
                else:
                    return False, {"error": f"Server returned success=false: {response_data.get('message', 'Unknown')}"}
            except ValueError:
                return False, {"error": "Invalid JSON response"}
        else:
            return False, {"error": f"Server returned {response.status_code}: {response.text}"}
    except Exception as e:
        return False, {"error": str(e)}

def find_hospitals_fast(lat, lon, radius=0.15):
    """Fast hospital search"""
    try:
        query = f'[out:json];node["amenity"="hospital"]({lat-radius},{lon-radius},{lat+radius},{lon+radius});out;'
        response = requests.get(
            f"https://overpass-api.de/api/interpreter?data={quote(query)}", 
            timeout=3
        )
        
        if response.status_code == 200:
            data = response.json()
            hospitals = []
            for el in data.get('elements', []):
                if el.get('type') == 'node':
                    name = el.get('tags', {}).get('name', 'Unknown Hospital')
                    hospitals.append({
                        'name': name,
                        'lat': el['lat'],
                        'lon': el['lon'],
                        'distance': np.sqrt((el['lat']-lat)**2 + (el['lon']-lon)**2) * 111
                    })
            hospitals.sort(key=lambda x: x['distance'])
            return hospitals[:5]
    except Exception as e:
        pass
    return []

def draw_box_fast(frame, det):
    """Minimal box drawing"""
    x1, y1, x2, y2 = det['bbox']
    
    if det['severity'] == 'Severe':
        color = (0, 0, 255)
        thickness = 3
    elif det['severity'] == 'Moderate':
        color = (0, 165, 255)
        thickness = 2
    else:
        color = (0, 255, 0)
        thickness = 2
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    label = f"{det['severity'][:1]}:{det['score']}"
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    
    cv2.rectangle(frame, (x1, y1-label_size[1]-4), (x1+label_size[0]+4, y1), color, -1)
    cv2.putText(frame, label, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    return frame

@st.cache_resource
def get_detector():
    return UltraFastDetector()

# ==========================================
# OPTIMIZED STREAMING WITH SMART ALERTS
# ==========================================
def stream_ultra_fast(video_placeholder, metrics_placeholder):
    """Maximum speed streaming with smart alert logic"""
    detector = get_detector()
    frame_times = deque(maxlen=15)
    last_metric_update = 0
    
    display_every = 3
    process_every = 2
    frame_counter = 0
    
    display_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    while st.session_state.streaming and st.session_state.cap:
        if st.session_state.stop_streaming:
            break
        
        t0 = time.time()
        
        ret, frame = st.session_state.cap.read()
        if not ret:
            if st.session_state.is_camera:
                time.sleep(0.01)
                continue
            else:
                # VIDEO ENDED - Send final alert if any detections
                st.session_state.video_ended = True
                st.session_state.streaming = False
                
                # Send alert for max detection in video
                if st.session_state.video_max_detection:
                    det = st.session_state.video_max_detection
                    success, result = send_alert_fast(
                        st.session_state.current_lat,
                        st.session_state.current_lon,
                        f"VIDEO ANALYSIS COMPLETE - Maximum severity detected: {det['severity']} (Score: {det['score']}, Confidence: {det['confidence']:.1%})",
                        det.get('has_fire', False)
                    )
                    
                    alert_record = {
                        'timestamp': datetime.now().isoformat(),
                        'success': success,
                        'response': result,
                        'type': 'video_final',
                        'detection_score': det['score'],
                        'severity': det['severity']
                    }
                    st.session_state.alert_history.append(alert_record)
                    st.session_state.alert_sent = success
                    st.session_state.alert_response = result
                    st.session_state.server_response = result.get('server_response', 'No response')
                
                if hasattr(st.session_state, 'last_valid_frame'):
                    display_frame = cv2.resize(st.session_state.last_valid_frame, (640, 480))
                    rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    
                    overlay = rgb.copy()
                    cv2.rectangle(overlay, (150, 200), (490, 280), (0, 0, 0), -1)
                    cv2.putText(overlay, "VIDEO ENDED", (180, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(overlay, "Alert sent for max detection", (160, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
                    video_placeholder.image(overlay, channels="RGB", use_container_width=True)
                
                with metrics_placeholder.container():
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Frames", st.session_state.frame_count)
                    col2.metric("Detections", st.session_state.detection_count)
                    col3.metric("Final FPS", f"{st.session_state.fps:.1f}")
                    latest_severity = st.session_state.last_detection['severity'] if st.session_state.last_detection else "-"
                    col4.metric("Latest", latest_severity)
                
                break
        
        st.session_state.last_valid_frame = frame.copy()
        
        st.session_state.frame_count += 1
        frame_counter += 1
        
        if frame_counter % process_every == 0:
            detections, has_fire = detector.process_frame_ultra_fast(frame)
            
            if detections:
                st.session_state.detection_count += len(detections)
                latest_detection = None
                
                for det in detections:
                    st.session_state.detections.append(det)
                    frame = draw_box_fast(frame, det)
                    
                    if det['score'] >= 5 and (latest_detection is None or det['score'] > latest_detection['score']):
                        latest_detection = det
                
                # Update max detection for video
                if not st.session_state.is_camera and latest_detection:
                    if (st.session_state.video_max_detection is None or 
                        latest_detection['score'] > st.session_state.video_max_detection['score']):
                        st.session_state.video_max_detection = latest_detection
                
                # CAMERA MODE: Send alert for every significant detection (with cooldown)
                if st.session_state.is_camera and latest_detection:
                    current_time = time.time()
                    time_since_last_alert = current_time - st.session_state.camera_last_alert_time
                    
                    # Send alert if score >= 7 and 15 seconds have passed
                    if latest_detection['score'] >= 7 and time_since_last_alert > 15:
                        def send_camera_alert():
                            success, result = send_alert_fast(
                                st.session_state.current_lat,
                                st.session_state.current_lon,
                                f"LIVE CAMERA DETECTION - {latest_detection['severity']} accident! Score: {latest_detection['score']}, Confidence: {latest_detection['confidence']:.1%}",
                                has_fire
                            )
                            
                            alert_record = {
                                'timestamp': datetime.now().isoformat(),
                                'success': success,
                                'response': result,
                                'type': 'camera_live',
                                'detection_score': latest_detection['score'],
                                'severity': latest_detection['severity']
                            }
                            st.session_state.alert_history.append(alert_record)
                            st.session_state.alert_sent = success
                            st.session_state.alert_response = result
                            st.session_state.server_response = result.get('server_response', 'No response')
                            st.session_state.camera_last_alert_time = current_time
                        
                        alert_thread = threading.Thread(target=send_camera_alert)
                        alert_thread.daemon = True
                        alert_thread.start()
                
                st.session_state.last_detection = latest_detection
        
        if frame_counter % display_every == 0:
            display_frame = cv2.resize(frame, (640, 480))
            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb, channels="RGB", use_container_width=True)
        
        frame_time = time.time() - t0
        frame_times.append(frame_time)
        st.session_state.fps = len(frame_times) / sum(frame_times) if frame_times else 0
        
        current_time = time.time()
        if current_time - last_metric_update > 2.0:
            with metrics_placeholder.container():
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Frames", st.session_state.frame_count)
                col2.metric("Detections", st.session_state.detection_count)
                col3.metric("FPS", f"{st.session_state.fps:.1f}")
                latest_severity = st.session_state.last_detection['severity'] if st.session_state.last_detection else "-"
                col4.metric("Latest", latest_severity)
            last_metric_update = current_time
        
        elapsed = time.time() - t0
        target_time = 1/45
        sleep_time = max(0.001, target_time - elapsed)
        time.sleep(sleep_time)
    
    if st.session_state.cap and not st.session_state.is_camera:
        st.session_state.cap.release()
        st.session_state.cap = None

# ==========================================
# MAIN APP
# ==========================================
def main():
    st.title("🚨 Ultra-Fast Accident Detection")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙ Settings")
        
        # Server Status Check
        if st.button("🔌 Test Server Connection"):
            with st.spinner("Testing server connection..."):
                success, result = test_server_connection()
                if success:
                    st.success(f"✅ Server is running: {result.get('message', 'Connected')}")
                else:
                    st.error(f"❌ Server connection failed: {result.get('error', 'Unknown error')}")
        
        source = st.radio("Source", ["Camera", "Video"], horizontal=True)
        
        if source == "Camera":
            cam_idx = st.selectbox("Camera Index", [0, 1, 2, 3], index=0)
            uploaded = None
            st.session_state.is_camera = True
        else:
            cam_idx = 0
            uploaded = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'], label_visibility="collapsed")
            st.session_state.is_camera = False
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.current_lat = st.number_input("Latitude", value=DEFAULT_LAT, format="%.6f", key="lat_input")
        with col2:
            st.session_state.current_lon = st.number_input("Longitude", value=DEFAULT_LON, format="%.6f", key="lon_input")
        
        st.markdown("---")
        with st.expander("🚀 Alert Strategy"):
            st.info("""
            **Camera Mode:**
            - Sends alert for EVERY severe detection (score ≥ 7)
            - 15 second cooldown between alerts
            
            **Video Mode:**
            - Tracks maximum severity detection
            - Sends ONE final alert at video end
            - Contains highest score found
            """)
        
        # Alert History Display
        st.markdown("---")
        st.subheader("📋 Alert History")
        
        if len(st.session_state.alert_history) > 0:
            for i, alert in enumerate(list(st.session_state.alert_history)[::-1]):  # Show newest first
                alert_type_emoji = {
                    'camera_live': '📹',
                    'video_final': '🎬',
                    'manual': '👤'
                }
                emoji = alert_type_emoji.get(alert['type'], '📨')
                
                # Extract time from timestamp
                alert_time = alert['timestamp'][11:19] if len(alert['timestamp']) > 19 else alert['timestamp']
                
                with st.expander(f"{emoji} Alert {len(st.session_state.alert_history)-i} - {alert_time}", expanded=(i==0)):
                    status_icon = '✅' if alert['success'] else '❌'
                    st.write(f"**Status:** {status_icon} {'Success' if alert['success'] else 'Failed'}")
                    st.write(f"**Type:** {alert['type'].replace('_', ' ').title()}")
                    
                    if 'detection_score' in alert:
                        st.write(f"**Score:** {alert['detection_score']}")
                    if 'severity' in alert:
                        st.write(f"**Severity:** {alert['severity']}")
                    
                    if alert['success'] and 'server_response' in alert['response']:
                        server_resp = alert['response']['server_response']
                        if isinstance(server_resp, dict):
                            if 'message' in server_resp:
                                st.caption("Server Response:")
                                st.code(server_resp['message'][:100] + "..." if len(str(server_resp['message'])) > 100 else server_resp['message'])
        else:
            st.info("No alerts sent yet")
        
        # Clear History Button
        if len(st.session_state.alert_history) > 0:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.alert_history.clear()
                st.rerun()
    
    # Controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 START DETECTION", type="primary", use_container_width=True):
            st.session_state.stop_streaming = False
            st.session_state.video_ended = False
            st.session_state.frame_count = 0
            st.session_state.detection_count = 0
            st.session_state.detections.clear()
            st.session_state.alert_sent = False
            st.session_state.alert_response = None
            st.session_state.last_detection = None
            st.session_state.server_response = None
            st.session_state.video_max_detection = None
            st.session_state.camera_last_alert_time = 0
            
            if source == "Camera":
                cap = cv2.VideoCapture(cam_idx)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                st.session_state.is_camera = True
            else:
                if uploaded:
                    with open("temp_video.mp4", "wb") as f:
                        f.write(uploaded.getvalue())
                    cap = cv2.VideoCapture("temp_video.mp4")
                    st.session_state.is_camera = False
                else:
                    st.warning("📹 Please upload a video file first!")
                    cap = None
            
            if cap and cap.isOpened():
                st.session_state.cap = cap
                st.session_state.streaming = True
            else:
                st.error("❌ Failed to open video source")
    
    with col2:
        if st.button("⏹ STOP DETECTION", use_container_width=True):
            st.session_state.stop_streaming = True
            if st.session_state.cap:
                st.session_state.cap.release()
                st.session_state.cap = None
            st.session_state.streaming = False
            st.session_state.video_ended = False
            st.rerun()
    
    with col3:
        if st.session_state.video_ended:
            status_text = "🟡 Video Ended"
        elif st.session_state.streaming:
            status_text = "🟢 Active"
        else:
            status_text = "🔴 Stopped"
        
        st.metric("Status", status_text)
    
    # Real-time Metrics
    metrics_placeholder = st.empty()
    
    st.markdown("---")
    
    # Main Content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        video_placeholder = st.empty()
        
        if st.session_state.video_ended:
            video_placeholder.warning("🎬 Video processing completed! Click 'START DETECTION' to replay the video.")
        elif st.session_state.streaming and st.session_state.cap:
            stream_ultra_fast(video_placeholder, metrics_placeholder)
        else:
            video_placeholder.info("🎬 Click 'START DETECTION' to begin real-time accident monitoring")
    
    with col2:
        st.subheader("🚨 Emergency Panel")
        
        # Enhanced Server Response Display
        if st.session_state.server_response is not None:
            st.markdown("---")
            st.subheader("📡 Latest Server Response")
            
            if st.session_state.alert_sent:
                st.success("✅ Alert Successfully Sent!")
            else:
                st.error("❌ Failed to Send Alert!")
            
            with st.container():
                st.markdown('<div class="alert-response">', unsafe_allow_html=True)
                
                server_response = st.session_state.server_response
                
                if isinstance(server_response, dict):
                    if 'message' in server_response:
                        formatted_message = server_response['message'].replace('\n', '  \n')
                        st.markdown(f"**Message:**  \n{formatted_message}")
                    
                    if 'success' in server_response:
                        st.markdown(f"**Success:** `{server_response['success']}`")
                    
                    if 'alert_id' in server_response:
                        st.markdown(f"**Alert ID:** `{server_response['alert_id']}`")
                    
                    if 'notified_amenities' in server_response:
                        st.markdown(f"**Amenities Notified:** `{server_response['notified_amenities']}`")
                    
                    if 'fire_incident' in server_response:
                        st.markdown(f"**Fire Incident:** `{server_response['fire_incident']}`")
                    
                    if 'details' in server_response and server_response['details']:
                        st.markdown("**Notified Facilities:**")
                        for detail in server_response['details']:
                            emoji = "🏥" if detail.get('type') == 'hospital' else "🚔" if detail.get('type') == 'police' else "🚒"
                            st.write(f"{emoji} **{detail.get('name', 'Unknown')}**")
                            st.write(f"   Type: {detail.get('type', 'N/A')}")
                            st.write(f"   Email: {detail.get('email', 'N/A')}")
                            st.write(f"   Distance: {detail.get('distance_km', 'N/A')} km")
                    
                    with st.expander("📋 Raw Server Response"):
                        st.json(server_response)
                else:
                    st.code(str(server_response), language='json')
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with st.expander("🔍 Request Details", expanded=False):
                if st.session_state.alert_response and 'request_sent' in st.session_state.alert_response:
                    st.write("**Request Sent:**")
                    st.json(st.session_state.alert_response['request_sent'])
                
                if st.session_state.alert_response and 'status_code' in st.session_state.alert_response:
                    st.write(f"**HTTP Status:** `{st.session_state.alert_response.get('status_code', 'N/A')}`")
                
                if st.session_state.alert_response and 'error' in st.session_state.alert_response:
                    st.error(f"**Error:** {st.session_state.alert_response['error']}")
        
        # Map
        st.markdown("---")
        m = folium.Map(
            location=[st.session_state.current_lat, st.session_state.current_lon], 
            zoom_start=13,
            tiles='OpenStreetMap'
        )
        folium.Marker(
            [st.session_state.current_lat, st.session_state.current_lon],
            popup="Accident Location",
            icon=folium.Icon(color='red', icon='warning-sign')
        ).add_to(m)
        st_folium(m, width=400, height=250)
        
        # Emergency Actions
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🏥 Find Hospitals", use_container_width=True):
                with st.spinner("Searching nearby hospitals..."):
                    hospitals = find_hospitals_fast(st.session_state.current_lat, st.session_state.current_lon)
                    st.session_state.hospitals = hospitals
                    if hospitals:
                        st.success(f"✅ Found {len(hospitals)} hospitals")
                    else:
                        st.warning("❌ No hospitals found")
        
        with col_b:
            if st.button("🚨 Manual Alert", use_container_width=True):
                success, result = send_alert_fast(
                    st.session_state.current_lat, 
                    st.session_state.current_lon, 
                    "Manual emergency alert triggered by operator", 
                    False
                )
                
                # Add manual alert to history
                alert_record = {
                    'timestamp': datetime.now().isoformat(),
                    'success': success,
                    'response': result,
                    'type': 'manual',
                    'description': 'Manual alert'
                }
                st.session_state.alert_history.append(alert_record)
                
                st.session_state.alert_sent = success
                st.session_state.alert_response = result
                st.session_state.server_response = result.get('server_response', 'No response')
                
                if success:
                    st.success("✅ Manual alert sent!")
                else:
                    st.error("❌ Failed to send alert!")
                st.rerun()
    
    # Detection History
    if st.session_state.detections:
        st.markdown("---")
        st.subheader("📊 Detection Analytics")
        
        severe_count = sum(1 for d in st.session_state.detections if d['severity'] == 'Severe')
        moderate_count = sum(1 for d in st.session_state.detections if d['severity'] == 'Moderate')
        minor_count = sum(1 for d in st.session_state.detections if d['severity'] == 'Minor')
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("🔴 Severe", severe_count)
        col2.metric("🟠 Moderate", moderate_count)
        col3.metric("🟢 Minor", minor_count)
        col4.metric("Total", len(st.session_state.detections))
        col5.metric("Alerts Sent", len(st.session_state.alert_history))
        
        significant_detections = [d for d in st.session_state.detections if d['score'] >= 4]
        
        if significant_detections:
            st.markdown("**Recent Significant Detections:**")
            recent_detections = list(reversed(significant_detections))[:8]
            
            for det in recent_detections:
                emoji = {"Severe": "🔴", "Moderate": "🟠", "Minor": "🟢"}.get(det['severity'], "⚪")
                ts = datetime.fromtimestamp(det['timestamp']).strftime('%H:%M:%S')
                fire_emoji = " 🔥" if det.get('has_fire', False) else ""
                st.write(
                    f"{emoji} **{det['severity']}** | "
                    f"Score: `{det['score']}` | "
                    f"Conf: `{det['confidence']:.1%}` | "
                    f"Time: `{ts}`{fire_emoji}"
                )
        
        # Show max detection info for video mode
        if not st.session_state.is_camera and st.session_state.video_max_detection:
            st.markdown("---")
            st.info(f"**Maximum Detection:** {st.session_state.video_max_detection['severity']} "
                   f"(Score: {st.session_state.video_max_detection['score']}) - "
                   f"Alert will be sent at video end")
    
    # Hospital Information
    if st.session_state.hospitals:
        st.markdown("---")
        st.subheader("🏥 Nearby Medical Facilities")
        
        for i, hospital in enumerate(st.session_state.hospitals[:3], 1):
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{i}. {hospital['name']}**")
                with col2:
                    st.write(f"`{hospital['distance']:.1f} km`")
            if i < len(st.session_state.hospitals[:3]):
                st.divider()
    
    # Footer info
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #fff; padding: 10px;">
        <p><strong>Alert Strategy:</strong> Camera = Every detection (15s cooldown) | Video = One final alert (max severity)</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()