# 🚨 Guardian Connect - AI-Powered Accident Detection System

An intelligent real-time accident detection system using YOLO object detection, computer vision, and automated emergency response notifications.

## 📋 Overview

Guardian Connect is a comprehensive accident detection and emergency response system that:
- **Detects accidents in real-time** from live camera feeds or video files
- **Analyzes accident severity** using multi-feature analysis (area, position, debris, fire detection)
- **Automatically alerts emergency services** (hospitals, police, fire stations)
- **Provides location-based emergency response** with nearest facility identification

## ✨ Key Features

### 🎯 Accident Detection
- **YOLO-based object detection** for accident identification
- **Multi-severity classification**: Severe, Moderate, Minor (scored 0-12)
- **Fire detection** using color-based HSV analysis
- **Person detection** on roadways for critical incidents
- **GPU acceleration** with FP16 support for high-speed processing

### 📹 Dual Mode Operation

**Camera Mode (Live Detection):**
- Processes live camera feed in real-time
- Sends alert for **every severe detection** (score ≥ 7)
- 15-second cooldown between alerts
- Ideal for fixed surveillance cameras

**Video Mode (Batch Analysis):**
- Analyzes pre-recorded video files
- Tracks maximum severity detection throughout video
- Sends **one final alert** with highest score at video end
- Perfect for incident review and forensic analysis

### 🚨 Smart Alert System
- **Automatic emergency notification** to nearest facilities
- **Fire-aware routing**: 
  - No fire: Alerts 1 hospital + 1 police station
  - Fire detected: Alerts 1 hospital + 1 police + 1 fire station
- **Distance-based prioritization** using Haversine formula
- **Alert history tracking** with full response logging

### 📊 Real-Time Analytics
- Live FPS monitoring
- Detection count and severity distribution
- Frame-by-frame analysis display
- Historical detection tracking

## 🛠️ Tech Stack

### Frontend/Detection System
- **Python 3.8+**
- **Streamlit** - Web interface
- **OpenCV** - Video processing
- **Ultralytics YOLO** - Object detection
- **PyTorch** - Deep learning framework
- **CUDA** - GPU acceleration (optional)

### Backend API
- **Node.js** with Express
- **Supabase** - PostgreSQL database
- **CORS** - Cross-origin resource sharing

### Database Schema

**accident_alerts**
- `id` (UUID)
- `latitude` (FLOAT)
- `longitude` (FLOAT)
- `description` (TEXT)
- `created_at` (TIMESTAMP)

**amenities**
- `id` (UUID)
- `name` (TEXT)
- `type` (TEXT) - 'hospital', 'police', 'fire'
- `latitude` (FLOAT)
- `longitude` (FLOAT)
- `email` (TEXT)
- `address` (TEXT)

**alert_notifications**
- `id` (UUID)
- `alert_id` (UUID) - Foreign key to accident_alerts
- `amenity_id` (UUID) - Foreign key to amenities
- `distance_km` (FLOAT)
- `created_at` (TIMESTAMP)

## 🚀 Installation

### Prerequisites
```bash
# Python 3.8 or higher
python --version

# Node.js 16+ and npm
node --version
npm --version

# CUDA Toolkit (optional, for GPU acceleration)
nvcc --version
```

### 1. Clone Repository
```bash
git clone <my git url>
cd Accident-Detector
pip install -r requirements.txt
```

### 2. Setup Backend API

```bash
# Navigate to backend directory
cd backend

# Install dependencies
npm install

# Create .env file
cat > .env << EOL
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_SUPABASE_ANON_KEY=your_supabase_anon_key
PORT=3000
EOL

# Start server
node server.js or npm run dev
```

### 3. Setup Detection System

```bash
# Navigate to Accident directory
cd ..

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install streamlit opencv-python ultralytics torch torchvision
pip install requests python-dotenv folium streamlit-folium numpy
or
pip install -r requirements.txt

# Download YOLO model weights
# Place your trained YOLO model at:
# models/best.pt
```

### 4. Configure Detection System

Update the following in your main Python file:
```python
# Model path
YOLO_MODEL_PATH = r"path/to/your/best.pt"

# Default location coordinates
DEFAULT_LAT = 17.3850  # Your latitude
DEFAULT_LON = 78.4867  # Your longitude

# API endpoint
API_BASE = 'http://localhost:3000'
```

## 📖 Usage

### Starting the System

**Terminal 1 - Backend API:**
```bash
cd dashboard
node server.js
```

**Terminal 2 - Detection System:**
```bash
cd Accident-Detector
streamlit run appst.py
```
cd dashboard
run the index.html
The web interface will open at `http://localhost:8501`

### Using the Interface

1. **Select Video Source**
   - Camera: Choose camera index (0, 1, 2, etc.)
   - Video: Upload MP4, AVI, or MOV file

2. **Set Location**
   - Enter accident location coordinates
   - Or use default location

3. **Start Detection**
   - Click "🚀 START DETECTION"
   - Monitor real-time processing
   - View detections and severity scores

4. **Emergency Response**
   - Automatic alerts sent for severe incidents
   - Manual alert button for operator intervention
   - Find nearby hospitals button
   - View alert history in sidebar

### Alert Logic

**Camera Mode:**
```
Detection (score ≥ 7) → Wait 15s → Alert → Repeat
```

**Video Mode:**
```
Process entire video → Track max score → Video ends → Send one alert
```

## 🎮 GPU Acceleration

For maximum performance with NVIDIA GPUs:

```bash
# Install CUDA Toolkit (if not already installed)
# Download from: https://developer.nvidia.com/cuda-downloads

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

**Expected Performance:**
- CPU: 10-20 FPS
- GPU (NVIDIA): 50-100+ FPS with FP16

## 📊 Severity Scoring System

Accidents are scored from 0-12 based on multiple factors:

| Factor | Max Points | Description |
|--------|-----------|-------------|
| Accident Area | 4 | Large debris field = higher score |
| Road Position | 3 | Center of road = more dangerous |
| Dark Debris/Fluids | 2 | Blood, oil, body = severe |
| Edge Density | 2 | Scattered debris = impact |
| Orientation | 3 | Fallen person/vehicle = critical |
| Contrast/Damage | 2 | High variance = severe damage |
| Person Detection | 8 | Body on road = automatic severe |

**Classification:**
- **Severe**: 8-12 points (🔴 Immediate response)
- **Moderate**: 5-7 points (🟠 Response needed)
- **Minor**: 0-4 points (🟢 Low priority)

## 🔧 API Endpoints

### POST /api/accident-alert
Create new accident alert and notify emergency services.

**Request:**
```json
{
  "lat": 17.3850,
  "lon": 78.4867,
  "description": "Severe accident detected! Score: 9",
  "fire": 0
}
```

**Response:**
```json
{
  "success": true,
  "alert_id": "uuid",
  "notified_amenities": 2,
  "fire_incident": false,
  "message": "Alert created successfully!...",
  "details": [
    {
      "name": "City Hospital",
      "type": "hospital",
      "email": "hospital@example.com",
      "distance_km": "2.34"
    }
  ]
}
```

### GET /api/amenities
List all registered emergency facilities.

### GET /api/health
Check API server status.

## 🗃️ Database Setup

```sql
-- Create accident_alerts table
CREATE TABLE accident_alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    latitude FLOAT NOT NULL,
    longitude FLOAT NOT NULL,
    description TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create amenities table
CREATE TABLE amenities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    type TEXT NOT NULL CHECK (type IN ('hospital', 'police', 'fire')),
    latitude FLOAT NOT NULL,
    longitude FLOAT NOT NULL,
    email TEXT NOT NULL,
    address TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create alert_notifications table
CREATE TABLE alert_notifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_id UUID REFERENCES accident_alerts(id) ON DELETE CASCADE,
    amenity_id UUID REFERENCES amenities(id) ON DELETE CASCADE,
    distance_km FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Sample data
INSERT INTO amenities (name, type, latitude, longitude, email, address) VALUES
('City General Hospital', 'hospital', 17.3850, 78.4867, 'contact@cityhospital.com', '123 Main St'),
('Central Police Station', 'police', 17.3900, 78.4900, 'central@police.gov', '456 Police Rd'),
('Fire Station 1', 'fire', 17.3800, 78.4850, 'fire1@fireservice.gov', '789 Fire Lane');
```

## 🐛 Troubleshooting

### Server Connection Failed
```bash
# Check if server is running
curl http://localhost:3000/api/health

# Check ports
netstat -ano | findstr :3000
```

### YOLO Model Not Loading
- Verify model path is correct
- Check file permissions
- Ensure model is compatible with ultralytics version

### Low FPS Performance
- Enable GPU acceleration
- Reduce video resolution
- Close other applications
- Check display_every and process_every settings

### No Amenities Found
- Add facilities to database using SQL or API
- Verify Supabase connection
- Check database credentials in .env

## 📈 Future Enhancements

- [ ] Email notifications to emergency services
- [ ] SMS alerts integration
- [ ] Multi-camera support
- [ ] Historical analytics dashboard
- [ ] Mobile app for emergency responders
- [ ] Audio alert detection
- [ ] Weather condition analysis
- [ ] Traffic density estimation

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 👥 Authors

- **Your Name** - Initial work - [YourGitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- Ultralytics YOLO for object detection framework
- OpenCV community for computer vision tools
- Supabase for backend infrastructure
- Streamlit for rapid UI development



**⚠️ Disclaimer**: This system is designed as an assistance tool for emergency response. It should not replace human judgment or official emergency services. Always verify detections before taking action.