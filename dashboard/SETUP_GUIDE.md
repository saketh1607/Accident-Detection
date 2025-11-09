# Guardian Connect - Complete Setup Guide

## 🎯 What You've Built

A real-time emergency response network that:
- Automatically imported **747 emergency facilities** from OpenStreetMap
  - 728 Hospitals
  - 15 Police Stations
  - 4 Fire Stations
- Provides email-only login (no passwords!)
- Sends instant alerts to facilities within 5km of accidents
- Shows live map with all emergency facilities
- Mobile app for Android & iOS

## 📋 Quick Start

### 1. Backend Setup (Already Complete!)

The database is populated with all amenities. To see credentials:
```bash
npm run import-osm
```

Or check `LOGIN_CREDENTIALS.txt` for sample logins.

### 2. Start the API Server

```bash
npm run dev
```

Server runs on `http://localhost:3000`

### 3. Setup Mobile App

```bash
cd mobile-app
npm install
npx expo start
```

Choose your platform:
- Press `a` for Android emulator
- Press `i` for iOS simulator
- Scan QR code with Expo Go app on physical device

## 🔐 How to Login

**No password required!** Just use any registered email:

### Try These Sample Logins:

**Hospitals:**
- `apollohospital.hospital694@guardianconnect.emergency`
- `kimshospital.hospital50@guardianconnect.emergency`
- `yashodahospital.hospital696@guardianconnect.emergency`

**Police Stations:**
- `afzalgunj.police1@guardianconnect.emergency`
- `bowenpally.police3@guardianconnect.emergency`

**Fire Stations:**
- `firestation2@guardianconnect.emergency`

## 🚨 Testing the Alert System

### Method 1: Using the Test Script

```bash
./test-alert.sh
```

### Method 2: Manual API Call

```bash
curl -X POST http://localhost:3000/api/accident-alert \
  -H "Content-Type: application/json" \
  -d '{
    "lat": 17.5385,
    "lon": 78.3965,
    "description": "Road accident near Kukatpally Circle"
  }'
```

### What Happens:
1. Alert created in database
2. System finds all facilities within 5km
3. Notifications sent to nearby facilities
4. Mobile app shows flash alert with:
   - Distance from facility
   - Timestamp
   - Description
   - Real-time updates

## 📱 Mobile App Features

### Dashboard
- **Flash Alerts**: New unread alerts pulse in red
- **Distance Display**: Shows how far each incident is
- **Read Status**: Tap to mark as read
- **Auto-Refresh**: Pull down or wait for real-time updates
- **Stats**: Total alerts and unread count

### Map View
- 🏥 Green markers: Hospitals
- 🚔 Blue markers: Police Stations
- 🚒 Red markers: Fire Stations
- ⚠️ Yellow markers with red circles: Active accidents
- Blue circle: Your facility's 5km notification radius

## 🗺️ Region Coverage

Currently covering Hyderabad region:
- South-West: (17.450000, 78.300000)
- North-East: (17.720000, 78.560000)

To change region, edit `scripts/import-osm-data.js`:

```javascript
const BOUNDS = {
  south: YOUR_VALUE,
  west: YOUR_VALUE,
  north: YOUR_VALUE,
  east: YOUR_VALUE
};
```

Then run: `npm run import-osm`

## 🛠️ Technical Architecture

### Database (Supabase)
- **amenities**: All emergency facilities
- **accident_alerts**: Reported incidents
- **alert_notifications**: Links alerts to nearby facilities

### Real-Time Features
- Supabase Realtime subscriptions
- Automatic distance calculation
- Push notification ready (FCM integration point available)

### API Endpoints
- `POST /api/accident-alert` - Create new alert
- `GET /api/amenities` - List all facilities
- `GET /api/health` - Health check

## 🔧 Configuration

### Change Notification Radius

Edit `server.js`:
```javascript
const RADIUS_KM = 5; // Change to your preferred radius
```

### Add Push Notifications

The app is ready for Firebase Cloud Messaging:
1. Add Firebase config to `mobile-app/app.json`
2. Add FCM logic in `DashboardScreen.js`
3. Send notifications from `server.js` when alerts created

## 📊 System Status

Run to check imported facilities:
```bash
npm run import-osm
```

Shows complete list with:
- Facility names
- Login emails
- Type breakdown

## 🎯 Next Steps

1. **Deploy Backend**:
   - Deploy to Heroku/Railway/Render
   - Set environment variables from `.env`

2. **Build Mobile App**:
   ```bash
   cd mobile-app
   eas build --platform all
   ```

3. **Add Push Notifications**:
   - Set up Firebase project
   - Configure FCM in app
   - Update server to send push notifications

4. **Production Security**:
   - Re-enable RLS policies after testing
   - Add rate limiting to API
   - Set up proper authentication

## 📝 Notes

- Email-only login is intentional for emergency response speed
- All 747 facilities are already registered in database
- System automatically calculates distances using Haversine formula
- Real-time updates via Supabase channels
- OpenStreetMap data is cached and can be re-imported anytime

## 🆘 Troubleshooting

**Can't login?**
- Use exact email from credentials list
- Check network connection
- Verify API server is running

**No alerts appearing?**
- Check if alert location is within 5km of facility
- Pull down to refresh dashboard
- Verify API call succeeded

**Import fails?**
- Check internet connection (needs OSM API access)
- Verify Supabase credentials in `.env`
- Try running again (may be rate limited)

## 📞 Emergency Response Flow

1. **Accident occurs** → Someone calls API with location
2. **System calculates** → Finds all facilities within 5km
3. **Notifications sent** → Creates alerts for nearby facilities
4. **Dashboard updates** → Real-time flash alert appears
5. **Response team acts** → Facility dispatches emergency response

---

**Built with:**
- React Native (Expo)
- Node.js + Express
- Supabase (PostgreSQL + Realtime)
- OpenStreetMap Overpass API
- React Native Maps

**Total Coverage:** 747 emergency facilities across Hyderabad
