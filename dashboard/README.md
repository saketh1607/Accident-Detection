# Guardian Connect - Emergency Response Network

A real-time emergency response network connecting hospitals, police stations, and fire stations with automatic accident alerts based on proximity.

## Features

- 🌍 **Automatic OSM Import**: Fetches and registers amenities from OpenStreetMap
- 📱 **Mobile App**: React Native (Expo) app for Android & iOS
- 🔐 **Email-Only Login**: No password required, just registered email
- 🚨 **Real-Time Alerts**: Flash notifications for nearby accidents (5km radius)
- 🗺️ **Interactive Map**: View all amenities and active alerts
- 📍 **Distance-Based**: Automatically calculates and displays distance from incidents
- 🔔 **Live Updates**: Real-time dashboard updates via Supabase subscriptions

## Setup Instructions

### 1. Install Backend Dependencies

```bash
npm install
```

### 2. Import OSM Data

This will automatically fetch all hospitals, police stations, and fire stations from OpenStreetMap within the specified region and register them in your database:

```bash
npm run import-osm
```

The script will:
- Fetch amenities from OSM
- Generate unique emails for each (e.g., `facilityname.hospital1@guardianconnect.emergency`)
- Save to Supabase database
- Display all login credentials

**Save the credentials output - you'll need these emails to login to the mobile app!**

### 3. Start the Backend API

```bash
npm run dev
```

The API will run on `http://localhost:3000`

**Endpoints:**
- `POST /api/accident-alert` - Create new accident alert
- `GET /api/amenities` - List all registered amenities
- `GET /api/health` - Health check

### 4. Setup Mobile App

Navigate to the mobile app directory:

```bash
cd mobile-app
npm install
```

### 5. Run the Mobile App

For Android:
```bash
npm run android
```

For iOS:
```bash
npm run ios
```

For web (development):
```bash
npm run web
```

## Using the System

### Login to Mobile App

1. Open the app
2. Enter any registered email from the import output
3. No password needed - just tap Login

### Create an Accident Alert

Send a POST request to create an alert:

```bash
curl -X POST http://localhost:3000/api/accident-alert \
  -H "Content-Type: application/json" \
  -d '{
    "lat": 17.5385,
    "lon": 78.3965,
    "description": "Two-wheeler accident near Pragathi Nagar"
  }'
```

All amenities within 5km will receive:
- Real-time notification in dashboard
- Flash alert animation
- Distance from incident
- Timestamp

### Dashboard Features

- **Flash Alerts**: New unread alerts pulse in red
- **Distance Display**: Shows how far each incident is from your facility
- **Read Status**: Tap any alert to mark as read
- **Auto-Refresh**: Pull down to refresh or wait for real-time updates
- **Stats**: See total alerts and unread count

### Map View

- 🏥 **Green markers**: Hospitals
- 🚔 **Blue markers**: Police Stations
- 🚒 **Red markers**: Fire Stations
- ⚠️ **Yellow markers**: Active accident alerts
- Blue circle: Your facility's 5km notification radius
- Red circles: 5km radius around each accident

## Database Schema

### Tables

**amenities**
- Stores all registered emergency facilities
- Auto-populated from OSM data
- Contains location, contact info, and login email

**accident_alerts**
- Each reported accident/incident
- Includes location and description
- Tracks resolved status

**alert_notifications**
- Junction table linking amenities to relevant alerts
- Stores distance and read status
- Only created for amenities within 5km

## Tech Stack

**Backend:**
- Node.js + Express
- Supabase (PostgreSQL)
- OpenStreetMap Overpass API

**Mobile:**
- React Native (Expo)
- React Navigation
- React Native Maps
- Supabase JS Client

**Real-time:**
- Supabase Realtime subscriptions
- PostgreSQL triggers

## Configuration

### Region Bounds

Edit bounds in `scripts/import-osm-data.js`:

```javascript
const BOUNDS = {
  south: 17.450000,
  west: 78.300000,
  north: 17.720000,
  east: 78.560000
};
```

### Notification Radius

Change radius in `server.js`:

```javascript
const RADIUS_KM = 5; // Change to desired km
```

## Testing the Alert System

1. Import OSM data and note login credentials
2. Login to mobile app with any facility email
3. Send a test alert via API (see example above)
4. Watch the dashboard for real-time flash alert
5. Check the map to see the incident marker
6. Tap the alert to mark as read

## Production Deployment

### Backend
- Deploy to any Node.js hosting (Heroku, Railway, Render)
- Set environment variables from `.env`
- Run `npm run import-osm` once after deployment

### Mobile App
- Build with EAS: `eas build --platform all`
- Submit to app stores: `eas submit`
- Configure push notifications with FCM/APNs

## Support

For issues or questions about Guardian Connect, check the database tables and API responses for debugging.
