# 🚀 Guardian Connect - Quick Start (5 Minutes)

## What You Have

✅ **747 emergency facilities** automatically registered from OpenStreetMap
✅ **Backend API** ready to receive accident alerts
✅ **Mobile app** for Android & iOS
✅ **Email-only login** - no passwords needed!

---

## Step 1: Start the Backend (30 seconds)

```bash
npm run dev
```

You should see:
```
🚨 Guardian Connect API running on port 3000
📍 POST /api/accident-alert - Create new accident alert
📋 GET /api/amenities - List all registered amenities
```

---

## Step 2: Launch Mobile App (1 minute)

Open a new terminal:

```bash
cd mobile-app
npm install
npx expo start
```

Then:
- **For Android**: Press `a`
- **For iOS**: Press `i`
- **Physical Device**: Scan QR with Expo Go app

---

## Step 3: Login (30 seconds)

Use **any of these emails** (no password!):

### Hospitals 🏥
- `apollohospital.hospital694@guardianconnect.emergency`
- `kimshospital.hospital50@guardianconnect.emergency`
- `yashodahospital.hospital696@guardianconnect.emergency`

### Police Stations 🚔
- `afzalgunj.police1@guardianconnect.emergency`
- `bowenpally.police3@guardianconnect.emergency`

### Fire Stations 🚒
- `firestation2@guardianconnect.emergency`

---

## Step 4: Test an Alert (1 minute)

Open another terminal:

```bash
./test-alert.sh
```

Or manually:
```bash
curl -X POST http://localhost:3000/api/accident-alert \
  -H "Content-Type: application/json" \
  -d '{
    "lat": 17.5385,
    "lon": 78.3965,
    "description": "Road accident near Kukatpally Circle"
  }'
```

---

## Step 5: See It Work! ✨

On your mobile app you'll see:
- 🚨 **Flash alert** with red pulsing animation
- 📍 **Distance** from your facility
- ⏰ **Timestamp** (e.g., "Just now")
- 📱 **Description** of the incident

Tap the alert to mark it as read.

---

## 🗺️ Check the Map

Switch to the **Map** tab to see:
- 🏥 Green markers: All hospitals
- 🚔 Blue markers: All police stations
- 🚒 Red markers: All fire stations
- ⚠️ Yellow markers: Active accidents
- Blue circle: Your 5km notification radius

---

## 🎯 That's It!

You now have a fully functional emergency response network with:
- 747 registered facilities
- Real-time alerts
- Distance-based notifications
- Interactive map
- Email-only access

---

## 📋 View All 747 Login Credentials

```bash
npm run import-osm
```

This shows the complete list of all registered facilities and their login emails.

---

## 🆘 Troubleshooting

**Backend won't start?**
```bash
npm install
npm run dev
```

**Mobile app won't load?**
```bash
cd mobile-app
rm -rf node_modules
npm install
npx expo start --clear
```

**Can't login?**
- Copy-paste email exactly from credentials list
- Make sure backend is running
- Check network connection

**No alerts showing?**
- Pull down to refresh
- Check if alert is within 5km of your facility
- Verify backend received the alert: `http://localhost:3000/api/health`

---

## 📚 More Information

- **Complete Setup**: `SETUP_GUIDE.md`
- **Technical Details**: `README.md`
- **Project Overview**: `PROJECT_SUMMARY.md`
- **Sample Logins**: `LOGIN_CREDENTIALS.txt`

---

**Built for:** Hyderabad Emergency Response Network
**Coverage:** 17.45°N - 17.72°N, 78.30°E - 78.56°E
**Total Facilities:** 747 (728 hospitals, 15 police, 4 fire)
