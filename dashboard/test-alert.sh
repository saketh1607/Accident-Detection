#!/bin/bash

echo "🚨 Testing Guardian Connect Alert System..."
echo ""

curl -X POST http://localhost:3000/api/accident-alert \
  -H "Content-Type: application/json" \
  -d '{
    "lat": 17.5385,
    "lon": 78.3965,
    "description": "Two-wheeler accident near Pragathi Nagar - TESTING"
  }' \
  --max-time 30

echo ""
echo ""
echo "✅ Alert sent! Check the mobile app dashboard for notifications."
