/*
  # Guardian Connect Database Schema

  1. New Tables
    - `amenities`
      - `id` (uuid, primary key)
      - `name` (text) - Name of the amenity
      - `email` (text, unique) - Auto-generated email for login
      - `type` (text) - hospital/police/fire_station
      - `latitude` (numeric) - Location latitude
      - `longitude` (numeric) - Location longitude
      - `contact_number` (text, nullable) - Contact number
      - `osm_id` (text, unique, nullable) - OpenStreetMap ID
      - `address` (text, nullable) - Address from OSM
      - `created_at` (timestamptz) - Registration timestamp
      - `last_login` (timestamptz, nullable) - Last login time

    - `accident_alerts`
      - `id` (uuid, primary key)
      - `latitude` (numeric) - Accident location latitude
      - `longitude` (numeric) - Accident location longitude
      - `description` (text) - Alert description
      - `created_at` (timestamptz) - Alert timestamp
      - `resolved` (boolean) - Resolution status

    - `alert_notifications`
      - `id` (uuid, primary key)
      - `amenity_id` (uuid, foreign key) - Receiving amenity
      - `alert_id` (uuid, foreign key) - Related alert
      - `distance_km` (numeric) - Distance from amenity
      - `read` (boolean) - Read status
      - `created_at` (timestamptz) - Notification time

  2. Security
    - Enable RLS on all tables
    - Amenities can read their own profile and alerts
    - Public access for alert posting endpoint (handled by backend)
    - Amenities can update their own read status
*/

-- Create amenities table
CREATE TABLE IF NOT EXISTS amenities (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name text NOT NULL,
  email text UNIQUE NOT NULL,
  type text NOT NULL CHECK (type IN ('hospital', 'police', 'fire_station')),
  latitude numeric NOT NULL,
  longitude numeric NOT NULL,
  contact_number text,
  osm_id text UNIQUE,
  address text,
  created_at timestamptz DEFAULT now(),
  last_login timestamptz
);

-- Create accident_alerts table
CREATE TABLE IF NOT EXISTS accident_alerts (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  latitude numeric NOT NULL,
  longitude numeric NOT NULL,
  description text NOT NULL,
  created_at timestamptz DEFAULT now(),
  resolved boolean DEFAULT false
);

-- Create alert_notifications table
CREATE TABLE IF NOT EXISTS alert_notifications (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  amenity_id uuid NOT NULL REFERENCES amenities(id) ON DELETE CASCADE,
  alert_id uuid NOT NULL REFERENCES accident_alerts(id) ON DELETE CASCADE,
  distance_km numeric NOT NULL,
  read boolean DEFAULT false,
  created_at timestamptz DEFAULT now()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_amenities_type ON amenities(type);
CREATE INDEX IF NOT EXISTS idx_amenities_location ON amenities(latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_accident_alerts_created_at ON accident_alerts(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_alert_notifications_amenity ON alert_notifications(amenity_id, created_at DESC);

-- Enable Row Level Security
ALTER TABLE amenities ENABLE ROW LEVEL SECURITY;
ALTER TABLE accident_alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE alert_notifications ENABLE ROW LEVEL SECURITY;

-- RLS Policies for amenities
CREATE POLICY "Amenities can read own profile"
  ON amenities FOR SELECT
  TO authenticated
  USING (email = current_setting('request.jwt.claims', true)::json->>'email');

CREATE POLICY "Amenities can update own profile"
  ON amenities FOR UPDATE
  TO authenticated
  USING (email = current_setting('request.jwt.claims', true)::json->>'email')
  WITH CHECK (email = current_setting('request.jwt.claims', true)::json->>'email');

-- RLS Policies for accident_alerts (read-only for amenities)
CREATE POLICY "Amenities can read all alerts"
  ON accident_alerts FOR SELECT
  TO authenticated
  USING (true);

-- RLS Policies for alert_notifications
CREATE POLICY "Amenities can read own notifications"
  ON alert_notifications FOR SELECT
  TO authenticated
  USING (
    amenity_id IN (
      SELECT id FROM amenities 
      WHERE email = current_setting('request.jwt.claims', true)::json->>'email'
    )
  );

CREATE POLICY "Amenities can update own notification read status"
  ON alert_notifications FOR UPDATE
  TO authenticated
  USING (
    amenity_id IN (
      SELECT id FROM amenities 
      WHERE email = current_setting('request.jwt.claims', true)::json->>'email'
    )
  )
  WITH CHECK (
    amenity_id IN (
      SELECT id FROM amenities 
      WHERE email = current_setting('request.jwt.claims', true)::json->>'email'
    )
  );

-- Function to calculate distance between two coordinates (Haversine formula)
CREATE OR REPLACE FUNCTION calculate_distance(
  lat1 numeric, lon1 numeric, lat2 numeric, lon2 numeric
) RETURNS numeric AS $$
DECLARE
  R numeric := 6371; -- Earth radius in km
  dLat numeric;
  dLon numeric;
  a numeric;
  c numeric;
BEGIN
  dLat := radians(lat2 - lat1);
  dLon := radians(lon2 - lon1);
  
  a := sin(dLat/2) * sin(dLat/2) +
       cos(radians(lat1)) * cos(radians(lat2)) *
       sin(dLon/2) * sin(dLon/2);
  
  c := 2 * atan2(sqrt(a), sqrt(1-a));
  
  RETURN R * c;
END;
$$ LANGUAGE plpgsql IMMUTABLE;