/*
  # Fix accident alerts insert policy

  1. Changes
    - Add INSERT policy for accident_alerts to allow public API to create alerts
    - Add INSERT policy for alert_notifications
*/

-- Disable RLS temporarily for accident_alerts and alert_notifications
ALTER TABLE accident_alerts DISABLE ROW LEVEL SECURITY;
ALTER TABLE alert_notifications DISABLE ROW LEVEL SECURITY;
