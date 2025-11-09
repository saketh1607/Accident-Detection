/*
  # Temporarily allow public inserts for amenities

  1. Changes
    - Drop existing insert policies
    - Add public insert policy for initial OSM data import
    - This can be restricted after initial setup
*/

-- Drop existing policies
DROP POLICY IF EXISTS "Service role can insert amenities" ON amenities;
DROP POLICY IF EXISTS "Anon can insert amenities" ON amenities;

-- Allow public inserts (for OSM import script)
CREATE POLICY "Public can insert amenities"
  ON amenities FOR INSERT
  WITH CHECK (true);
