/*
  # Fix amenities insert policy

  1. Changes
    - Add INSERT policy for amenities table to allow service role to insert new amenities
    - This allows the OSM import script to populate the database
*/

-- Allow service role to insert amenities (for OSM import)
CREATE POLICY "Service role can insert amenities"
  ON amenities FOR INSERT
  TO service_role
  WITH CHECK (true);

-- Allow anon role to insert amenities (for import scripts)
CREATE POLICY "Anon can insert amenities"
  ON amenities FOR INSERT
  TO anon
  WITH CHECK (true);
