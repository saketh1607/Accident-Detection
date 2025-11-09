/*
  # Temporarily disable RLS for amenities import

  1. Changes
    - Disable RLS on amenities table to allow OSM import
    - Can be re-enabled after import is complete
*/

ALTER TABLE amenities DISABLE ROW LEVEL SECURITY;
