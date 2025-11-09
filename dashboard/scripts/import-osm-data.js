import fetch from 'node-fetch';
import { createClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';

dotenv.config();

const supabase = createClient(
  process.env.VITE_SUPABASE_URL,
  process.env.VITE_SUPABASE_SUPABASE_ANON_KEY
);

const BOUNDS = {
  south: 17.450000,
  west: 78.300000,
  north: 17.720000,
  east: 78.560000
};

const OVERPASS_URL = 'https://overpass-api.de/api/interpreter';

function generateEmail(name, type, index) {
  const cleanName = name
    .toLowerCase()
    .replace(/[^a-z0-9]/g, '')
    .substring(0, 20);

  return `${cleanName}.${type}${index}@guardianconnect.emergency`;
}

async function fetchOSMData(query) {
  try {
    const response = await fetch(OVERPASS_URL, {
      method: 'POST',
      body: query,
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error fetching OSM data:', error);
    throw error;
  }
}

async function importHospitals() {
  console.log('🏥 Fetching hospitals from OpenStreetMap...');

  const query = `
    [out:json][timeout:60];
    (
      node["amenity"="hospital"](${BOUNDS.south},${BOUNDS.west},${BOUNDS.north},${BOUNDS.east});
      way["amenity"="hospital"](${BOUNDS.south},${BOUNDS.west},${BOUNDS.north},${BOUNDS.east});
      node["healthcare"](${BOUNDS.south},${BOUNDS.west},${BOUNDS.north},${BOUNDS.east});
      way["healthcare"](${BOUNDS.south},${BOUNDS.west},${BOUNDS.north},${BOUNDS.east});
    );
    out center;
  `;

  const data = await fetchOSMData(query);
  const hospitals = [];

  data.elements.forEach((element, index) => {
    const name = element.tags?.name || `Hospital ${index + 1}`;
    const lat = element.lat || element.center?.lat;
    const lon = element.lon || element.center?.lon;

    if (lat && lon) {
      hospitals.push({
        name,
        email: generateEmail(name, 'hospital', index + 1),
        type: 'hospital',
        latitude: lat,
        longitude: lon,
        osm_id: `osm_${element.type}_${element.id}`,
        address: element.tags?.['addr:full'] || element.tags?.['addr:street'] || null,
        contact_number: element.tags?.phone || null
      });
    }
  });

  console.log(`✅ Found ${hospitals.length} hospitals`);
  return hospitals;
}

async function importPoliceStations() {
  console.log('🚔 Fetching police stations from OpenStreetMap...');

  const query = `
    [out:json][timeout:60];
    (
      node["amenity"="police"](${BOUNDS.south},${BOUNDS.west},${BOUNDS.north},${BOUNDS.east});
      way["amenity"="police"](${BOUNDS.south},${BOUNDS.west},${BOUNDS.north},${BOUNDS.east});
    );
    out center;
  `;

  const data = await fetchOSMData(query);
  const policeStations = [];

  data.elements.forEach((element, index) => {
    const name = element.tags?.name || `Police Station ${index + 1}`;
    const lat = element.lat || element.center?.lat;
    const lon = element.lon || element.center?.lon;

    if (lat && lon) {
      policeStations.push({
        name,
        email: generateEmail(name, 'police', index + 1),
        type: 'police',
        latitude: lat,
        longitude: lon,
        osm_id: `osm_${element.type}_${element.id}`,
        address: element.tags?.['addr:full'] || element.tags?.['addr:street'] || null,
        contact_number: element.tags?.phone || null
      });
    }
  });

  console.log(`✅ Found ${policeStations.length} police stations`);
  return policeStations;
}

async function importFireStations() {
  console.log('🚒 Fetching fire stations from OpenStreetMap...');

  const query = `
    [out:json][timeout:60];
    (
      node["amenity"="fire_station"](${BOUNDS.south},${BOUNDS.west},${BOUNDS.north},${BOUNDS.east});
      way["amenity"="fire_station"](${BOUNDS.south},${BOUNDS.west},${BOUNDS.north},${BOUNDS.east});
    );
    out center;
  `;

  const data = await fetchOSMData(query);
  const fireStations = [];

  data.elements.forEach((element, index) => {
    const name = element.tags?.name || `Fire Station ${index + 1}`;
    const lat = element.lat || element.center?.lat;
    const lon = element.lon || element.center?.lon;

    if (lat && lon) {
      fireStations.push({
        name,
        email: generateEmail(name, 'firestation', index + 1),
        type: 'fire_station',
        latitude: lat,
        longitude: lon,
        osm_id: `osm_${element.type}_${element.id}`,
        address: element.tags?.['addr:full'] || element.tags?.['addr:street'] || null,
        contact_number: element.tags?.phone || null
      });
    }
  });

  console.log(`✅ Found ${fireStations.length} fire stations`);
  return fireStations;
}

async function saveToDatabase(amenities) {
  console.log(`\n💾 Saving ${amenities.length} amenities to database...`);

  const { data: existing } = await supabase
    .from('amenities')
    .select('osm_id');

  const existingIds = new Set(existing?.map(a => a.osm_id) || []);
  const newAmenities = amenities.filter(a => !existingIds.has(a.osm_id));

  if (newAmenities.length === 0) {
    console.log('ℹ️  No new amenities to import. All existing amenities are up to date.');
    return;
  }

  const { data, error } = await supabase
    .from('amenities')
    .insert(newAmenities)
    .select();

  if (error) {
    console.error('❌ Error saving to database:', error);
    throw error;
  }

  console.log(`✅ Successfully saved ${data.length} new amenities`);
}

async function exportCredentials() {
  console.log('\n📋 Fetching all registered amenities...');

  const { data: amenities, error } = await supabase
    .from('amenities')
    .select('name, email, type')
    .order('type', { ascending: true })
    .order('name', { ascending: true });

  if (error) {
    console.error('❌ Error fetching amenities:', error);
    throw error;
  }

  console.log('\n' + '='.repeat(80));
  console.log('🔐 GUARDIAN CONNECT - LOGIN CREDENTIALS');
  console.log('='.repeat(80));
  console.log('\n📧 EMAIL-ONLY LOGIN (No password required)\n');

  const grouped = {
    hospital: [],
    police: [],
    fire_station: []
  };

  amenities.forEach(a => grouped[a.type].push(a));

  console.log('🏥 HOSPITALS:\n');
  grouped.hospital.forEach((a, i) => {
    console.log(`  ${i + 1}. ${a.name}`);
    console.log(`     Email: ${a.email}\n`);
  });

  console.log('\n🚔 POLICE STATIONS:\n');
  grouped.police.forEach((a, i) => {
    console.log(`  ${i + 1}. ${a.name}`);
    console.log(`     Email: ${a.email}\n`);
  });

  console.log('\n🚒 FIRE STATIONS:\n');
  grouped.fire_station.forEach((a, i) => {
    console.log(`  ${i + 1}. ${a.name}`);
    console.log(`     Email: ${a.email}\n`);
  });

  console.log('='.repeat(80));
  console.log(`Total Amenities: ${amenities.length}`);
  console.log('='.repeat(80) + '\n');
}

async function main() {
  try {
    console.log('🌍 Starting Guardian Connect OSM Data Import...\n');
    console.log(`📍 Region: (${BOUNDS.south}, ${BOUNDS.west}) to (${BOUNDS.north}, ${BOUNDS.east})\n`);

    const [hospitals, policeStations, fireStations] = await Promise.all([
      importHospitals(),
      importPoliceStations(),
      importFireStations()
    ]);

    const allAmenities = [...hospitals, ...policeStations, ...fireStations];

    console.log(`\n📊 Total amenities found: ${allAmenities.length}`);

    await saveToDatabase(allAmenities);
    await exportCredentials();

    console.log('✅ Import completed successfully!\n');

  } catch (error) {
    console.error('❌ Fatal error:', error);
    process.exit(1);
  }
}

main();
