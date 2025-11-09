import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ActivityIndicator,
  TouchableOpacity,
} from 'react-native';
import MapView, { Marker, Circle, PROVIDER_DEFAULT } from 'react-native-maps';
import { supabase } from '../config/supabase';

const AMENITY_COLORS = {
  hospital: '#10B981',
  police: '#3B82F6',
  fire_station: '#EF4444',
};

const AMENITY_EMOJI = {
  hospital: '🏥',
  police: '🚔',
  fire_station: '🚒',
};

export default function MapScreen({ route }) {
  const { amenity } = route.params;
  const [amenities, setAmenities] = useState([]);
  const [recentAlerts, setRecentAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showAlerts, setShowAlerts] = useState(true);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      const [amenitiesRes, alertsRes] = await Promise.all([
        supabase.from('amenities').select('*'),
        supabase
          .from('accident_alerts')
          .select('*')
          .eq('resolved', false)
          .order('created_at', { ascending: false })
          .limit(20),
      ]);

      if (amenitiesRes.data) setAmenities(amenitiesRes.data);
      if (alertsRes.data) setRecentAlerts(alertsRes.data);
    } catch (error) {
      console.error('Error fetching map data:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#EF4444" />
        <Text style={styles.loadingText}>Loading map...</Text>
      </View>
    );
  }

  const initialRegion = {
    latitude: amenity.latitude,
    longitude: amenity.longitude,
    latitudeDelta: 0.15,
    longitudeDelta: 0.15,
  };

  return (
    <View style={styles.container}>
      <MapView
        style={styles.map}
        provider={PROVIDER_DEFAULT}
        initialRegion={initialRegion}
        showsUserLocation={false}
        showsMyLocationButton={true}
      >
        {amenities.map((item) => (
          <Marker
            key={item.id}
            coordinate={{
              latitude: parseFloat(item.latitude),
              longitude: parseFloat(item.longitude),
            }}
            title={item.name}
            description={`${item.type} - ${item.email}`}
            pinColor={AMENITY_COLORS[item.type]}
          >
            <View style={[
              styles.markerContainer,
              { backgroundColor: AMENITY_COLORS[item.type] }
            ]}>
              <Text style={styles.markerEmoji}>{AMENITY_EMOJI[item.type]}</Text>
            </View>
          </Marker>
        ))}

        {showAlerts && recentAlerts.map((alert) => (
          <React.Fragment key={alert.id}>
            <Marker
              coordinate={{
                latitude: parseFloat(alert.latitude),
                longitude: parseFloat(alert.longitude),
              }}
              title="Accident Alert"
              description={alert.description}
            >
              <View style={styles.alertMarker}>
                <Text style={styles.alertMarkerText}>⚠️</Text>
              </View>
            </Marker>
            <Circle
              center={{
                latitude: parseFloat(alert.latitude),
                longitude: parseFloat(alert.longitude),
              }}
              radius={5000}
              strokeColor="rgba(239, 68, 68, 0.5)"
              fillColor="rgba(239, 68, 68, 0.1)"
            />
          </React.Fragment>
        ))}

        <Circle
          center={{
            latitude: amenity.latitude,
            longitude: amenity.longitude,
          }}
          radius={5000}
          strokeColor="rgba(59, 130, 246, 0.5)"
          fillColor="rgba(59, 130, 246, 0.05)"
        />
      </MapView>

      <View style={styles.header}>
        <Text style={styles.headerTitle}>Emergency Network Map</Text>
        <TouchableOpacity
          style={styles.toggleButton}
          onPress={() => setShowAlerts(!showAlerts)}
        >
          <Text style={styles.toggleButtonText}>
            {showAlerts ? 'Hide Alerts' : 'Show Alerts'}
          </Text>
        </TouchableOpacity>
      </View>

      <View style={styles.legend}>
        <Text style={styles.legendTitle}>Legend</Text>
        <View style={styles.legendItem}>
          <View style={[styles.legendDot, { backgroundColor: AMENITY_COLORS.hospital }]} />
          <Text style={styles.legendText}>Hospitals ({amenities.filter(a => a.type === 'hospital').length})</Text>
        </View>
        <View style={styles.legendItem}>
          <View style={[styles.legendDot, { backgroundColor: AMENITY_COLORS.police }]} />
          <Text style={styles.legendText}>Police Stations ({amenities.filter(a => a.type === 'police').length})</Text>
        </View>
        <View style={styles.legendItem}>
          <View style={[styles.legendDot, { backgroundColor: AMENITY_COLORS.fire_station }]} />
          <Text style={styles.legendText}>Fire Stations ({amenities.filter(a => a.type === 'fire_station').length})</Text>
        </View>
        {recentAlerts.length > 0 && (
          <View style={styles.legendItem}>
            <Text style={styles.alertEmoji}>⚠️</Text>
            <Text style={styles.legendText}>Active Alerts ({recentAlerts.length})</Text>
          </View>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  map: {
    flex: 1,
  },
  loadingContainer: {
    flex: 1,
    backgroundColor: '#0F172A',
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 12,
    fontSize: 16,
    color: '#94A3B8',
  },
  header: {
    position: 'absolute',
    top: 60,
    left: 16,
    right: 16,
    backgroundColor: 'rgba(30, 41, 59, 0.95)',
    borderRadius: 12,
    padding: 16,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#F1F5F9',
  },
  toggleButton: {
    backgroundColor: '#EF4444',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 6,
  },
  toggleButtonText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '600',
  },
  legend: {
    position: 'absolute',
    bottom: 20,
    left: 16,
    right: 16,
    backgroundColor: 'rgba(30, 41, 59, 0.95)',
    borderRadius: 12,
    padding: 16,
  },
  legendTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#F1F5F9',
    marginBottom: 12,
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  legendDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: 8,
  },
  legendText: {
    fontSize: 12,
    color: '#E2E8F0',
  },
  alertEmoji: {
    fontSize: 12,
    marginRight: 8,
  },
  markerContainer: {
    width: 32,
    height: 32,
    borderRadius: 16,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 2,
    borderColor: '#fff',
  },
  markerEmoji: {
    fontSize: 16,
  },
  alertMarker: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: '#EF4444',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 3,
    borderColor: '#fff',
  },
  alertMarkerText: {
    fontSize: 18,
  },
});
