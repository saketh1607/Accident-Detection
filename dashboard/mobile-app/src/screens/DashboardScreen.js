import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  RefreshControl,
  TouchableOpacity,
  Animated,
} from 'react-native';
import { supabase } from '../config/supabase';
import { formatDistance, getRelativeTime } from '../utils/distance';

export default function DashboardScreen({ route }) {
  const { amenity } = route.params;
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const flashAnim = new Animated.Value(1);

  const fetchAlerts = async (isRefresh = false) => {
    try {
      if (isRefresh) setRefreshing(true);

      const { data: notifications, error } = await supabase
        .from('alert_notifications')
        .select(`
          id,
          distance_km,
          read,
          created_at,
          alert_id,
          accident_alerts (
            id,
            latitude,
            longitude,
            description,
            created_at,
            resolved
          )
        `)
        .eq('amenity_id', amenity.id)
        .order('created_at', { ascending: false })
        .limit(50);

      if (error) throw error;

      const formattedAlerts = notifications.map(notif => ({
        id: notif.id,
        alertId: notif.alert_id,
        description: notif.accident_alerts.description,
        distance: notif.distance_km,
        timestamp: notif.accident_alerts.created_at,
        read: notif.read,
        resolved: notif.accident_alerts.resolved,
      }));

      setAlerts(formattedAlerts);

      if (formattedAlerts.some(a => !a.read)) {
        startFlashAnimation();
      }

    } catch (error) {
      console.error('Error fetching alerts:', error);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const startFlashAnimation = () => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(flashAnim, {
          toValue: 0.3,
          duration: 500,
          useNativeDriver: true,
        }),
        Animated.timing(flashAnim, {
          toValue: 1,
          duration: 500,
          useNativeDriver: true,
        }),
      ])
    ).start();
  };

  const markAsRead = async (alertId) => {
    try {
      await supabase
        .from('alert_notifications')
        .update({ read: true })
        .eq('id', alertId);

      setAlerts(prev =>
        prev.map(alert =>
          alert.id === alertId ? { ...alert, read: true } : alert
        )
      );
    } catch (error) {
      console.error('Error marking alert as read:', error);
    }
  };

  useEffect(() => {
    fetchAlerts();

    const subscription = supabase
      .channel('alert_notifications')
      .on(
        'postgres_changes',
        {
          event: 'INSERT',
          schema: 'public',
          table: 'alert_notifications',
          filter: `amenity_id=eq.${amenity.id}`,
        },
        () => {
          fetchAlerts();
        }
      )
      .subscribe();

    return () => {
      subscription.unsubscribe();
    };
  }, [amenity.id]);

  const onRefresh = useCallback(() => {
    fetchAlerts(true);
  }, []);

  const renderAlert = ({ item }) => (
    <TouchableOpacity
      style={[styles.alertCard, !item.read && styles.alertCardUnread]}
      onPress={() => markAsRead(item.id)}
    >
      {!item.read && (
        <Animated.View
          style={[styles.newBadge, { opacity: flashAnim }]}
        >
          <Text style={styles.newBadgeText}>NEW</Text>
        </Animated.View>
      )}

      <View style={styles.alertHeader}>
        <Text style={styles.alertDistance}>{formatDistance(item.distance)}</Text>
        <Text style={styles.alertTime}>{getRelativeTime(item.timestamp)}</Text>
      </View>

      <Text style={styles.alertDescription}>{item.description}</Text>

      {item.resolved && (
        <View style={styles.resolvedBadge}>
          <Text style={styles.resolvedText}>Resolved</Text>
        </View>
      )}
    </TouchableOpacity>
  );

  const unreadCount = alerts.filter(a => !a.read).length;

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <View>
          <Text style={styles.facilityName}>{amenity.name}</Text>
          <Text style={styles.facilityType}>
            {amenity.type === 'hospital' ? '🏥 Hospital' :
             amenity.type === 'police' ? '🚔 Police Station' :
             '🚒 Fire Station'}
          </Text>
        </View>
        {unreadCount > 0 && (
          <View style={styles.unreadBadge}>
            <Text style={styles.unreadCount}>{unreadCount}</Text>
          </View>
        )}
      </View>

      <View style={styles.statsContainer}>
        <View style={styles.statBox}>
          <Text style={styles.statValue}>{alerts.length}</Text>
          <Text style={styles.statLabel}>Total Alerts</Text>
        </View>
        <View style={styles.statBox}>
          <Text style={styles.statValue}>{unreadCount}</Text>
          <Text style={styles.statLabel}>Unread</Text>
        </View>
      </View>

      <FlatList
        data={alerts}
        renderItem={renderAlert}
        keyExtractor={item => item.id}
        contentContainerStyle={styles.listContent}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={onRefresh}
            tintColor="#EF4444"
          />
        }
        ListEmptyComponent={
          <View style={styles.emptyContainer}>
            <Text style={styles.emptyText}>
              {loading ? 'Loading alerts...' : 'No alerts yet'}
            </Text>
          </View>
        }
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0F172A',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    padding: 20,
    paddingTop: 60,
    backgroundColor: '#1E293B',
    borderBottomWidth: 1,
    borderBottomColor: '#334155',
  },
  facilityName: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#F1F5F9',
    marginBottom: 4,
  },
  facilityType: {
    fontSize: 14,
    color: '#94A3B8',
  },
  unreadBadge: {
    backgroundColor: '#EF4444',
    borderRadius: 12,
    minWidth: 24,
    height: 24,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 8,
  },
  unreadCount: {
    color: '#fff',
    fontSize: 12,
    fontWeight: 'bold',
  },
  statsContainer: {
    flexDirection: 'row',
    padding: 16,
    gap: 12,
  },
  statBox: {
    flex: 1,
    backgroundColor: '#1E293B',
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
  },
  statValue: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#EF4444',
    marginBottom: 4,
  },
  statLabel: {
    fontSize: 12,
    color: '#94A3B8',
  },
  listContent: {
    padding: 16,
  },
  alertCard: {
    backgroundColor: '#1E293B',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#475569',
  },
  alertCardUnread: {
    borderLeftColor: '#EF4444',
    backgroundColor: '#1E2938',
  },
  newBadge: {
    position: 'absolute',
    top: 12,
    right: 12,
    backgroundColor: '#EF4444',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  newBadgeText: {
    color: '#fff',
    fontSize: 10,
    fontWeight: 'bold',
  },
  alertHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  alertDistance: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#EF4444',
  },
  alertTime: {
    fontSize: 12,
    color: '#64748B',
  },
  alertDescription: {
    fontSize: 14,
    color: '#E2E8F0',
    lineHeight: 20,
  },
  resolvedBadge: {
    marginTop: 8,
    alignSelf: 'flex-start',
    backgroundColor: '#10B981',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  resolvedText: {
    color: '#fff',
    fontSize: 10,
    fontWeight: 'bold',
  },
  emptyContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 60,
  },
  emptyText: {
    fontSize: 16,
    color: '#64748B',
  },
});
