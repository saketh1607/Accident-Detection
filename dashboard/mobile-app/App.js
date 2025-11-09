import React from 'react';
import { Text } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { StatusBar } from 'expo-status-bar';

import LoginScreen from './src/screens/LoginScreen';
import DashboardScreen from './src/screens/DashboardScreen';
import MapScreen from './src/screens/MapScreen';

const Stack = createNativeStackNavigator();
const Tab = createBottomTabNavigator();

function TabIcon({ icon, color }) {
  return (
    <Text style={{ fontSize: 24, opacity: color === '#EF4444' ? 1 : 0.5 }}>
      {icon}
    </Text>
  );
}

function MainTabs({ route }) {
  const { amenity } = route.params;

  return (
    <Tab.Navigator
      screenOptions={{
        headerShown: false,
        tabBarStyle: {
          backgroundColor: '#1E293B',
          borderTopColor: '#334155',
          borderTopWidth: 1,
        },
        tabBarActiveTintColor: '#EF4444',
        tabBarInactiveTintColor: '#64748B',
      }}
    >
      <Tab.Screen
        name="Dashboard"
        component={DashboardScreen}
        initialParams={{ amenity }}
        options={{
          tabBarLabel: 'Alerts',
          tabBarIcon: ({ color }) => <TabIcon icon="🚨" color={color} />,
        }}
      />
      <Tab.Screen
        name="Map"
        component={MapScreen}
        initialParams={{ amenity }}
        options={{
          tabBarLabel: 'Map',
          tabBarIcon: ({ color }) => <TabIcon icon="🗺️" color={color} />,
        }}
      />
    </Tab.Navigator>
  );
}

export default function App() {
  return (
    <NavigationContainer>
      <StatusBar style="light" />
      <Stack.Navigator screenOptions={{ headerShown: false }}>
        <Stack.Screen name="Login" component={LoginScreen} />
        <Stack.Screen name="Main" component={MainTabs} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
