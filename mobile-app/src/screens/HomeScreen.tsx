import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Image,
  Alert,
  Platform,
} from 'react-native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import * as ImagePicker from 'expo-image-picker';
import { RootStackParamList } from '../types';
import { predictBreed, checkServerHealth } from '../services/api';

type HomeScreenNavigationProp = NativeStackNavigationProp<RootStackParamList, 'Home'>;

interface HomeScreenProps {
  navigation: HomeScreenNavigationProp;
}

export const HomeScreen: React.FC<HomeScreenProps> = ({ navigation }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [serverOnline, setServerOnline] = useState<boolean | null>(null);

  useEffect(() => {
    // Check permissions on mount
    checkPermissions();
    // Check server health
    checkServer();
  }, []);

  const checkPermissions = async () => {
    if (Platform.OS !== 'web') {
      const cameraStatus = await ImagePicker.requestCameraPermissionsAsync();
      const mediaStatus = await ImagePicker.requestMediaLibraryPermissionsAsync();
      
      if (cameraStatus.status !== 'granted' || mediaStatus.status !== 'granted') {
        Alert.alert(
          'Permissions Required',
          'PetSnap needs camera and photo library access to identify dog breeds.',
          [{ text: 'OK' }]
        );
      }
    }
  };

  const checkServer = async () => {
    const isOnline = await checkServerHealth();
    setServerOnline(isOnline);
  };

  const handleImageSelected = async (imageUri: string) => {
    setIsLoading(true);
    try {
      const response = await predictBreed(imageUri);
      
      if (response.success && response.predictions.length > 0) {
        navigation.navigate('Result', {
          predictions: response.predictions,
          breedInfo: response.breed_info,
          imageUri,
        });
      } else {
        Alert.alert('Error', response.message || 'Could not identify the breed');
      }
    } catch (error) {
      Alert.alert(
        'Error',
        error instanceof Error ? error.message : 'Failed to analyze image'
      );
    } finally {
      setIsLoading(false);
    }
  };

  const takePhoto = async () => {
    try {
      const result = await ImagePicker.launchCameraAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [4, 3],
        quality: 0.8,
      });

      if (!result.canceled && result.assets[0]) {
        await handleImageSelected(result.assets[0].uri);
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to take photo');
      console.error(error);
    }
  };

  const pickImage = async () => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [4, 3],
        quality: 0.8,
      });

      if (!result.canceled && result.assets[0]) {
        await handleImageSelected(result.assets[0].uri);
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to pick image');
      console.error(error);
    }
  };

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.logoEmoji}>🐕</Text>
        <Text style={styles.title}>PetSnap</Text>
        <Text style={styles.subtitle}>AI-Powered Dog Breed Recognition</Text>
        
        {/* Server status indicator */}
        <View style={styles.statusContainer}>
          <View
            style={[
              styles.statusDot,
              { backgroundColor: serverOnline === true ? '#10b981' : serverOnline === false ? '#ef4444' : '#6b7280' },
            ]}
          />
          <Text style={styles.statusText}>
            {serverOnline === true
              ? 'Server Online'
              : serverOnline === false
              ? 'Server Offline'
              : 'Checking...'}
          </Text>
        </View>
      </View>

      {/* Action Buttons */}
      <View style={styles.actionContainer}>
        <TouchableOpacity
          style={[styles.button, styles.primaryButton, isLoading && styles.disabledButton]}
          onPress={takePhoto}
          disabled={isLoading}
        >
          <Text style={styles.buttonIcon}>📷</Text>
          <Text style={styles.buttonText}>Take Photo</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.button, styles.secondaryButton, isLoading && styles.disabledButton]}
          onPress={pickImage}
          disabled={isLoading}
        >
          <Text style={styles.buttonIcon}>🖼️</Text>
          <Text style={styles.buttonText}>Choose from Gallery</Text>
        </TouchableOpacity>
      </View>

      {/* Info Section */}
      <View style={styles.infoContainer}>
        <Text style={styles.infoTitle}>How it works:</Text>
        <View style={styles.infoStep}>
          <Text style={styles.infoStepNumber}>1</Text>
          <Text style={styles.infoStepText}>Take or select a photo of a dog</Text>
        </View>
        <View style={styles.infoStep}>
          <Text style={styles.infoStepNumber}>2</Text>
          <Text style={styles.infoStepText}>Our AI analyzes the image</Text>
        </View>
        <View style={styles.infoStep}>
          <Text style={styles.infoStepNumber}>3</Text>
          <Text style={styles.infoStepText}>Get breed predictions and information</Text>
        </View>
      </View>

      {isLoading && (
        <View style={styles.loadingOverlay}>
          <View style={styles.loadingBox}>
            <Text style={styles.loadingText}>Analyzing image... 🐕</Text>
          </View>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f9fafb',
  },
  header: {
    alignItems: 'center',
    paddingTop: 60,
    paddingBottom: 40,
    backgroundColor: '#fff',
    borderBottomLeftRadius: 30,
    borderBottomRightRadius: 30,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 5,
  },
  logoEmoji: {
    fontSize: 80,
    marginBottom: 16,
  },
  title: {
    fontSize: 36,
    fontWeight: 'bold',
    color: '#111827',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    color: '#6b7280',
    textAlign: 'center',
  },
  statusContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 16,
    paddingHorizontal: 16,
    paddingVertical: 8,
    backgroundColor: '#f3f4f6',
    borderRadius: 20,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 8,
  },
  statusText: {
    fontSize: 12,
    color: '#4b5563',
  },
  actionContainer: {
    padding: 24,
  },
  button: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
    borderRadius: 16,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  primaryButton: {
    backgroundColor: '#3b82f6',
  },
  secondaryButton: {
    backgroundColor: '#10b981',
  },
  disabledButton: {
    opacity: 0.5,
  },
  buttonIcon: {
    fontSize: 24,
    marginRight: 12,
  },
  buttonText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#fff',
  },
  infoContainer: {
    padding: 24,
    backgroundColor: '#fff',
    margin: 16,
    borderRadius: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  infoTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#111827',
    marginBottom: 16,
  },
  infoStep: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  infoStepNumber: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: '#dbeafe',
    color: '#1e40af',
    fontSize: 14,
    fontWeight: 'bold',
    textAlign: 'center',
    lineHeight: 28,
    marginRight: 12,
  },
  infoStepText: {
    fontSize: 14,
    color: '#4b5563',
    flex: 1,
  },
  loadingOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingBox: {
    backgroundColor: '#fff',
    padding: 24,
    borderRadius: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 8,
    elevation: 5,
  },
  loadingText: {
    fontSize: 16,
    color: '#111827',
    fontWeight: '600',
  },
});
