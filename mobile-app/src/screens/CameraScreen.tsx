import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Alert,
  Platform,
} from 'react-native';
import { Camera, CameraType, CameraView } from 'expo-camera';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../types';
import { predictBreed } from '../services/api';

type CameraScreenNavigationProp = NativeStackNavigationProp<
  RootStackParamList,
  'Camera'
>;

interface CameraScreenProps {
  navigation: CameraScreenNavigationProp;
}

export const CameraScreen: React.FC<CameraScreenProps> = ({ navigation }) => {
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [type, setType] = useState<CameraType>('back');
  const [isCapturing, setIsCapturing] = useState(false);
  const cameraRef = useRef<CameraView>(null);

  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === 'granted');
    })();
  }, []);

  const takePicture = async () => {
    if (!cameraRef.current || isCapturing) return;

    setIsCapturing(true);
    try {
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.8,
      });

      if (photo?.uri) {
        // Analyze the photo
        const response = await predictBreed(photo.uri);

        if (response.success && response.predictions.length > 0) {
          navigation.replace('Result', {
            predictions: response.predictions,
            breedInfo: response.breed_info,
            imageUri: photo.uri,
          });
        } else {
          Alert.alert('Error', response.message || 'Could not identify the breed');
          setIsCapturing(false);
        }
      }
    } catch (error) {
      Alert.alert(
        'Error',
        error instanceof Error ? error.message : 'Failed to analyze photo'
      );
      setIsCapturing(false);
    }
  };

  const toggleCameraType = () => {
    setType((current) => (current === 'back' ? 'front' : 'back'));
  };

  if (hasPermission === null) {
    return (
      <View style={styles.container}>
        <Text>Requesting camera permission...</Text>
      </View>
    );
  }

  if (hasPermission === false) {
    return (
      <View style={styles.container}>
        <Text style={styles.permissionText}>
          Camera permission is required to take photos
        </Text>
        <TouchableOpacity
          style={styles.button}
          onPress={() => navigation.goBack()}
        >
          <Text style={styles.buttonText}>Go Back</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <CameraView style={styles.camera} facing={type} ref={cameraRef}>
        <View style={styles.overlay}>
          {/* Top Bar */}
          <View style={styles.topBar}>
            <TouchableOpacity
              style={styles.iconButton}
              onPress={() => navigation.goBack()}
            >
              <Text style={styles.iconText}>✕</Text>
            </TouchableOpacity>
          </View>

          {/* Center Guide */}
          <View style={styles.centerGuide}>
            <View style={styles.guideFrame} />
            <Text style={styles.guideText}>
              Position the dog within the frame
            </Text>
          </View>

          {/* Bottom Controls */}
          <View style={styles.bottomBar}>
            <View style={styles.controls}>
              <TouchableOpacity
                style={styles.iconButton}
                onPress={toggleCameraType}
              >
                <Text style={styles.iconText}>🔄</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[styles.captureButton, isCapturing && styles.capturing]}
                onPress={takePicture}
                disabled={isCapturing}
              >
                <View style={styles.captureButtonInner} />
              </TouchableOpacity>

              <View style={styles.iconButton} />
            </View>
            
            {isCapturing && (
              <Text style={styles.capturingText}>Analyzing... 🐕</Text>
            )}
          </View>
        </View>
      </CameraView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
    justifyContent: 'center',
    alignItems: 'center',
  },
  camera: {
    flex: 1,
    width: '100%',
  },
  overlay: {
    flex: 1,
    backgroundColor: 'transparent',
  },
  topBar: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    padding: 20,
    paddingTop: Platform.OS === 'ios' ? 60 : 20,
  },
  centerGuide: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  guideFrame: {
    width: 280,
    height: 280,
    borderWidth: 3,
    borderColor: '#fff',
    borderRadius: 20,
    backgroundColor: 'transparent',
  },
  guideText: {
    marginTop: 20,
    fontSize: 16,
    color: '#fff',
    textAlign: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 20,
  },
  bottomBar: {
    paddingBottom: Platform.OS === 'ios' ? 40 : 20,
  },
  controls: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    paddingHorizontal: 40,
  },
  iconButton: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  iconText: {
    fontSize: 24,
    color: '#fff',
  },
  captureButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: '#fff',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 4,
    borderColor: '#3b82f6',
  },
  capturing: {
    opacity: 0.5,
  },
  captureButtonInner: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: '#3b82f6',
  },
  capturingText: {
    marginTop: 16,
    fontSize: 16,
    color: '#fff',
    textAlign: 'center',
    fontWeight: '600',
  },
  permissionText: {
    fontSize: 16,
    color: '#fff',
    textAlign: 'center',
    marginBottom: 20,
    paddingHorizontal: 40,
  },
  button: {
    backgroundColor: '#3b82f6',
    paddingHorizontal: 32,
    paddingVertical: 16,
    borderRadius: 12,
  },
  buttonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
  },
});
