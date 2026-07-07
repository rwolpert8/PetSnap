import axios, { AxiosError } from 'axios';
import { API_CONFIG } from '../constants/config';
import { PredictionResponse, ApiError } from '../types';

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_CONFIG.BASE_URL,
  timeout: 30000, // 30 seconds
  headers: {
    'Content-Type': 'multipart/form-data',
  },
});

// Add authentication interceptor
apiClient.interceptors.request.use(
  (config) => {
    config.headers.Authorization = `Bearer ${API_CONFIG.API_KEY}`;
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

/**
 * Upload an image and get breed predictions
 * @param imageUri - Local URI of the image to classify
 * @returns Prediction response with breed information
 */
export const predictBreed = async (imageUri: string): Promise<PredictionResponse> => {
  try {
    // Create form data
    const formData = new FormData();
    
    // Extract filename from URI
    const filename = imageUri.split('/').pop() || 'photo.jpg';
    
    // For React Native, we need to provide the file in this format
    const file: any = {
      uri: imageUri,
      type: 'image/jpeg',
      name: filename,
    };
    
    formData.append('file', file);

    // Make API request
    const response = await apiClient.post<PredictionResponse>(
      API_CONFIG.ENDPOINTS.PREDICT,
      formData
    );

    return response.data;
  } catch (error) {
    console.error('API Error:', error);
    
    if (axios.isAxiosError(error)) {
      const axiosError = error as AxiosError<ApiError>;
      
      if (axiosError.response) {
        // Server responded with error
        throw new Error(
          axiosError.response.data?.detail || 
          `Server error: ${axiosError.response.status}`
        );
      } else if (axiosError.request) {
        // Request made but no response
        throw new Error(
          'Cannot connect to server. Make sure:\n' +
          '1. FastAPI server is running\n' +
          '2. You\'re using the correct IP address in config.ts\n' +
          '3. Both devices are on the same WiFi network'
        );
      }
    }
    
    throw new Error('An unexpected error occurred');
  }
};

/**
 * Batch predict multiple images
 * @param imageUris - Array of local image URIs
 * @returns Array of prediction responses
 */
export const batchPredictBreeds = async (
  imageUris: string[]
): Promise<PredictionResponse[]> => {
  try {
    const formData = new FormData();
    
    imageUris.forEach((uri, index) => {
      const filename = uri.split('/').pop() || `photo${index}.jpg`;
      const file: any = {
        uri,
        type: 'image/jpeg',
        name: filename,
      };
      formData.append('files', file);
    });

    const response = await apiClient.post<PredictionResponse[]>(
      API_CONFIG.ENDPOINTS.BATCH_PREDICT,
      formData
    );

    return response.data;
  } catch (error) {
    console.error('Batch API Error:', error);
    throw error;
  }
};

/**
 * Check if the API server is reachable
 * @returns true if server is reachable, false otherwise
 */
export const checkServerHealth = async (): Promise<boolean> => {
  try {
    const response = await axios.get(`${API_CONFIG.BASE_URL}/`, {
      timeout: 5000,
    });
    return response.status === 200;
  } catch (error) {
    return false;
  }
};
