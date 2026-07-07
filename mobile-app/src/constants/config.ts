// API Configuration
// IMPORTANT: Replace 'YOUR_COMPUTER_IP' with your actual local IP address
// To find your IP: Run 'ipconfig' in PowerShell and look for IPv4 Address
// Example: '192.168.1.100'

export const API_CONFIG = {
  // Change this to your computer's IP address when testing on a physical device
  // Use 'localhost' only when testing on web or emulator on the same machine
  BASE_URL: 'http://192.168.1.153:8000',
  API_KEY: 'KWkKo1HmrQ3UWm9SvhOk3g8OgT4qcEPX',
  ENDPOINTS: {
    PREDICT: '/predict',
    BATCH_PREDICT: '/batch-predict',
  },
};

export const APP_CONFIG = {
  MAX_IMAGE_SIZE: 10 * 1024 * 1024, // 10MB
  SUPPORTED_FORMATS: ['image/jpeg', 'image/jpg', 'image/png'],
  IMAGE_QUALITY: 0.8,
};
