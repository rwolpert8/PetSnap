# PetSnap Mobile App

Cross-platform mobile application for dog breed identification using AI. Built with React Native and Expo.

## Features

- **Camera Integration**: Take photos directly from the app
- **Gallery Selection**: Choose existing photos from your device
- **AI-Powered Recognition**: Identify dog breeds with high accuracy
- **Top 5 Predictions**: See confidence scores for multiple breed predictions
- **Breed Information**: Get detailed information from AKC (American Kennel Club)
- **Cross-Platform**: Works on both iOS and Android

## Prerequisites

- **Node.js** (v16 or higher)
- **Expo Go** app on your phone ([iOS](https://apps.apple.com/app/expo-go/id982107779) | [Android](https://play.google.com/store/apps/details?id=host.exp.exponent))
- **PetSnap API server** running

## Setup

### 1. Install Dependencies

```bash
cd mobile-app
npm install
```

### 2. Find Your Computer's IP Address

**Windows:**
```powershell
ipconfig
```
Look for "IPv4 Address" (e.g., `192.168.1.100`)

**Mac/Linux:**
```bash
ifconfig | grep "inet "
```

### 3. Configure API Endpoint

Edit `src/constants/config.ts`:

```typescript
export const API_CONFIG = {
  BASE_URL: 'http://192.168.1.100:8000',  // ← Use your IP here
  API_KEY: 'KWkKo1HmrQ3UWm9SvhOk3g8OgT4qcEPX',
  // ...
};
```

### 4. Start the API Server

```bash
python api/api_server.py
```

Wait for: `Uvicorn running on http://0.0.0.0:8000`

### 5. Start the Mobile App

```bash
cd mobile-app
npm start
```

### 6. Scan QR Code

1. Open **Expo Go** on your phone
2. Scan the QR code from the terminal
3. Wait for app to load

**Important**: Your phone and computer must be on the same WiFi network

## Project Structure

```
mobile-app/
├── App.tsx                    # Main app entry point with navigation
├── app.json                   # Expo configuration
├── package.json              # Dependencies
├── src/
│   ├── screens/
│   │   ├── HomeScreen.tsx    # Landing page with action buttons
│   │   ├── CameraScreen.tsx  # Camera interface (optional)
│   │   └── ResultScreen.tsx  # Display predictions
│   ├── components/
│   │   ├── BreedCard.tsx     # Breed prediction card component
│   │   └── LoadingSpinner.tsx # Loading indicator
│   ├── services/
│   │   └── api.ts            # API client for FastAPI server
│   ├── types/
│   │   └── index.ts          # TypeScript type definitions
│   └── constants/
│       └── config.ts         # App configuration (API URL, etc.)
└── assets/                   # Images, fonts, icons
```

## Usage

### Taking a Photo

1. Tap **"Take Photo"** on the home screen
2. Point camera at a dog
3. Tap the capture button
4. Wait for AI analysis (~2-5 seconds)
5. View results with breed predictions and information

### Selecting from Gallery

1. Tap **"Choose from Gallery"** on the home screen
2. Select a photo of a dog
3. Wait for AI analysis
4. View results

### Understanding Results

- **Most Likely Breed**: Top prediction with detailed information
- **Confidence Score**: Percentage showing AI's confidence (higher is better)
- **Other Possibilities**: Alternative breed predictions
- **Breed Information**: Details from AKC including:
  - Group (e.g., Sporting, Working)
  - Size and temperament
  - Life span
  - Physical characteristics
  - Description

## Troubleshooting

### Cannot Connect to Server
1. Verify FastAPI server is running: `python api/api_server.py`
2. Check both devices are on the same WiFi network
3. Confirm IP address in `src/constants/config.ts` is correct
4. Test in phone browser: `http://YOUR_IP:8000`
5. Check firewall isn't blocking port 8000

### Camera Permission Denied
1. Phone Settings → Expo Go → Enable Camera
2. Restart Expo Go app
3. Use "Choose from Gallery" as alternative

### App Not Loading
1. Clear cache: `npx expo start -c`
2. Check terminal for errors
3. Restart Expo Go app

## Tech Stack

- **React Native**: Mobile framework
- **Expo SDK 51**: Development platform
- **TypeScript**: Type safety
- **React Navigation**: Screen navigation
- **Axios**: HTTP client
- **Expo Camera**: Camera access
- **Expo Image Picker**: Gallery access

## API

**Endpoint**: `POST /predict`  
**Headers**: `Authorization: Bearer {API_KEY}`  
**Body**: FormData with image file  
**Response**: Breed predictions with confidence scores

See `src/services/api.ts` for implementation.

## License

MIT

MIT License - See main PetSnap LICENSE.md

## Support

For issues or questions:
- Check [Expo Documentation](https://docs.expo.dev/)
- Check [React Native Documentation](https://reactnative.dev/)
- Review API server logs for backend issues
Tech Stack

- **React Native**: Mobile framework
- **Expo SDK 51**: Development platform
- **TypeScript**: Type safety
- **React Navigation**: Screen navigation
- **Axios**: HTTP client
- **Expo Camera**: Camera access
- **Expo Image Picker**: Gallery access

## API

**Endpoint**: `POST /predict`  
**Headers**: `Authorization: Bearer {API_KEY}`  
**Body**: FormData with image file  
**Response**: Breed predictions with confidence scores

See `src/services/api.ts` for implementation.