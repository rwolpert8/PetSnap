// Type definitions for PetSnap mobile app

export interface BreedPrediction {
  breed: string;
  confidence: number;
  rank: number;
}

export interface BreedInfo {
  breed: string;
  group?: string;
  size?: string;
  temperament?: string;
  life_span?: string;
  weight?: string;
  height?: string;
  description?: string;
  akc_url?: string;
}

export interface PredictionResponse {
  predictions: BreedPrediction[];
  breed_info?: BreedInfo;
  success: boolean;
  message?: string;
}

export interface ApiError {
  detail: string;
  status?: number;
}

export type RootStackParamList = {
  Home: undefined;
  Camera: undefined;
  Result: {
    predictions: BreedPrediction[];
    breedInfo?: BreedInfo;
    imageUri: string;
  };
};
