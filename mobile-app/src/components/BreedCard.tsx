import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { BreedPrediction, BreedInfo } from '../types';

interface BreedCardProps {
  prediction: BreedPrediction;
  breedInfo?: BreedInfo;
  isTopPrediction?: boolean;
}

export const BreedCard: React.FC<BreedCardProps> = ({
  prediction,
  breedInfo,
  isTopPrediction = false,
}) => {
  const confidence = (prediction.confidence * 100).toFixed(1);

  return (
    <View style={[styles.container, isTopPrediction && styles.topPrediction]}>
      <View style={styles.header}>
        <View style={styles.rankBadge}>
          <Text style={styles.rankText}>#{prediction.rank}</Text>
        </View>
        <View style={styles.breedInfo}>
          <Text style={[styles.breedName, isTopPrediction && styles.topBreedName]}>
            {prediction.breed}
          </Text>
          <Text style={styles.confidence}>{confidence}% confident</Text>
        </View>
      </View>

      {isTopPrediction && breedInfo && (
        <View style={styles.details}>
          {breedInfo.group && (
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Group:</Text>
              <Text style={styles.detailValue}>{breedInfo.group}</Text>
            </View>
          )}
          {breedInfo.size && (
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Size:</Text>
              <Text style={styles.detailValue}>{breedInfo.size}</Text>
            </View>
          )}
          {breedInfo.temperament && (
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Temperament:</Text>
              <Text style={styles.detailValue}>{breedInfo.temperament}</Text>
            </View>
          )}
          {breedInfo.life_span && (
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Life Span:</Text>
              <Text style={styles.detailValue}>{breedInfo.life_span}</Text>
            </View>
          )}
          {breedInfo.description && (
            <View style={styles.description}>
              <Text style={styles.descriptionText}>{breedInfo.description}</Text>
            </View>
          )}
        </View>
      )}

      {/* Progress bar */}
      <View style={styles.progressContainer}>
        <View
          style={[
            styles.progressBar,
            { width: `${prediction.confidence * 100}%` },
            isTopPrediction && styles.topProgressBar,
          ]}
        />
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginVertical: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  topPrediction: {
    backgroundColor: '#f0f9ff',
    borderWidth: 2,
    borderColor: '#3b82f6',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  rankBadge: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#e5e7eb',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  rankText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#374151',
  },
  breedInfo: {
    flex: 1,
  },
  breedName: {
    fontSize: 18,
    fontWeight: '600',
    color: '#111827',
    marginBottom: 4,
  },
  topBreedName: {
    fontSize: 20,
    color: '#1e40af',
  },
  confidence: {
    fontSize: 14,
    color: '#6b7280',
  },
  details: {
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#e5e7eb',
  },
  detailRow: {
    flexDirection: 'row',
    marginBottom: 8,
  },
  detailLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#374151',
    width: 100,
  },
  detailValue: {
    fontSize: 14,
    color: '#6b7280',
    flex: 1,
  },
  description: {
    marginTop: 8,
    paddingTop: 8,
    borderTopWidth: 1,
    borderTopColor: '#e5e7eb',
  },
  descriptionText: {
    fontSize: 14,
    color: '#4b5563',
    lineHeight: 20,
  },
  progressContainer: {
    height: 4,
    backgroundColor: '#e5e7eb',
    borderRadius: 2,
    marginTop: 12,
    overflow: 'hidden',
  },
  progressBar: {
    height: '100%',
    backgroundColor: '#10b981',
    borderRadius: 2,
  },
  topProgressBar: {
    backgroundColor: '#3b82f6',
  },
});
