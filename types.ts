export interface AnalysisResult {
  diagnosis: 'CNV' | 'DME' | 'Drusen' | 'Normal' | 'AMD' | 'Geographic Atrophy' | 'Requires Further Review';
  confidence: string;
  explanation: string;
  explainability: string;
  uncertaintyStatement: string;
  segmentationUncertaintyStatement: string;
  anomalyReport?: string;
}

export type ImageStatus = 'pending' | 'loading' | 'success' | 'error';

export interface AnalyzableImage {
  id: string;
  file: File;
  previewUrl: string;
  status: ImageStatus;
  result?: AnalysisResult;
  error?: string;
  segmentedImageUrl?: string;
  heatmapImageUrl?: string;
  segmentationUncertaintyMapUrl?: string;
}
