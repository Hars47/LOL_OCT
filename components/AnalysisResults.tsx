import React, { useState } from 'react';
import type { AnalyzableImage, AnalysisResult } from '../types';
import { SparklesIcon, DownloadIcon, InfoIcon, TrashIcon } from './icons';
import { Loader } from './Loader';

interface AnalysisCardProps {
  imageState: AnalyzableImage;
  onRefine: (id: string, feedback: string) => void;
  onDelete: (id: string) => void;
}

const getDiagnosisColor = (diagnosis: AnalysisResult['diagnosis']) => {
    switch(diagnosis) {
        case 'CNV': return 'text-red-400 border-red-400 bg-red-900/30';
        case 'AMD': return 'text-purple-400 border-purple-400 bg-purple-900/30';
        case 'DME': return 'text-orange-400 border-orange-400 bg-orange-900/30';
        case 'Drusen': return 'text-yellow-400 border-yellow-400 bg-yellow-900/30';
        case 'Geographic Atrophy': return 'text-slate-400 border-slate-400 bg-slate-700/50';
        case 'Requires Further Review': return 'text-gray-400 border-gray-400 bg-gray-700/30';
        case 'Normal':
        default: return 'text-green-400 border-green-400 bg-green-900/30';
    }
}

const ResultContent: React.FC<{ result: AnalysisResult, onSave: () => void, onRefineToggle: () => void, isRefining: boolean }> = ({ result, onSave, onRefineToggle, isRefining }) => (
    <div className="mt-4">
         <div className="flex flex-col sm:flex-row justify-between sm:items-start gap-4 mb-4">
            <h2 className="text-2xl font-bold text-slate-100">Diagnostic Report</h2>
            <div className="flex flex-col sm:flex-row gap-2 w-full sm:w-auto">
                <button 
                    onClick={onRefineToggle}
                    className="flex items-center justify-center gap-2 px-4 py-2 text-sm font-semibold bg-slate-700 hover:bg-slate-600 text-cyan-300 rounded-lg transition-colors duration-200"
                >
                    <SparklesIcon className="w-5 h-5"/>
                    {isRefining ? 'Cancel' : 'Refine Analysis'}
                </button>
                 <button 
                    onClick={onSave}
                    className="flex items-center justify-center gap-2 px-4 py-2 text-sm font-semibold bg-slate-700 hover:bg-slate-600 text-cyan-300 rounded-lg transition-colors duration-200"
                >
                    <DownloadIcon className="w-5 h-5"/>
                    Save Analysis
                </button>
            </div>
        </div>
        
        <div className="border-y border-slate-700 py-6 my-6">
            <div className="flex flex-col md:flex-row md:items-center gap-4 md:gap-6 mb-6">
                <div className={`text-lg font-bold px-4 py-2 rounded-lg border ${getDiagnosisColor(result.diagnosis)}`}>
                    Diagnosis: {result.diagnosis}
                </div>
                <div className="text-lg font-semibold text-slate-300">
                    Confidence: <span className="text-cyan-400">{result.confidence}</span>
                </div>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-slate-200 mb-2 flex items-center gap-2">
                  <InfoIcon className="w-5 h-5 text-cyan-400 flex-shrink-0"/>
                  Uncertainty Assessment
              </h3>
              <p className="text-slate-400 text-sm leading-relaxed">{result.uncertaintyStatement}</p>
            </div>
        </div>
        
        {result.anomalyReport && (
            <div className="border-b border-slate-700 pb-6 mb-6">
                <h3 className="text-lg font-semibold text-slate-200 mb-2">Ancillary Findings</h3>
                <p className="text-slate-400 text-sm leading-relaxed">{result.anomalyReport}</p>
            </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div>
                <h3 className="text-xl font-semibold text-cyan-400 mb-3">Clinical Explanation</h3>
                <p className="text-slate-300 leading-relaxed prose prose-invert prose-p:text-slate-300">{result.explanation}</p>
            </div>
             <div>
                <h3 className="text-xl font-semibold text-cyan-400 mb-3">Model Interpretability</h3>
                <p className="text-slate-300 leading-relaxed prose prose-invert prose-p:text-slate-300">{result.explainability}</p>
            </div>
            <div className="lg:col-span-2">
                <h3 className="text-xl font-semibold text-cyan-400 mb-3">Segmentation Uncertainty Analysis</h3>
                <p className="text-slate-300 leading-relaxed prose prose-invert prose-p:text-slate-300">{result.segmentationUncertaintyStatement}</p>
            </div>
        </div>
    </div>
);

export const AnalysisResults: React.FC<AnalysisCardProps> = ({ imageState, onRefine, onDelete }) => {
  const [isRefining, setIsRefining] = useState(false);
  const [refinementText, setRefinementText] = useState('');

  const handleRefineSubmit = () => {
    if (refinementText.trim()) {
      onRefine(imageState.id, refinementText);
      setIsRefining(false);
      setRefinementText('');
    }
  };

  const handleSaveAnalysis = () => {
    if (!imageState.result) return;
    const { result } = imageState;
    const reportContent = `
# Retinal OCT Analysis Report
**File:** ${imageState.file.name}
**Date:** ${new Date().toLocaleDateString()}
---
## Diagnosis
- **Condition:** ${result.diagnosis}
- **Confidence:** ${result.confidence}
- **Uncertainty Assessment:** ${result.uncertaintyStatement}
---
## Clinical Explanation
${result.explanation}
---
## Model Interpretability
${result.explainability}
---
## Segmentation Uncertainty Analysis
${result.segmentationUncertaintyStatement}
${result.anomalyReport ? `\n---\n\n## Ancillary Findings\n${result.anomalyReport}` : ''}
---
*Disclaimer: This report is generated by an AI model and is for informational purposes only. It is not a substitute for professional medical advice.*`;

    const blob = new Blob([reportContent.trim()], { type: 'text/markdown;charset=utf-t' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `report-${imageState.file.name.split('.')[0]}.md`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(link.href);
  };
  
  return (
    <div className="bg-slate-800 rounded-2xl p-6 shadow-lg border border-slate-700">
        <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold text-slate-300 truncate pr-4" title={imageState.file.name}>
                {imageState.file.name}
            </h3>
            <button onClick={() => onDelete(imageState.id)} className="text-slate-500 hover:text-red-400 transition-colors">
                <TrashIcon className="w-5 h-5" />
            </button>
        </div>

        <div className="relative">
            {imageState.status === 'loading' && (
                <div className="absolute inset-0 bg-slate-900/80 backdrop-blur-sm flex items-center justify-center z-10 rounded-2xl">
                    <div className="text-center">
                        <div className="w-8 h-8 border-4 border-cyan-400 border-t-transparent rounded-full animate-spin mx-auto mb-2"></div>
                        <p className="text-slate-300 font-semibold">Analyzing...</p>
                    </div>
                </div>
            )}
             <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <img src={imageState.previewUrl} alt="Original OCT Scan" className="rounded-lg w-full object-contain" />
                {imageState.status === 'success' && imageState.segmentedImageUrl && (
                    <img src={imageState.segmentedImageUrl} alt="Segmented OCT Scan" className="rounded-lg w-full object-contain" />
                )}
                {imageState.status === 'success' && imageState.heatmapImageUrl && (
                    <img src={imageState.heatmapImageUrl} alt="Attention Heatmap" className="rounded-lg w-full object-contain" />
                )}
            </div>
        </div>

        {imageState.status === 'pending' && <p className="text-center text-slate-400 mt-4">Ready for analysis.</p>}
        {imageState.status === 'error' && (
            <div className="mt-4 text-red-300 bg-red-900/30 p-4 rounded-lg border border-red-500">
                <p className="font-bold">Analysis Failed</p>
                <p className="text-sm">{imageState.error}</p>
            </div>
        )}

        {imageState.status === 'success' && imageState.result && (
            <>
                {isRefining && (
                    <div className="my-6 p-4 bg-slate-900/50 rounded-lg border border-slate-700">
                        <h3 className="font-semibold text-slate-200 mb-2">Provide Feedback</h3>
                        <textarea
                            value={refinementText}
                            onChange={(e) => setRefinementText(e.target.value)}
                            className="w-full h-24 p-2 bg-slate-800 border border-slate-600 rounded-md focus:ring-2 focus:ring-cyan-500 focus:outline-none text-slate-200"
                            placeholder="e.g., 'Focus on the subretinal fluid.'"
                        />
                        <div className="flex justify-end mt-3">
                            <button onClick={handleRefineSubmit} disabled={!refinementText.trim()} className="px-4 py-2 font-bold text-white bg-cyan-600 hover:bg-cyan-500 rounded-lg disabled:bg-slate-600 disabled:cursor-not-allowed">
                                Submit Refinement
                            </button>
                        </div>
                    </div>
                )}
                <ResultContent result={imageState.result} onSave={handleSaveAnalysis} onRefineToggle={() => setIsRefining(!isRefining)} isRefining={isRefining} />
            </>
        )}
    </div>
  );
};
