import React, { useState, useCallback, useMemo } from 'react';
import { Header } from './components/Header';
import { ImageUploader } from './components/ImageUploader';
import { AnalysisResults } from './components/AnalysisResults';
import { analyzeImage } from './services/geminiService';
import type { AnalyzableImage } from './types';
import { UploadIcon, AlertTriangleIcon, CheckCircleIcon } from './components/icons';

const App: React.FC = () => {
  const [images, setImages] = useState<AnalyzableImage[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleImageUpload = (files: FileList) => {
    const newImages: AnalyzableImage[] = Array.from(files)
      .filter(file => file.type.startsWith('image/'))
      .map(file => ({
        id: crypto.randomUUID(),
        file: file,
        previewUrl: URL.createObjectURL(file),
        status: 'pending',
      }));
    setImages(prev => [...prev, ...newImages]);
    setError(null);
  };

  const handleAnalyzeAll = useCallback(async () => {
    setIsAnalyzing(true);
    setError(null);

    const imagesToAnalyze = images.filter(img => img.status === 'pending');
    if (imagesToAnalyze.length === 0) {
        setIsAnalyzing(false);
        return;
    }

    setImages(prev => prev.map(img => 
        img.status === 'pending' ? { ...img, status: 'loading' } : img
    ));

    for (const image of imagesToAnalyze) {
        try {
            const { analysis, segmentedImageBase64, heatmapImageBase64, segmentationUncertaintyMapBase64 } = await analyzeImage(image.file);
            setImages(prev => prev.map(img =>
                img.id === image.id ? {
                    ...img,
                    status: 'success',
                    result: analysis,
                    segmentedImageUrl: `data:image/jpeg;base64,${segmentedImageBase64}`,
                    heatmapImageUrl: `data:image/jpeg;base64,${heatmapImageBase64}`,
                    segmentationUncertaintyMapUrl: `data:image/jpeg;base64,${segmentationUncertaintyMapBase64}`
                } : img
            ));
        } catch (err) {
            setImages(prev => prev.map(img =>
                img.id === image.id ? { ...img, status: 'error', error: (err as Error).message } : img
            ));
        }
    }
    setIsAnalyzing(false);
  }, [images]);
  
  const handleRefineAnalysis = useCallback(async (id: string, feedback: string) => {
    const imageToRefine = images.find(img => img.id === id);
    if (!imageToRefine) return;

    setImages(prev => prev.map(img => img.id === id ? { ...img, status: 'loading' } : img));

    try {
        const { analysis, heatmapImageBase64 } = await analyzeImage(imageToRefine.file, feedback);
        setImages(prev => prev.map(img =>
            img.id === id ? {
                ...img,
                status: 'success',
                result: analysis,
                heatmapImageUrl: `data:image/jpeg;base64,${heatmapImageBase64}`
                // Note: segmentation map is intentionally not updated on refine
            } : img
        ));
    } catch (err) {
        setImages(prev => prev.map(img =>
            img.id === id ? { ...img, status: 'error', error: (err as Error).message } : img
        ));
    }
  }, [images]);

  const handleDeleteImage = (id: string) => {
    setImages(prev => {
        const imageToDelete = prev.find(img => img.id === id);
        if (imageToDelete) {
            URL.revokeObjectURL(imageToDelete.previewUrl);
        }
        return prev.filter(img => img.id !== id);
    });
  };
  
  const pendingCount = useMemo(() => images.filter(i => i.status === 'pending').length, [images]);

  const WelcomeState: React.FC = () => (
    <div className="text-center p-8 border-2 border-dashed border-slate-600 rounded-2xl bg-slate-800/50 col-span-full">
      <div className="flex justify-center mb-4">
        <div className="p-4 bg-slate-700 rounded-full">
          <UploadIcon className="w-8 h-8 text-cyan-400" />
        </div>
      </div>
      <h2 className="text-2xl font-bold text-slate-100 mb-2">Upload Retinal OCT Images</h2>
      <p className="text-slate-400">
        Select one or more images to begin a batch diagnostic analysis.
      </p>
    </div>
  );

  return (
    <div className="min-h-screen bg-slate-900 font-sans">
      <Header />
      <main className="container mx-auto p-4 md:p-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          <aside className="lg:col-span-4 xl:col-span-3">
            <div className="sticky top-8 bg-slate-800 rounded-2xl p-6 shadow-2xl shadow-slate-950/50 border border-slate-700">
              <h2 className="text-xl font-semibold mb-4 text-cyan-400">Control Panel</h2>
              <ImageUploader onImageUpload={handleImageUpload} />
              {images.length > 0 && (
                <div className="mt-4 flex items-center text-sm text-green-400 bg-green-900/50 p-3 rounded-lg">
                    <CheckCircleIcon className="w-5 h-5 mr-2 flex-shrink-0"/>
                    <span className="truncate">
                        {images.length} image{images.length > 1 ? 's' : ''} loaded.
                    </span>
                </div>
              )}
              <button
                onClick={handleAnalyzeAll}
                disabled={pendingCount === 0 || isAnalyzing}
                className="w-full mt-4 bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-600 disabled:cursor-not-allowed text-white font-bold py-3 px-4 rounded-lg transition-all duration-300 transform hover:scale-105 focus:outline-none focus:ring-2 focus:ring-cyan-400 focus:ring-opacity-75"
              >
                {isAnalyzing ? `Analyzing...` : `Analyze ${pendingCount > 0 ? pendingCount : ''} Image${pendingCount !== 1 ? 's' : ''}`}
              </button>
              {error && (
                <div className="mt-4 bg-red-900/30 border border-red-500 text-red-300 px-4 py-3 rounded-lg" role="alert">
                  <div className="flex">
                    <div className="py-1"><AlertTriangleIcon className="w-6 h-6 text-red-400 mr-3 flex-shrink-0"/></div>
                    <div>
                      <p className="font-bold">Analysis Failed</p>
                      <p className="text-sm">{error}</p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </aside>

          <div className="lg:col-span-8 xl:col-span-9">
            <div className="grid grid-cols-1 gap-8">
                {images.length === 0 && <WelcomeState />}
                {images.map(image => (
                    <AnalysisResults
                        key={image.id}
                        imageState={image}
                        onRefine={handleRefineAnalysis}
                        onDelete={handleDeleteImage}
                    />
                ))}
            </div>
          </div>
        </div>
      </main>
      <footer className="text-center py-6 text-slate-500 text-sm">
        <p>This tool is for informational purposes only and not a substitute for professional medical advice.</p>
      </footer>
    </div>
  );
};

export default App;
