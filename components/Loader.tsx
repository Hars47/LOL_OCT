
import React from 'react';

export const Loader: React.FC = () => {
  return (
    <div className="flex flex-col items-center justify-center p-8 bg-slate-800/50 rounded-2xl border border-slate-700">
      <div className="w-12 h-12 border-4 border-cyan-400 border-t-transparent rounded-full animate-spin mb-4"></div>
      <h2 className="text-xl font-semibold text-slate-200">Analyzing Image...</h2>
      <p className="text-slate-400 mt-2">The AI is processing the OCT scan. This may take a moment.</p>
    </div>
  );
};
