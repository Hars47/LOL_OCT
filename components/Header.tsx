
import React from 'react';
import { EyeIcon } from './icons';

export const Header: React.FC = () => {
  return (
    <header className="bg-slate-800/50 backdrop-blur-lg border-b border-slate-700 sticky top-0 z-10">
      <div className="container mx-auto px-4 md:px-8 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <EyeIcon className="w-8 h-8 text-cyan-400" />
          <h1 className="text-2xl font-bold text-slate-100 tracking-tight">
            Retinal OCT AI Analyzer
          </h1>
        </div>
        <div className="text-sm text-slate-400">
        </div>
      </div>
    </header>
  );
};
