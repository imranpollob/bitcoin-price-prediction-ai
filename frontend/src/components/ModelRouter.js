import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import Dashboard from '../Dashboard';
import NaiveModelView from './models/NaiveModelView';
import DenseModelView from './models/DenseModelView';
import Conv1DModelView from './models/Conv1DModelView';
import LSTMModelView from './models/LSTMModelView';
import NBEATSModelView from './models/NBEATSModelView';
import EnsembleModelView from './models/EnsembleModelView';

const ModelRouter = () => {
  return (
    <Routes>
      <Route path="/" element={<Dashboard />} />
      <Route path="/dashboard" element={<Dashboard />} />
      <Route path="/naive" element={<NaiveModelView />} />
      <Route path="/dense" element={<DenseModelView />} />
      <Route path="/conv1d" element={<Conv1DModelView />} />
      <Route path="/lstm" element={<LSTMModelView />} />
      <Route path="/nbeats" element={<NBEATSModelView />} />
      <Route path="/ensemble" element={<EnsembleModelView />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
};

export default ModelRouter;