import React, { useState } from 'react';
import axios from 'axios';
import { Upload, Activity, Shield, Info, Loader2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
    }
  };

  const analyzeScan = async () => {
    if (!file) return;
    setLoading(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      // Connects to your live FastAPI backend
      const response = await axios.post('http://localhost:8000/predict', formData);
      setResult(response.data);
    } catch (error) {
      console.error("Diagnosis Failed:", error);
      alert("Backend Connection Error. Ensure your FastAPI server is running on port 8000.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dashboard-container">
      <header className="header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <Shield color="#58a6ff" size={32} />
          <h1 style={{ margin: 0, fontSize: '1.5rem' }}>LungNet Vision <span style={{ color: '#8b949e', fontWeight: 'normal' }}>| v1.0</span></h1>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '2rem' }}>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '0.8rem', color: '#8b949e' }}>System Status</div>
            <div style={{ fontSize: '0.9rem', color: '#3fb950' }}>● AI Core Online</div>
          </div>
        </div>
      </header>

      <main>
        {/* Upload Section */}
        <section className="dropzone" onClick={() => document.getElementById('fileInput').click()}>
          <input type="file" id="fileInput" hidden onChange={handleFileChange} accept="image/*" />
          <Upload size={48} color="#58a6ff" style={{ marginBottom: '1rem' }} />
          <h2 style={{ margin: '0 0 0.5rem 0' }}>Upload Medical Scan (DICOM/PNG)</h2>
          <p style={{ color: '#8b949e', margin: 0 }}>Drag and drop or click to browse patient files</p>
        </section>

        {preview && (
          <div style={{ marginTop: '2rem', textAlign: 'center' }}>
            <button
              onClick={analyzeScan}
              disabled={loading}
              style={{
                background: '#58a6ff',
                color: '#fff',
                border: 'none',
                padding: '1rem 3rem',
                borderRadius: '8px',
                fontWeight: 'bold',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                margin: '0 auto'
              }}
            >
              {loading ? <Loader2 className="animate-spin" /> : <Activity size={20} />}
              {loading ? "Analyzing Neural Pathways..." : "Run AI Diagnosis"}
            </button>
          </div>
        )}

        {/* Results Window */}
        <AnimatePresence>
          {result && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="result-grid"
            >
              <div className="image-card">
                <div style={{ marginBottom: '1rem', color: '#8b949e', fontSize: '0.8rem' }}>SOURCE RADIOGRAPH</div>
                <img src={preview} alt="Original" />
              </div>

              <div className="image-card">
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem' }}>
                  <span style={{ color: '#8b949e', fontSize: '0.8rem' }}>AI FOCUS MAP (GRAD-CAM)</span>
                  <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                    <span className={`diagnosis-badge ${result.diagnosis}`} style={{ margin: 0 }}>
                      {result.diagnosis.toUpperCase()}
                    </span>
                    <span style={{ fontSize: '0.85rem', color: '#8b949e', fontWeight: 'bold' }}>
                      {result.certainty}% Certainty
                    </span>
                  </div>
                </div>
                <img src={result.heatmap} alt="AI Heatmap" />
                
                {/* Trust Meter */}
                <div style={{ marginTop: '1.5rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                    <span style={{ fontSize: '0.75rem', color: '#8b949e' }}>MODEL TRUST METER</span>
                    <span style={{ fontSize: '0.75rem', color: result.certainty > 90 ? '#3fb950' : '#d29922' }}>
                      {result.certainty > 90 ? 'High Confidence' : 'Review Required'}
                    </span>
                  </div>
                  <div className="trust-meter-bg">
                    <motion.div 
                      className={`trust-meter-fill ${result.diagnosis}`}
                      initial={{ width: 0 }}
                      animate={{ width: `${result.certainty}%` }}
                      transition={{ duration: 1, ease: "easeOut" }}
                    />
                  </div>
                </div>

                <div style={{ marginTop: '1.5rem', display: 'flex', gap: '1rem', alignItems: 'flex-start' }}>
                  <Info size={20} color="#8b949e" />
                  <p style={{ margin: 0, fontSize: '0.85rem', color: '#8b949e', lineHeight: '1.4' }}>
                    The heatmap indicates the model's spatial attention. Red areas represent high-contribution features for the <strong>{result.diagnosis}</strong> classification.
                  </p>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}

export default App;
