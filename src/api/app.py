"""FastAPI backend for multimodal content safety reviewer"""

import json
import logging
import os
from pathlib import Path
from uuid import uuid4
from datetime import datetime
from typing import Dict

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.api.models import (
    AnalysisRequest, AnalysisResponse, HealthCheck
)
from src.api.inference import InferenceOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multimodal Content Safety Reviewer",
    description="Production-grade detection of weapons, NSFW, and counterfeit content",
    version="1.0.0"
)

# CORS middleware for demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference orchestrator (lazy-loaded)
orchestrator = None
analysis_jobs: Dict[str, AnalysisResponse] = {}


def get_orchestrator():
    """Lazy-load orchestrator on first use"""
    global orchestrator
    if orchestrator is None:
        logger.info("Loading inference orchestrator...")
        try:
            from src.api.inference import InferenceOrchestrator
            orchestrator = InferenceOrchestrator()
            logger.info("✅ Orchestrator loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load orchestrator: {e}")
            raise
    return orchestrator


@app.on_event("startup")
async def startup_event():
    """Startup event - just log, don't load models yet"""
    logger.info("=" * 70)
    logger.info("PHASE 6: FASTAPI BACKEND STARTUP")
    logger.info("=" * 70)
    logger.info("✅ Server starting (models loaded on first request)")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down FastAPI server")


@app.get("/", response_class=HTMLResponse)
async def demo():
    """Serve interactive HTML demo"""
    return get_demo_html()


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    try:
        orch = get_orchestrator()
        return HealthCheck(
            status="healthy",
            models_loaded=orch.models_status,
            gpu_available=orch.gpu_available,
            memory_usage_mb=orch.get_memory_usage()
        )
    except:
        return HealthCheck(
            status="loading",
            models_loaded={"status": "initializing"},
            gpu_available=False,
            memory_usage_mb=0.0
        )


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    detector_threshold: float = 0.45,
    ocr_enabled: bool = True,
    reasoning_enabled: bool = True
):
    """Analyze an uploaded image for violations"""

    try:
        orch = get_orchestrator()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to load models: {str(e)}")

    job_id = str(uuid4())[:8]

    # Save uploaded file
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    file_path = upload_dir / file.filename

    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)

        logger.info(f"\n{'='*70}")
        logger.info(f"JOB {job_id}: Analyzing {file.filename}")
        logger.info(f"{'='*70}")

        # Run inference
        result = await orch.analyze_image(
            file_path,
            detector_threshold=detector_threshold,
            ocr_enabled=ocr_enabled,
            reasoning_enabled=reasoning_enabled
        )

        # Create response
        response = AnalysisResponse(
            job_id=job_id,
            status="completed",
            file_name=file.filename,
            frames_analyzed=len(result.get("frames", [])),
            violations_detected=result.get("violations_detected", 0),
            frames=result.get("frames", []),
            summary=result.get("summary", {}),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        # Store result
        analysis_jobs[job_id] = response

        logger.info(f"✅ Job {job_id} completed")
        logger.info(f"   Violations detected: {response.violations_detected}")

        return response

    except Exception as e:
        logger.error(f"❌ Job {job_id} failed: {e}")

        response = AnalysisResponse(
            job_id=job_id,
            status="failed",
            file_name=file.filename,
            error=str(e),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        analysis_jobs[job_id] = response
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/results/{job_id}", response_model=AnalysisResponse)
async def get_results(job_id: str):
    """Get analysis results by job ID"""
    if job_id not in analysis_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return analysis_jobs[job_id]


@app.get("/jobs")
async def list_jobs():
    """List all analysis jobs"""
    return {
        "total_jobs": len(analysis_jobs),
        "jobs": [
            {
                "job_id": job_id,
                "status": result.status,
                "file_name": result.file_name,
                "violations_detected": result.violations_detected,
                "created_at": result.created_at.isoformat()
            }
            for job_id, result in analysis_jobs.items()
        ]
    }


@app.get("/metrics")
async def get_metrics():
    """Get inference metrics and performance stats"""
    if orchestrator is None:
        return {"error": "Orchestrator not initialized"}

    return {
        "total_jobs": len(analysis_jobs),
        "completed_jobs": sum(1 for r in analysis_jobs.values() if r.status == "completed"),
        "failed_jobs": sum(1 for r in analysis_jobs.values() if r.status == "failed"),
        "total_violations_detected": sum(r.violations_detected for r in analysis_jobs.values()),
        "models_status": orchestrator.models_status,
        "average_latency_ms": orchestrator.get_average_latency(),
    }


def get_demo_html() -> str:
    """Generate demo HTML with interactive UI"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Multimodal Content Safety Reviewer</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }

            .container {
                background: white;
                border-radius: 16px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                max-width: 800px;
                width: 100%;
                padding: 40px;
            }

            header {
                text-align: center;
                margin-bottom: 40px;
            }

            h1 {
                font-size: 32px;
                color: #333;
                margin-bottom: 8px;
            }

            .subtitle {
                color: #666;
                font-size: 16px;
            }

            .upload-section {
                margin-bottom: 40px;
            }

            .upload-area {
                border: 2px dashed #667eea;
                border-radius: 12px;
                padding: 40px;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s ease;
                background: #f8f9ff;
            }

            .upload-area:hover {
                border-color: #764ba2;
                background: #f0f2ff;
            }

            .upload-area.dragover {
                border-color: #764ba2;
                background: #f0f2ff;
            }

            .upload-icon {
                font-size: 48px;
                margin-bottom: 12px;
            }

            .upload-text {
                color: #333;
                font-size: 16px;
                font-weight: 500;
                margin-bottom: 4px;
            }

            .upload-hint {
                color: #999;
                font-size: 14px;
            }

            #fileInput {
                display: none;
            }

            .settings {
                margin-bottom: 30px;
                padding: 20px;
                background: #f8f9ff;
                border-radius: 12px;
            }

            .setting-group {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 16px;
            }

            .setting-group:last-child {
                margin-bottom: 0;
            }

            .setting-label {
                display: flex;
                flex-direction: column;
                gap: 4px;
            }

            .setting-title {
                color: #333;
                font-weight: 500;
                font-size: 14px;
            }

            .setting-description {
                color: #999;
                font-size: 12px;
            }

            input[type="checkbox"] {
                width: 20px;
                height: 20px;
                cursor: pointer;
            }

            input[type="range"] {
                width: 150px;
                cursor: pointer;
            }

            .threshold-value {
                color: #667eea;
                font-weight: 600;
                min-width: 40px;
            }

            .button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 14px 32px;
                font-size: 16px;
                font-weight: 600;
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.3s ease;
                width: 100%;
            }

            .button:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
            }

            .button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }

            .results {
                margin-top: 40px;
                padding: 20px;
                background: #f8f9ff;
                border-radius: 12px;
                display: none;
            }

            .results.show {
                display: block;
            }

            .result-item {
                margin-bottom: 12px;
                padding: 12px;
                background: white;
                border-radius: 8px;
                border-left: 4px solid #667eea;
            }

            .result-violation {
                border-left-color: #e74c3c;
                background: #fff5f5;
            }

            .result-title {
                font-weight: 600;
                color: #333;
                margin-bottom: 4px;
            }

            .result-detail {
                font-size: 13px;
                color: #666;
            }

            .badge {
                display: inline-block;
                padding: 4px 12px;
                border-radius: 12px;
                font-size: 12px;
                font-weight: 600;
                margin-right: 8px;
                margin-bottom: 8px;
            }

            .badge-weapon {
                background: #ffe0e0;
                color: #c00;
            }

            .badge-nsfw {
                background: #fff0e0;
                color: #c60;
            }

            .badge-counterfeit {
                background: #e0f0ff;
                color: #0066cc;
            }

            .badge-flag {
                background: #ffe0e0;
                color: #c00;
            }

            .badge-review {
                background: #fff0e0;
                color: #c60;
            }

            .badge-allow {
                background: #e0ffe0;
                color: #0c0;
            }

            .loading {
                display: none;
                text-align: center;
                padding: 20px;
            }

            .loading.show {
                display: block;
            }

            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 30px;
                height: 30px;
                animation: spin 1s linear infinite;
                margin: 0 auto 12px;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            .error {
                color: #c00;
                padding: 12px;
                background: #ffe0e0;
                border-radius: 8px;
                margin-top: 20px;
                display: none;
            }

            .error.show {
                display: block;
            }

            .stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 12px;
                margin-top: 20px;
            }

            .stat-box {
                background: white;
                padding: 16px;
                border-radius: 8px;
                text-align: center;
                border: 1px solid #eee;
            }

            .stat-value {
                font-size: 24px;
                font-weight: 700;
                color: #667eea;
            }

            .stat-label {
                font-size: 12px;
                color: #999;
                margin-top: 4px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>🛡️ Content Safety Reviewer</h1>
                <p class="subtitle">Detect weapons, NSFW, and counterfeit content with AI</p>
            </header>

            <div class="upload-section">
                <div class="upload-area" id="uploadArea">
                    <div class="upload-icon">📸</div>
                    <div class="upload-text">Click or drag to upload</div>
                    <div class="upload-hint">Supports JPG, PNG (under 10MB)</div>
                </div>
                <input type="file" id="fileInput" accept="image/*">
            </div>

            <div class="settings">
                <div class="setting-group">
                    <div class="setting-label">
                        <span class="setting-title">Detector Confidence</span>
                        <span class="setting-description">Higher = more strict</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <input type="range" id="threshold" min="0.1" max="0.9" step="0.05" value="0.45">
                        <span class="threshold-value" id="thresholdValue">0.45</span>
                    </div>
                </div>

                <div class="setting-group">
                    <div class="setting-label">
                        <span class="setting-title">OCR Text Extraction</span>
                        <span class="setting-description">Extract visible text</span>
                    </div>
                    <input type="checkbox" id="ocrEnabled" checked>
                </div>

                <div class="setting-group">
                    <div class="setting-label">
                        <span class="setting-title">VLM Reasoning</span>
                        <span class="setting-description">Generate explanations</span>
                    </div>
                    <input type="checkbox" id="reasoningEnabled" checked>
                </div>
            </div>

            <button class="button" id="analyzeBtn">Analyze Image</button>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Analyzing image...</p>
            </div>

            <div class="error" id="error"></div>

            <div class="results" id="results">
                <h2 style="margin-bottom: 16px;">📊 Analysis Results</h2>
                <div id="resultsList"></div>
            </div>
        </div>

        <script>
            const uploadArea = document.getElementById('uploadArea');
            const fileInput = document.getElementById('fileInput');
            const analyzeBtn = document.getElementById('analyzeBtn');
            const loading = document.getElementById('loading');
            const error = document.getElementById('error');
            const results = document.getElementById('results');
            const resultsList = document.getElementById('resultsList');
            const threshold = document.getElementById('threshold');
            const thresholdValue = document.getElementById('thresholdValue');

            let selectedFile = null;

            // Threshold slider
            threshold.addEventListener('input', (e) => {
                thresholdValue.textContent = parseFloat(e.target.value).toFixed(2);
            });

            // Upload area click
            uploadArea.addEventListener('click', () => fileInput.click());

            // File selection
            fileInput.addEventListener('change', (e) => {
                selectedFile = e.target.files[0];
                if (selectedFile) {
                    uploadArea.innerHTML = `<div class="upload-icon">✅</div><div class="upload-text">${selectedFile.name}</div><div class="upload-hint">Ready to analyze</div>`;
                }
            });

            // Drag and drop
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });

            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });

            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                selectedFile = e.dataTransfer.files[0];
                if (selectedFile) {
                    uploadArea.innerHTML = `<div class="upload-icon">✅</div><div class="upload-text">${selectedFile.name}</div><div class="upload-hint">Ready to analyze</div>`;
                }
            });

            // Analyze button
            analyzeBtn.addEventListener('click', async () => {
                if (!selectedFile) {
                    showError('Please select an image first');
                    return;
                }

                analyzeBtn.disabled = true;
                loading.classList.add('show');
                error.classList.remove('show');
                results.classList.remove('show');

                try {
                    const formData = new FormData();
                    formData.append('file', selectedFile);
                    formData.append('detector_threshold', threshold.value);
                    formData.append('ocr_enabled', document.getElementById('ocrEnabled').checked);
                    formData.append('reasoning_enabled', document.getElementById('reasoningEnabled').checked);

                    const response = await fetch('/analyze', {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) {
                        throw new Error(await response.text());
                    }

                    const data = await response.json();
                    displayResults(data);

                } catch (err) {
                    showError('Analysis failed: ' + err.message);
                } finally {
                    analyzeBtn.disabled = false;
                    loading.classList.remove('show');
                }
            });

            function displayResults(data) {
                const violations = data.violations_detected;
                let html = '<div class="stats">';
                html += `<div class="stat-box"><div class="stat-value">${violations}</div><div class="stat-label">Violations</div></div>`;
                html += `<div class="stat-box"><div class="stat-value">${data.frames_analyzed}</div><div class="stat-label">Frames</div></div>`;
                html += '</div>';

                if (violations === 0) {
                    html += '<div class="result-item"><div class="result-title">✅ No violations detected</div></div>';
                } else {
                    data.frames.forEach(frame => {
                        frame.reasoning.forEach(verdict => {
                            const badgeClass = verdict.violation_type.includes('weapon') ? 'badge-weapon' :
                                              verdict.violation_type.includes('nsfw') ? 'badge-nsfw' :
                                              'badge-counterfeit';

                            const actionBadgeClass = 'badge-' + verdict.recommended_action;

                            html += `<div class="result-item result-violation">
                                <div class="result-title">
                                    <span class="badge ${badgeClass}">${verdict.violation_type.toUpperCase()}</span>
                                    <span class="badge ${actionBadgeClass}">${verdict.recommended_action.toUpperCase()}</span>
                                </div>
                                <div class="result-detail"><strong>Confidence:</strong> ${(verdict.confidence * 100).toFixed(1)}%</div>
                                <div class="result-detail"><strong>Reasoning:</strong> ${verdict.reasoning}</div>
                                <div class="result-detail"><strong>Evidence:</strong> ${verdict.evidence.join(', ')}</div>
                            </div>`;
                        });
                    });
                }

                resultsList.innerHTML = html;
                results.classList.add('show');
            }

            function showError(message) {
                error.textContent = message;
                error.classList.add('show');
            }
        </script>
    </body>
    </html>
    """


def run():
    """Run the FastAPI server"""
    logger.info("\n" + "="*70)
    logger.info("PHASE 6: FASTAPI BACKEND")
    logger.info("="*70)
    logger.info("\n🚀 Starting FastAPI server...")
    logger.info("   URL: http://localhost:8000")
    logger.info("   Demo: http://localhost:8000/")
    logger.info("   Health: http://localhost:8000/health")
    logger.info("   Metrics: http://localhost:8000/metrics")

    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    run()
