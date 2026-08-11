# GitHub Portfolio Spec

Generated: All projects at a glance


## Agents & RAG

### Agentic-Interview-prepartion
- **URL:** https://github.com/oladri-renuka/Agentic-Interview-prepartion
- **Language:** Jupyter Notebook
- **Stars:** 0
- **Description:** LangGraph-based interview prep tool. Generates contextual questions by job title, evaluates answer quality across correctness and clarity dimensions, delivers constructive feedback through state-based agent workflow.

### Autonomous-Game-Playing-Agent-using-Deep-Reinforcement-Learning
- **URL:** https://github.com/oladri-renuka/Autonomous-Game-Playing-Agent-using-Deep-Reinforcement-Learning
- **Language:** Jupyter Notebook
- **Stars:** 0
- **Description:** Board game agent learning strategy through PPO self-play over 5,000 episodes. 3-layer network (256->128->64), action masking for valid moves. Achieves 78% win rate vs random opponent, up from 56% at training start. 100% valid move rate maintained throughout.

### MLOps
- **URL:** https://github.com/oladri-renuka/MLOps
- **Language:** Python
- **Stars:** 0
- **Description:** Description: End-to-end ML pipeline for NYC taxi trip duration prediction. XGBoost with haversine/temporal feature engineering, MLflow experiment tracking, FastAPI serving on Render, 99% pytest coverage.

### arXiv-Research-Paper-Recommendation-System
- **URL:** https://github.com/oladri-renuka/arXiv-Research-Paper-Recommendation-System
- **Language:** Jupyter Notebook
- **Stars:** 0
- **Description:** Research paper discovery across 12,760+ papers using Universal Sentence Encoder embeddings, knowledge graphs, and Graph Convolutional Networks. Precision@5: 0.80, nDCG@5: 0.84, sub-2s response time. Trend analysis and A/B testing framework. Gradio frontend.

### code-memory-agent
- **URL:** https://github.com/oladri-renuka/code-memory-agent
- **Language:** Python
- **Stars:** 0
- **Description:** Coding agent that builds persistent SQLite memory of a codebase, queries memory before reading files, and proves decision consistency across multi-step tasks

### rag-sec-project
- **URL:** https://github.com/oladri-renuka/rag-sec-project
- **Language:** Python
- **Stars:** 0
- **Description:** RAG pipeline for SEC 10-K filings benchmarking 4 retrieval strategies across 116 runs. Hybrid reciprocal rank fusion achieves 0.898 faithfulness (+17% over BM25). Cross-encoder reranker adds 8x latency without proportional gains. Section-aware chunking. Gradio UI.


## Computer Vision

### Automated-Knee-Arthritis-Severity-Classification
- **URL:** https://github.com/oladri-renuka/Automated-Knee-Arthritis-Severity-Classification
- **Language:** Jupyter Notebook
- **Stars:** 0
- **Description:** Knee arthritis severity from X-rays using three approaches: CBAM attention (InceptionResNetV2+MobileNetV2+EfficientNetB5, 96.6% test acc), HOG+SVM (84%), and cascade architecture (EfficientNet->ResNet->DenseNet, 92.3%). Data augmentation for class balancing.

### Brain-tumor-detection-ConvNeXt
- **URL:** https://github.com/oladri-renuka/Brain-tumor-detection-ConvNeXt
- **Language:** Jupyter Notebook
- **Stars:** 0
- **Description:** Brain tumor classification from MRI scans into 42 categories (14 tumor types x 3 imaging modalities) using ConvNeXt Tiny with ImageNet transfer learning. 99.64% validation accuracy vs 92.86% EfficientNetB5 baseline. Data augmentation with random rotation.

### Multimodal-AI-Image-Caption-Audio-Generation-System
- **URL:** https://github.com/oladri-renuka/Multimodal-AI-Image-Caption-Audio-Generation-System
- **Language:** Jupyter Notebook
- **Stars:** 0
- **Description:**  Image-to-audio pipeline: BLIP generates captions, custom neural network synthesizes audio from visual features. 8,475 triplets. BLEU 0.3399, METEOR 0.4878, audio SNR 26.82 dB. Mel-spectrogram processing with GPU acceleration.

### multimodal-ad
- **URL:** https://github.com/oladri-renuka/multimodal-ad
- **Language:** Python
- **Stars:** 0
- **Description:** Multimodal content safety system for detecting weapons,  NSFW content, and counterfeit products in images/videos. Fine-tuned YOLOv8n  (mAP50: 0.649, +110% improvement), OCR/ASR extraction, explainable reasoning,  and interactive Streamlit demo. 100% real data (3,292 OpenImages images),  F1: 0.83, FastAPI backend + Docker containerization. 


## LLM & Inference

### adaptive_agent
- **URL:** https://github.com/oladri-renuka/adaptive_agent
- **Language:** Python
- **Stars:** 0
- **Description:** LangGraph routing agent that classifies LLM requests by complexity and dispatches to cheap (Haiku) or strong (Sonnet) models. Input/output guardrails detect prompt injection and hallucination. 98% routing accuracy, 28% cost reduction vs always-strong baseline on 50-question benchmark.

### early_detection
- **URL:** https://github.com/oladri-renuka/early_detection
- **Language:** Python
- **Stars:** 0
- **Description:** Research: can LLM internal activations predict reasoning failure before it's visible? Linear probe on DeepSeek-R1 hidden states at 150 tokens achieves AUC 0.612 vs 0.445 baseline (p=0.001) on AIME math problems. Signal emerges when surface-level features carry zero information.

### feature-store
- **URL:** https://github.com/oladri-renuka/feature-store
- **Language:** Python
- **Stars:** 0
- **Description:** Real-time feature store with online/offline consistency guarantee. Feature logic defined once, derived to both Redis incremental and Parquet batch paths. 0 mismatches across 200 held-out pairs, p95 4.9ms at 9,300 req/s. Wired into SASRec for live inference.

### inference-router-validation
- **URL:** https://github.com/oladri-renuka/inference-router-validation
- **Language:** Python
- **Stars:** 0
- **Description:** Controlled A/B benchmark of smart routing vs round-robin for LLM inference.

### inference-server
- **URL:** https://github.com/oladri-renuka/inference-server
- **Language:** Python
- **Stars:** 0
- **Description:** Three LLM serving backends for GPT-2-124M built from scratch: naive serial, static batching, and continuous batching with paged KV-cache. Continuous+paged achieves 2.91 req/s with 0 failures vs 2.51 req/s and 6 failures for serial under mixed-length traffic at concurrency=20.

### llm-post-training-pipeline
- **URL:** https://github.com/oladri-renuka/llm-post-training-pipeline
- **Language:** Python
- **Stars:** 0
- **Description:** Complete post-training for LLaMA-3.2-1B: SFT on 52K Alpaca examples, reward modeling on 4.7K preference pairs, DPO. Pivoted from PPO due to TRL/rotary attention incompatibility. Factual accuracy +9pp (p=0.030), format compliance -16.7pp (p=0.0003).

### on-device-camera-enhancement
- **URL:** https://github.com/oladri-renuka/on-device-camera-enhancement
- **Language:** Python
- **Stars:** 0
- **Description:** On-device low-light enhancement and portrait segmentation pipeline.  Implements Zero-DCE and U-Net with INT8 quantization for real-time  inference on mobile and edge devices.

### probe-guided-inference
- **URL:** https://github.com/oladri-renuka/probe-guided-inference
- **Language:** Python
- **Stars:** 0
- **Description:** Routing an inference scheduler on a mechanistic interpretability probe  a layer-16 activation read at token 150 predicts reasoning convergence and decides batch priority, before any behavioral signal is visible.

### rag-app
- **URL:** https://github.com/oladri-renuka/rag-app
- **Language:** Python
- **Stars:** 0
- **Description:** Production RAG system: LangGraph orchestration, FAISS vector store, Ollama local LLM, MLflow query logging. FastAPI + Streamlit frontend. Dockerized, deployable on HuggingFace Spaces.

### recsys
- **URL:** https://github.com/oladri-renuka/recsys
- **Language:** Python
- **Stars:** 0
- **Description:** SASRec Movie Recommendation API on AWS. Self-attention sequential recommender with 2 attention blocks, 50-dim embeddings. NDCG@10: 58.11%, Hit@10: 78.49%. 1.81ms inference, 8,366 req/s on CPU. Deployed on EC2 with Docker, Streamlit dashboard. Trained on MovieLens-1M.

### silent-failures
- **URL:** https://github.com/oladri-renuka/silent-failures
- **Language:** Python
- **Stars:** 0
- **Description:** Official code for Silent Failures in Quantized LLM Reasoning — a taxonomy-based analysis of failure mode shifts under post-training quantization


## ML/Data

### two-stage-recsys
- **URL:** https://github.com/oladri-renuka/two-stage-recsys
- **Language:** Python
- **Stars:** 0
- **Description:** Two-tower retrieval (FAISS) + SASRec ranking, wired to a real-time Kafka/Redis feature store  a two-stage retrieve-then-rank pipeline extending recsys and feature-store.


## Other

### knowledge-agent
- **URL:** https://github.com/oladri-renuka/knowledge-agent
- **Language:** HTML
- **Stars:** 0
- **Description:** Persistent belief graph from document sequences. Extracts entities/claims, resolves duplicates via two-pass entity resolution (alias + cosine similarity), detects contradictions across sources, answers questions with citations. 936 entities from 173 pages using 9 API calls. MCP server for Claude Desktop/Cursor.

### mindmirror
- **URL:** https://github.com/oladri-renuka/mindmirror
- **Language:** Python
- **Stars:** 0
- **Description:** Real-time AI interview coach analyzing eye contact, facial expressions, speech, vocal patterns, and filler words every 2 seconds. Personal baseline calibration, multiplicative delivery x content scoring. MediaPipe + faster-whisper + LangGraph. Full pipeline cycle ~1.2s.

### resume_classification
- **URL:** https://github.com/oladri-renuka/resume_classification
- **Language:** Jupyter Notebook
- **Stars:** 0
- **Description:** Resume classification into 25 job categories using TF-IDF and SMOTE class balancing. 8 classifiers benchmarked. Logistic Regression and SVM tied at 99.48% accuracy (F1 99.49%) on 962 resumes. Full preprocessing: URL/mention removal, stopwords, normalization.

### sparse-factor-modeling
- **URL:** https://github.com/oladri-renuka/sparse-factor-modeling
- **Language:** Jupyter Notebook
- **Stars:** 0
- **Description:** Equity return prediction using Fama-French factors with 9 LASSO solvers from scratch (proximal GD, FISTA, Barzilai-Borwein, coordinate descent, online, DRO). Walk-forward backtest, no look-ahead bias. Best Sharpe 5.061, 59.14% annual return. Streamlit dashboard.

### token-efficiency-math-reasoning
- **URL:** https://github.com/oladri-renuka/token-efficiency-math-reasoning
- **Language:** Python
- **Stars:** 0
- **Description:** Description: Does more thinking tokens improve reasoning accuracy? Tests DeepSeek-R1 on GSM8K, MATH-500, AIME. Accuracy plateaus at 256 tokens on easy benchmarks. On AIME: bimodal split where 57% converge at 4100 tokens (96.5% acc), 43% never converge (11.5% acc).


## Research

### Fine-grained-Factual-Consistency-Evaluation-for-LLM
- **URL:** https://github.com/oladri-renuka/Fine-grained-Factual-Consistency-Evaluation-for-LLM
- **Language:** Jupyter Notebook
- **Stars:** 0
- **Description:** Multi-stage factual accuracy pipeline: Mistral-7B generates, T5-Flan-Large decomposes into atomic facts, RoBERTa-Large-MNLI verifies via NLI. 400 ASQA samples. Entailment 28.7%, contradiction 11.2%, neutral 43.5%. Confidence correlates with accuracy (ANOVA p<0.001).

### code-gen-eval
- **URL:** https://github.com/oladri-renuka/code-gen-eval
- **Language:** Python
- **Stars:** 0
- **Description:** Failure taxonomy for Codestral across HumanEval + MBPP (538 problems). Hypothesis: failure mode distribution shifts with difficulty. Result: it doesn't for a capable model — when Codestral fails it's almost always logic error regardless of difficulty. Chi-square p=0.487. Honest null result including why the test had limited power.

### factuality-verification-analysis
- **URL:** https://github.com/oladri-renuka/factuality-verification-analysis
- **Language:** Python
- **Stars:** 0
- **Description:** Error analysis and failure mode comparison of atomic factuality verification methods on FActScore dataset

### vlm-hallucination
- **URL:** https://github.com/oladri-renuka/vlm-hallucination
- **Language:** Python
- **Stars:** 0
- **Description:** Description: Domain-shift failure analysis of LLaVA-1.5-7B and InternVL2-8B across photos, charts, medical, and screenshots. 945 probes, 6 failure categories. Chart OCR: LLaVA 0% vs InternVL2 71%. Domain-dependent failure confirmed via chi-square (p<0.0001).


## Systems/Performance

### cpp-simd-quant
- **URL:** https://github.com/oladri-renuka/cpp-simd-quant
- **Language:** C++
- **Stars:** 0
- **Description:** ARM NEON SIMD optimizations on Apple Silicon for attention (31.88 GFLOPS/s, 11.1x speedup) and Black-Scholes (1.03x, proving SIMD fails on transcendental-heavy workloads). Roofline analysis predicts optimization potential before implementation. OpenMP parallelization.

### cuda-attention-kernel
- **URL:** https://github.com/oladri-renuka/cuda-attention-kernel
- **Language:** Cuda
- **Stars:** 0
- **Description:** Transformer attention in CUDA: naive (global DRAM) vs tiled (32x32 shared memory) kernels on A100. Tiled: 515.5 GFLOPS/s, 1.2x over naive, ~145x over CPU. Demonstrates why A100 L2 cache masks tiling benefits and the progression toward Flash Attention.

