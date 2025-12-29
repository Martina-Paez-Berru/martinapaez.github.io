# Portfolio

I build production-grade AI systems that translate complex data into measurable business value. My work spans enterprise RAG pipelines, large-scale forecasting, and agentic workflows—combining rigorous ML engineering with a focus on scalability, interpretability, and real-world impact.

Currently seeking Applied AI/ML Engineer and Quantitative Research roles in financial technology, infrastructure analytics, and enterprise AI systems.

---

## Featured Projects

<div class="projects-grid">

<div class="project-card">
  <h3>🏆 AWS Bedrock Innovation Challenge Winner</h3>
  <p class="project-meta">Amazon Bedrock • Multi-Agent RAG • Lambda • OpenSearch • S3 • Real-Time Inference</p>
  <p><strong>Most Scalable Solution</strong> among 40+ engineering teams. Architected an end-to-end agentic AI research-to-strategy system that automatically transforms 1,000+ academic papers into executable trading algorithms across equities, crypto, and futures markets.</p>
  <p><strong>Technical Approach:</strong> Built a multi-agent RAG pipeline using Amazon Bedrock (Claude 3.5 Sonnet + Titan Embeddings) with Lambda orchestration, OpenSearch vector database, and S3 document storage. Implemented secure grounding with verifiable citations, query routing logic, and parallel processing for sub-second latency.</p>
  <p><strong>Why This Matters:</strong> Quantitative research traditionally requires weeks of manual paper review and strategy formulation. This system achieves >90% reduction in research-to-execution time while maintaining citation accuracy and investment-grade rigor—making sophisticated quant strategies accessible at unprecedented speed.</p>
  <p><strong>Key Innovation:</strong> Near-linear scalability achieved through intelligent chunking, semantic caching, and distributed inference across AWS regions. Cost-optimized through selective model routing (Haiku for routing, Sonnet for complex reasoning).</p>
  <div class="project-links">
    <span class="pill status-award">🏆 Competition Winner</span>
    <span class="pill status-private">🔒 Private Repository</span>
  </div>
</div>

<div class="project-card">
  <h3>AI-Powered Investment Research Platform</h3>
  <p class="project-meta">LightRAG • Neo4j • ChromaDB • Sentence Transformers • LangChain • Python</p>
  <p>Led cross-functional team at Columbia Engineering's Quantitative Research Lab (MathWorks-sponsored) to build an investment ideation acceleration system. Reduced research time by >100 hours/month for portfolio managers and analysts.</p>
  <p><strong>Technical Approach:</strong> Architected modular ingestion-to-retrieval pipeline processing 3.41 GB of financial research data. Engineered Neo4j graph database integrated with ChromaDB vector store, unifying 5,000+ academic papers and 20,000+ social media posts. Implemented LightRAG for entity extraction and relationship mapping, with Sentence Transformers for semantic search.</p>
  <p><strong>Why This Matters:</strong> Portfolio research requires synthesizing disparate sources—academic research, market commentary, regulatory filings. Traditional keyword search misses semantic relationships. Our system enables natural language queries like "find papers connecting ESG metrics to alpha generation in emerging markets" and returns contextualized insights with provenance.</p>
  <p><strong>Impact Metrics:</strong> Reduced signal exploration time from 8 hours to 20 minutes per hypothesis. Increased research coverage by 3x while maintaining quality. Enabled backtesting on 5 years of historical research that was previously unsearchable.</p>
  <p><strong>Design Philosophy:</strong> Built for explainability—every recommendation includes source attribution, confidence scores, and relationship paths through the knowledge graph. Critical for investment decisions where "why" matters as much as "what."</p>
  <div class="project-links">
    <span class="pill status-nda">📄 Technical Brief Available</span>
    <span class="pill status-private">🔒 Columbia IP</span>
  </div>
</div>

<div class="project-card">
  <h3>30-Year Electricity Price Forecasting System</h3>
  <p class="project-meta">Databricks • Apache Spark • RNN • XGBoost • Time Series • Feature Engineering</p>
  <p>Built end-to-end forecasting infrastructure at Ardian (€150B AUM private equity firm) for long-horizon European electricity market modeling. System informs infrastructure investment decisions worth hundreds of millions.</p>
  <p><strong>Technical Approach:</strong> Preprocessed 1.5TB+ of electricity market data spanning 10+ years across 20+ European markets and variables (spot prices, renewable generation, cross-border flows, weather, demand). Engineered Databricks/Spark pipeline for distributed feature computation. Deployed hybrid RNN-XGBoost ensemble combining sequential pattern learning with gradient boosting for regime shifts.</p>
  <p><strong>The Challenge:</strong> Long-horizon electricity forecasting is notoriously difficult—markets exhibit multiple seasonalities, structural breaks from policy changes, and extreme volatility from renewable intermittency. Traditional models either overfit short-term patterns or miss regime changes. We needed 30-year forecasts accurate enough to underwrite billion-dollar infrastructure investments.</p>
  <p><strong>Results:</strong> Reduced mean absolute error by 21% versus Ardian's existing proprietary system. Achieved stable performance across different market regimes (low/high renewable penetration, crisis periods). Enabled scenario analysis: quantified price impact of policy changes like carbon pricing or interconnector additions.</p>
  <p><strong>Key Learning:</strong> Ensemble strength came from complementary failure modes—RNNs captured daily/weekly patterns while XGBoost handled structural breaks. Feature engineering (rolling volatility, renewable capacity factors, cross-market spreads) drove 60% of performance gains.</p>
  <div class="project-links">
    <span class="pill status-nda">📊 Methodology Doc Available</span>
    <span class="pill status-private">🔒 Proprietary Data/Models</span>
  </div>
</div>

<div class="project-card">
  <h3>Full-Stack AI Automation for VC Tech Scouting</h3>
  <p class="project-meta">FastAPI • GPT-4 • PostgreSQL • React • Scrapy • Playwright • Google Slides API</p>
  <p>Engineered complete AI system at Deep Venture Partners to automate technology scouting across 25+ research universities. System identifies emerging technologies, evaluates commercial potential, and generates investment memos automatically.</p>
  <p><strong>Architecture:</strong> Built microservices architecture with FastAPI backend, React frontend, and PostgreSQL for structured data. Orchestrated multi-agent workflows: (1) Research sourcing via Scrapy + Playwright, (2) GPT-4 analysis with prompt chains for technology assessment, (3) Automated presentation generation via Google Slides API with template optimization.</p>
  <p><strong>Technical Innovations:</strong> Designed normalized schema supporting exploratory analysis across institution types, research domains, and funding patterns. Implemented real-time inference with response streaming for interactive user experience. Built experimentation framework for A/B testing presentation strategies—optimized content clarity and VC partner engagement.</p>
  <p><strong>Why This Matters:</strong> Early-stage VC tech scouting is labor-intensive—requires monitoring thousands of research labs, evaluating technical feasibility, and assessing market timing. Manual process meant missing opportunities or incomplete analysis. This system achieves comprehensive coverage with faster decision cycles.</p>
  <p><strong>Business Impact:</strong> Increased scouting coverage from ~50 to 300+ institutions. Reduced memo generation time from 6 hours to 15 minutes. Enabled systematic tracking of research-to-commercialization patterns that inform investment theses.</p>
  <div class="project-links">
    <span class="pill status-nda">🏗️ Architecture Diagram Available</span>
    <span class="pill status-private">🔒 Proprietary System</span>
  </div>
</div>

<div class="project-card">
  <h3>Score-Regularized Bivariate GAN for PE Cash Flow Simulation</h3>
  <p class="project-meta">GANs • PyTorch • Financial Modeling • Curriculum Learning • Deep Learning</p>
  <p>Master's thesis under Professor Ali Hirsa (Columbia IEOR 4742: Deep Learning for Finance and OR). Developed novel GAN architecture for private equity fund cash flow simulation—critical for portfolio construction and liquidity management.</p>
  <p><strong>The Problem:</strong> PE funds exhibit characteristic J-curve patterns (early capital calls, later distributions) with complex cross-channel dependencies. Traditional models either use simple parametric assumptions or Monte Carlo methods that fail to capture realistic temporal dynamics. Need: generate synthetic cash flows indistinguishable from real PE funds for stress testing and portfolio optimization.</p>
  <p><strong>Technical Innovation:</strong> Designed Score-Regularized Bivariate GAN combining three novel components: (1) Hard negative training with frozen Score Machine classifier to enforce realistic J-curves, (2) Dual discriminators for bivariate temporal dependencies between contributions and distributions, (3) Three-phase curriculum learning strategy progressively increasing generation difficulty.</p>
  <p><strong>Methodology:</strong> Implemented in PyTorch with systematic ablation studies. Evaluated via heat equation smoothing for post-processing, endpoint bias analysis, and cross-channel correlation metrics. Optimized lambda parameters for score regularization strength through grid search.</p>
  <p><strong>Results:</strong> Achieved realistic J-curve patterns matching real fund distributions. Maintained proper temporal dependencies (early contributions fund later distributions). Generated synthetic data passed statistical tests used by PE allocators for portfolio construction.</p>
  <p><strong>Research Contribution:</strong> First application of score-based regularization to financial time series GANs. Demonstrated curriculum learning effectiveness for complex temporal generation tasks. Published comprehensive 26-page thesis with formal mathematical formulations.</p>
  <div class="project-links">
    <a href="assets/pe_gan_thesis.pdf" class="pill">📄 Full Thesis</a>
    <span class="pill status-code">💻 Code Available on Request</span>
  </div>
</div>

<div class="project-card">
  <h3>Digital Marketing A/B Testing & Attribution</h3>
  <p class="project-meta">Experimentation • Causal Inference • Statistical Modeling • ROI Analytics</p>
  <p>Designed and executed rigorous A/B testing strategy for Stratyfy (AI-powered credit underwriting platform) comparing ethical vs. functional messaging frameworks for B2B financial services marketing.</p>
  <p><strong>Experimental Design:</strong> Implemented two-arm randomized controlled trial with statistical power analysis (80% power, 5% significance). Treatment: ethical framing emphasizing fairness and financial inclusion. Control: functional framing emphasizing accuracy and efficiency. Randomization at campaign level to prevent contamination.</p>
  <p><strong>Methodology:</strong> Applied causal inference techniques to isolate messaging impact from confounders (time of day, device type, audience segment). Built full-funnel attribution model tracking impression → click → landing page engagement → demo request conversion. Conducted heterogeneity analysis across customer segments.</p>
  <p><strong>Results:</strong> Ethical framing reduced cost-per-click by 18% (p < 0.05). Achieved 2.10% CTR (65+ clicks) versus industry baseline of 1.2%. ROI improvement of 25% when accounting for downstream conversion quality. Demonstrated that values-based messaging resonates in financial services B2B contexts.</p>
  <p><strong>Why This Matters:</strong> B2B SaaS marketing often defaults to feature-focused messaging. This experiment provided quantitative evidence that ethical positioning—when authentic to product capabilities—can improve both engagement metrics and lead quality. Insight informed broader marketing strategy and positioning.</p>
  <p><strong>Technical Rigor:</strong> Pre-registered analysis plan, corrected for multiple testing, validated assumptions (no spillover effects), conducted sensitivity analysis for unmeasured confounding. Built ROI dashboard tracking long-term cohort performance.</p>
  <div class="project-links">
    <a href="assets/stratyfy_ab_test.pdf" class="pill">📊 Experiment Report</a>
    <a href="assets/stratyfy_roi_dashboard.png" class="pill">📈 ROI Dashboard</a>
  </div>
</div>

</div>

---

## Academic & Research Projects

<div class="projects-grid">

<div class="project-card">
  <h3>Transfer Learning: BERT vs LSTM Sentiment Analysis</h3>
  <p class="project-meta">NLP • Transfer Learning • PyTorch • Model Comparison • Ablation Studies</p>
  <p>Comprehensive analysis comparing BERT fine-tuning against LSTM architectures for sentiment classification. Investigated transfer learning effectiveness and optimization strategies.</p>
  <p><strong>Approach:</strong> Implemented both architectures from scratch in PyTorch. BERT: fine-tuned bert-base-uncased with task-specific classification head. LSTM: 2-layer bidirectional architecture with attention mechanism and pre-trained GloVe embeddings.</p>
  <p><strong>Key Findings:</strong> BERT achieved 92.3% accuracy vs 87.1% for LSTM despite 10x fewer training epochs. Transfer learning from large text corpora provided stronger inductive bias than task-specific training. However, LSTM inference was 5x faster—highlighting accuracy-speed tradeoffs for production deployment.</p>
  <p><strong>Ablations:</strong> Studied effect of freezing BERT layers (speed vs accuracy), learning rate schedules (linear warmup optimal), and data augmentation techniques (back-translation improved both models).</p>
  <div class="project-links">
    <a href="https://github.com/martina-paez-berru/bert-lstm-comparison" class="pill">💻 Code & Analysis</a>
  </div>
</div>

<div class="project-card">
  <h3>CNN Image Classification (CIFAR-10)</h3>
  <p class="project-meta">Computer Vision • CNNs • PyTorch • Regularization • Hyperparameter Tuning</p>
  <p>Built and optimized convolutional neural networks for CIFAR-10 image classification. Explored architectural choices and regularization techniques.</p>
  <p><strong>Architecture Evolution:</strong> Started with LeNet-inspired baseline (68% accuracy). Progressed through VGG-style deeper networks (78% accuracy). Final ResNet-inspired architecture with skip connections achieved 89% accuracy.</p>
  <p><strong>Regularization Study:</strong> Systematically evaluated dropout (optimal: 0.3-0.5), batch normalization (essential for deep networks), weight decay (L2: 5e-4 optimal), and data augmentation (random crops + horizontal flips +7% accuracy).</p>
  <p><strong>Key Learning:</strong> Depth helps but with diminishing returns beyond 20 layers without skip connections. Batch normalization more impactful than dropout for CNNs. Data augmentation provides biggest single improvement for small datasets.</p>
  <div class="project-links">
    <a href="https://github.com/martina-paez-berru/cifar10-cnn" class="pill">💻 Code</a>
    <a href="assets/cifar10_results.pdf" class="pill">📊 Results Analysis</a>
  </div>
</div>

<div class="project-card">
  <h3>Generative Models: GANs & VAEs</h3>
  <p class="project-meta">Generative Models • GANs • VAEs • PyTorch • MNIST</p>
  <p>Implemented and compared generative adversarial networks and variational autoencoders for image generation. Studied training dynamics and generation quality.</p>
  <p><strong>GAN Implementation:</strong> Built DCGAN architecture with careful tuning of discriminator/generator learning rates, batch normalization placement, and activation functions. Addressed mode collapse through minibatch discrimination.</p>
  <p><strong>VAE Implementation:</strong> Implemented variational autoencoder with reparameterization trick, KL divergence annealing, and learned prior. Studied latent space structure and interpolation properties.</p>
  <p><strong>Comparison:</strong> GANs produced sharper images but harder to train (mode collapse, unstable gradients). VAEs more stable but slightly blurry outputs. VAEs better for controlled generation via latent space manipulation.</p>
  <div class="project-links">
    <a href="https://github.com/martina-paez-berru/generative-models" class="pill">💻 Code</a>
    <a href="assets/generated_samples.pdf" class="pill">🎨 Generated Samples</a>
  </div>
</div>

<div class="project-card">
  <h3>Neural Networks from Scratch</h3>
  <p class="project-meta">Neural Networks • NumPy • Optimization • Backpropagation</p>
  <p>Built feedforward neural networks from scratch using only NumPy. Deep dive into optimization algorithms, activation functions, and training dynamics.</p>
  <p><strong>Implementation:</strong> Manual backpropagation with computational graph. Implemented SGD, momentum, RMSprop, and Adam optimizers. Tested various activation functions (ReLU, Leaky ReLU, tanh, sigmoid) and initialization schemes (Xavier, He).</p>
  <p><strong>Key Insights:</strong> Adam optimizer most robust across tasks. ReLU + He initialization crucial for deep networks. Gradient clipping prevents exploding gradients. Batch normalization transformative for training stability.</p>
  <p><strong>Educational Value:</strong> Understanding low-level mechanics demystified frameworks like PyTorch. Debugged issues (vanishing gradients, dead ReLUs) by examining weight distributions and activation statistics.</p>
  <div class="project-links">
    <a href="https://github.com/martina-paez-berru/neural-nets-from-scratch" class="pill">💻 Code & Explanations</a>
  </div>
</div>

</div>

---

## Technical Skills

**Machine Learning & AI**  
PyTorch, TensorFlow, Keras, Scikit-learn, LangChain, Sentence Transformers, Hugging Face Transformers, Generative AI (LLMs, Diffusion, GANs), NLP, Computer Vision, Time Series Forecasting, Reinforcement Learning, Transfer Learning, Few-Shot Learning

**Deep Learning Architectures**  
Transformers (BERT, GPT, T5), RNNs/LSTMs/GRUs, CNNs (ResNet, VGG, U-Net), GANs (DCGAN, StyleGAN, Wasserstein), VAEs, Attention Mechanisms, Graph Neural Networks

**Data Engineering & Infrastructure**  
Databricks, Apache Spark (PySpark), PostgreSQL, MySQL, ChromaDB, Neo4j, LightRAG, Pinecone, Vector Databases, Similarity Search, ETL Pipelines, Data Modeling, Schema Design

**Cloud & MLOps**  
AWS (Bedrock, Lambda, S3, SageMaker, OpenSearch, EC2, DynamoDB, API Gateway, IAM, Step Functions, CloudWatch), Docker, Kubernetes (learning), Git, CI/CD, FastAPI, Celery + Redis, RESTful APIs, OAuth2, Model Deployment, A/B Testing Infrastructure

**Programming & Analytics**  
Python (Pandas, NumPy, SciPy, Matplotlib, Seaborn), R (tidyverse, ggplot2), Julia, SQL (advanced queries, optimization), Stata, C++, Excel (advanced formulas, VBA), Jupyter Notebook

**Experimentation & Statistics**  
A/B Testing, Causal Inference, Statistical Modeling, Hypothesis Testing, Bayesian Methods, Survival Analysis, Time Series Analysis, Experimental Design, Power Analysis, Multiple Testing Corrections

**Web Development & Automation**  
React, FastAPI, Flask, Playwright, Scrapy, Beautiful Soup, Selenium, Google APIs (Slides, Sheets, Drive), Web Scraping, Browser Automation

---

## Professional Experience

**Deep Venture Partners** — Data Engineer  
*New York, NY | Jun 2025 - Aug 2025*

Built full-stack AI automation system for venture capital technology scouting across 25+ research universities. Orchestrated multi-agent workflows with real-time GPT-4 inference, FastAPI microservices, and automated presentation generation via Google Slides API.

- Designed normalized PostgreSQL schema enabling scalable exploratory analysis across institution types, research domains, and funding patterns
- Implemented end-to-end pipeline consolidating LLM outputs with automated presentation generation, defining success metrics for template selection
- Sourced academic research via Scrapy + Playwright with robust error handling and rate limiting

**Columbia Engineering | Quantitative Research Lab (MathWorks)** — Project Manager & Data Scientist  
*New York, NY | Jun 2025 - Aug 2025*

Led cross-functional team developing AI-powered investment ideation platform reducing research time by >100 hours/month for portfolio managers and quantitative analysts.

- Architected modular ingestion-to-retrieval pipeline (3.41 GB) using LightRAG stack with Sentence Transformers, LangChain, and ChromaDB
- Engineered Neo4j–ChromaDB graph database unifying 5,000+ academic papers and 20,000+ social posts with semantic search capabilities
- Defined success metrics for retrieval accuracy, evaluated feature strategies, and generated actionable insights for research backtesting

**Ardian** — Data Scientist, Capstone Project  
*New York, NY | Jan 2025 - May 2025*

Developed long-horizon electricity price forecasting system for European markets informing infrastructure investment decisions at €150B AUM private equity firm.

- Preprocessed and extracted insights from 1.5TB+ of electricity market data spanning 10+ years across 20+ European markets
- Built end-to-end Databricks/Spark pipeline training RNN and XGBoost ensembles, reducing forecast error by 21%
- Partnered with energy market experts on feature engineering for price behavior modeling and regime change detection

**Stratyfy** — Digital Marketing Freelancer  
*New York, NY | Jan 2025 - Mar 2025*

Designed and executed A/B testing strategy for AI-powered credit underwriting platform, applying causal inference and statistical modeling to optimize marketing effectiveness.

- Implemented rigorous experimental design (ethical vs. functional framing) reducing cost-per-click by 18%
- Conducted full-funnel attribution analysis with statistical modeling, defining ROI metrics and quantifying long-term cohort performance
- Built interactive dashboards tracking campaign effectiveness (65+ clicks, 2.10% CTR) and downstream conversion quality

---

## Education

**Columbia University** — New York, NY  
Master of Science in Business Analytics, GPA: 3.66/4.00 | *Expected Dec 2025*

*Coursework:* Artificial Intelligence/Deep Learning (NLP, GenAI, Computer Vision), Applied Machine Learning, Optimization (Linear, Convex, Stochastic), Digital Marketing, Marketing Analytics, Probability & Statistics, Simulation, Data Analytics, Managerial Negotiations, Capital Markets

*Certificates:* Google Analytics Certificate

**École Polytechnique Paris** — Paris, France  
Bachelor of Science in Mathematics & Economics, GPA: 3.58/4.00 | *Jun 2024*

*Coursework:* Asymptotic Statistics, Machine Learning Theory, Convex Optimization and Optimal Control, Linear Algebra, Real Analysis, Probability Theory, Measure Theory, Econometrics

---

## Awards & Recognition

**AWS Bedrock Innovation Challenge Winner** — *Nov 2025*  
Most Scalable Solution among 40+ engineering teams. Led design, implementation, and presentation of agentic AI research system. Recognized for secure RAG grounding, verifiable citations, and >10× efficiency gains.

---

## Languages

**English:** Native proficiency  
**Spanish:** Native proficiency  
**French:** Professional working proficiency

---

## About Professional Constraints

Many projects were developed under NDA or within private organizational repositories (Deep Venture Partners, Ardian, Columbia Engineering, AWS). Where full code cannot be shared publicly, I've provided:
- Technical architecture documentation
- Methodology descriptions and design decisions  
- Impact metrics and evaluation results
- System design diagrams

For detailed discussions of implementation approaches, architecture trade-offs, or technical deep-dives, please reach out directly. I'm happy to discuss problem-solving strategies, design patterns, and lessons learned in appropriate contexts.

Academic coursework projects (BERT vs LSTM, CIFAR-10 CNN, Generative Models, Neural Networks from Scratch) have public code repositories available.