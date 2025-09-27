# 📚 21 Projects in 21 Days – ML, Deep Learning & GenAI

Welcome to my 21 Days, 21 Projects journey with GeeksforGeeks!

This repo is a daily log of the projects I’ll be building — covering Machine Learning, Deep Learning, and Generative AI.
________________________________________________________________________________________________________________________________
The goal?

🚀 Build consistency

🧠 Strengthen hands-on skills

🎯 Learn by doing
________________________________________________________________________________________________________________________________

📅 Challenge Overview

Duration: 21 Days

Projects: 21 end-to-end projects

Focus Areas: Machine Learning, Deep Learning, Generative AI

Daily Routine: Live class → implement project → push to GitHub
________________________________________________________________________________________________________________________________

🛠️ Tech Stack

Languages: Python

Core: NumPy, Pandas, Scikit-learn

Deep Learning: TensorFlow / PyTorch

GenAI: Hugging Face, OpenAI, LangChain (where applicable)

Other Tools: Streamlit, Flask, Jupyter
________________________________________________________________________________________________________________________________

🌟 Key Learnings (to be updated daily)

📌 Day 1 – Titanic Dataset 🛳️
- Covered: EDA, cleaning, feature engineering, correlation analysis  
- Key Learning: Choosing the right plots for the right analysis makes insights clearer.
  
📌 Day 2 – Netflix Dataset 🍿
- Covered: Data cleaning & transformation, time-series analysis, text data manipulation, geographical & rating analysis, feature engineering, advanced visualization
- Key Learning: EDA isn’t just about numbers — the real value comes from turning raw data into stories that highlight trends, shifts, and patterns clearly.

📌 Day 3 – House Price Prediction (Regression) 🏠
- Covered: Data import via Kaggle API, preprocessing, feature engineering, categorical encoding, Linear Regression vs. XGBoost, model evaluation.
- Key Learning: Regression demands careful handling of targets & features, while the Kaggle API made workflows cleaner and reproducible.

📌 Day 4 – Sentiment Analysis (NLP) 💬
- Covered: Text preprocessing (stopwords, stemming/lemmatization), vectorization (BoW & TF-IDF), ML models for sentiment classification, evaluation (precision/recall/F1).
- Key Learning: In NLP, strong preprocessing + representation often matter more than the choice of model.

📌 Day 5 – Customer Segmentation (Clustering) 🛍️
- Covered: 2D/3D EDA, k-means (income & age-based), optimal k via Elbow Method, hierarchical clustering validation, persona creation for marketing.
- Key Learning: Clustering depends heavily on feature choice + validation — the right setup makes results business-ready.

📌 Day 6 – Predicting Future Store Sales (Time Series) 🏪📈
- Covered: Time series decomposition, stationarity testing (ADF), log transform & differencing, ACF/PACF for parameter selection, ARIMA baseline, SARIMA for trend + seasonality, RMSE evaluation.
- Key Learning: Stationarity is the backbone of forecasting — SARIMA proved far superior by modeling seasonal patterns effectively.

📌 Day 7 – Customer Churn Prediction (Feature Engineering) 📡
- Covered: Advanced data cleaning, feature creation (binning, combining, simplifying), preprocessing pipelines with ColumnTransformer, model comparison to measure lift from engineered features.
- Key Learning: Strong features + careful selection > complex models. The right engineering brings models closer to practical use.

✨ Closing Note – Machine Learning Phase Complete (Projects 1–7) ✨

Over the past 7 projects, I explored the full ML workflow — from EDA, preprocessing, regression, clustering, time series forecasting, to feature engineering & model optimization. Each project reinforced that ML is not just about algorithms but about data understanding + transformation.

📌 Day 8 – Vision AI Fundamentals: Digit Recognizer 🔢🤖
- Covered: Preprocessing (normalization, reshaping, encoding), ANN → Basic CNN → Deeper CNN (with batch norm & dropout), early stopping & checkpoints, evaluation (accuracy, loss, confusion matrices), and prediction analysis.
- Key Learning: Deep learning success comes from solid preprocessing + smart architectures, not just making models bigger.

📌 Day 9 – Advanced Vision AI: Transfer Learning ⚡🖼️
- Covered: Applied transfer learning with ResNet50, VGG16, MobileNetV2 on CIFAR-100. Preprocessed data per model requirements, added custom classification layers, trained with early stopping + checkpoints, fine-tuned top layers, and compared performance.
- Key Learning: Transfer learning accelerates convergence and improves performance by reusing powerful pre-trained feature extractors instead of starting from scratch.

📌 Day 10 – Creative AI: Neural Style Transfer 🎨🤖
- Covered: Used GANs to generate faces from random latent vectors, explored latent space with a gender direction vector, and visualized smooth male ↔ female transitions.
- Key Learning: GAN latent space = a creative playground where AI can morph features, blend styles, and even prove transformations scientifically.

📌 Day 11 – Hugging Face Pipelines 🛠️🤖
- Covered: Explored Hugging Face pipelines for NLP (sentiment, summarization, QA, NER, text generation, translation, zero-shot) and Vision (classification, detection, segmentation, captioning).
- Key Learning: Pipelines make AI tasks effortless in one line, while diffusion models highlight the creative potential of generative AI.

📌 Day 12 – Real-World CV: YOLOv8 & U-Net 👁️📷
- Covered: Explored YOLOv8 for real-time object detection and U-Net for tasks beyond segmentation (upscaling, colorization, face sharpening).
- Key Learning: YOLOv8 shows speed in detection, while U-Net proves its versatility in restoration & enhancement — powerful pair for real-world CV.

📌 Day 13 – Next-Gen Forecasting: Stock Price Prediction 📈🧠
- Covered: Used sliding windows (30–250 days) with OHLC features to model NIFTY 50 stock prices.
- Models: ML (LinearRegression, Ridge, Lasso, RF, XGBoost, LightGBM, SVR, KNN) vs DL (RNN, LSTM, GRU, Bi-LSTM).
- Key Learning: ML offers baselines, but DL captures sequential dependencies — together they give stronger financial insights.

📌 Day 14 – Custom GPT: Story Generator Text Model
- Covered: TinyStories dataset, GPT-2 style architecture with Transformer model & tokenizer, training with PyTorch + AdamW + checkpoints, prompt filtering for Python coding.
- Key Learning: Even small LLMs can generate coherent and creative text when built with control mechanisms and reproducible practices.

✨ Closing Note – Deep Learning Phase Complete (Projects 8–14) ✨

From vision to language, generative creativity to real-world applications — these 7 projects showed me the true breadth of Deep Learning. Each step proved that progress comes not from stacking layers blindly, but from combining the right preprocessing, architectures, transfer learning, and control mechanisms. Deep Learning isn’t just about bigger models — it’s about smarter design, leveraging transfer learning, harnessing creativity, and aligning models with real-world needs.

📌 Day 15 – Text-to-SQL Generator 💬🗄️
- Covered: Natural Language Understanding for intent detection, entity-schema mapping, SQL generation from English prompts, and query execution with readable output.
- Key Learning: Text-to-SQL isn’t just about syntax—it’s about understanding user intent & schema to make databases accessible to non-technical users.

📌 Day 16 – Intelligent Document Automation: Smart OCR Bot 📄🤖
- Covered: Document ingestion (PDFs, scans, images), preprocessing for clarity, ML-based document classification, NLP + CV for data extraction, human-in-the-loop validation, and integration with ERP/CRM systems.
- Key Learning: Intelligent Document Processing goes beyond OCR — it automates comprehension of unstructured documents, boosting efficiency and enabling digital transformation across industries.

📌 Day 17 – Intelligent Internet Search Engine 🌐🤖
- Covered: Used Crawl4AI for LLM-powered web scraping with concurrent sessions, proxy rotation, retries, and clean markdown output. Explored faster crawling (6x) via heuristics, Docker deployment, and API-free workflows.
- Key Learning: Web scraping is evolving into intelligent crawling — powering GenAI agents and scalable RAG pipelines.

📌 Day 18 – RAG Chatbot 🤖📚
- Covered: Combined Information Retrieval (document embeddings + vector DB) with Natural Language Generation (LLM responses) to build a Retrieval-Augmented Generation (RAG) chatbot.
- Key Learning: RAG bridges raw data and intelligent dialogue — enabling smarter chatbots, enterprise search, and scalable GenAI pipelines.

📌 Day 19 – Autonomous Market Analyst 📊🤖
- Covered: Built LLM-powered AI agents to automate deep market research — handling data collection, summarization, and structured insight generation.
- Key Learning: AI agents can act as autonomous researchers, scaling repetitive analysis and empowering faster decision-making.
________________________________________________________________________________________________________________________________

📌 Notes

This repo isn’t just about code dumps — I’ll also be documenting learnings, challenges, and takeaways for each project.

Discipline > Perfection ✨
________________________________________________________________________________________________________________________________

🔗 Connect

Follow my journey here:

💼 LinkedIn: https://www.linkedin.com/in/mannatjain14/

📂 GeeksforGeeks Challenge: https://github.com/Mannat-jain/GFG160 and https://github.com/Mannat-jain/21-Days-21-Projects
