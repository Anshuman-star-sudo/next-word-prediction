# Next-Word Predictor using LSTM and Embeddings

A Deep Learning-powered next-word recommendation engine built using long short-term memory (LSTM) networks and word embeddings. The model captures long-range linguistic context and semantic relationships to dynamically forecast subsequent tokens in a sequence. 

The accompanying web application incorporates a unique interactive interface that pairs a robust backend with fluid, physics-inspired UI mechanics.

---

## Live Deployment

The application is fully operational and deployed on the cloud. You can interact with the live next-word prediction engine here:

* **Deployed Application Link:** https://next-word-prediction-hyxjnw87sus9sgxdwbeyww.streamlit.app/
* **Cloud Infrastructure:** Streamlit Community Cloud (Optimized for low-friction, cloud-native reactivity)

---

## Features

* **Sequential Deep Learning:** Built using an LSTM architecture to learn contextual, chronological dependencies across large text corpora.
* **Dense Semantic Representation:** Implements an Embedding layer to map vocabulary tokens into a continuous vector space, capturing nuanced word similarities.
* **Advanced Text Generation Control:** Decouples raw outputs using optimized decoding strategies, supporting Top-K and Top-P (Nucleus) sampling to mitigate repetitive loops.
* **Production-Ready Interface:** Packaged with a clean web interface optimized for rapid local or cloud-based interaction.

---

## Interactive UI: Google Antigravity Mechanics

The user interface implemented in `app.py` is deliberately styled and structurally inspired by the classic Google Antigravity environment.

* **Design Concept:** Much like the iconic interactive experience where rigid page elements break away from traditional structural grids, the layout is lightweight and entirely unburdened by conventional corporate design constraints.
* **User Experience:** It combines the mathematical precision of the text generation engine with a fluid, physics-defying front-end experience that offers a distinct and memorable user interaction.

---

## Model Performance

The predictive model was trained using an iterative categorical cross-entropy loss function on a structured textual corpus.

* **Target Training Accuracy:** 86%
* **Inference Latency:** Approximately 60ms to 90ms per token step.
* **Vocabulary Size:** 10,410 unique tokens.

---

## Architecture Stack

* **Language:** Python 3.12
* **Deep Learning Framework:** TensorFlow / Keras
* **Vector Optimization:** NumPy
* **Deployment & Interface Environment:** Streamlit Engine

---

├── app.py                      # Main production web interface application script
├── predict-next-word.ipynb     # Research, architecture development, and training pipeline
├── requirements.txt           # Minimal, cloud-compatible third-party dependencies
├── vocab.json                 # Exported tokenizer vocabulary mappings
├── config.json                # Model hyperparameters and sequence settings
└── .gitignore                 # Extraneous files and heavy binary tracking blocks
