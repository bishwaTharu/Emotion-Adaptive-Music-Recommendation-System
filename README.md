# 🎶 Words-to-Waves: Emotion-Adaptive Music Recommendation System

Words-to-Waves is an innovative, **emotion-adaptive music recommendation system** designed to personalize song suggestions based on a user's real-time emotional state, as expressed through natural language text.

---

## ✨ Key Features

* **Real-time Emotion Inference:** Infers a user's continuous emotional state in the **Valence–Arousal (VA) space** directly from free-form natural language text. 
* **Transformer-Based Language Model:** Utilizes a transformer for deep emotional representation and generalization across diverse linguistic expressions.
* **Wide-and-Deep Learning Architecture:** Combines a **deep component** (transformer sentence embeddings) for generalization and a **wide component** (engineered features/historical data) for memorizing structured emotional patterns.
* **Efficient Fine-Tuning (LoRA):** Adapts the transformer using **Low-Rank Adaptation (LoRA)** to significantly reduce the number of trainable parameters, enabling efficient fine-tuning on limited emotion-annotated data.
* **Personalized Recommendation Engine:** Generates suggestions by combining:
    * Emotional similarity in the Valence–Arousal space.
    * User $\times$ Emotion memory tables derived from listening history.
    * Emotion $\times$ Artist memory tables derived from listening history.

---

## 🛠️ Implementation Details

### **Model Architecture**

The core of the system is the **Wide-and-Deep learning model**:

* **Deep Component:** Leverages pre-trained transformer embeddings, fine-tuned with LoRA, to generate continuous emotional vectors.
* **Wide Component:** Uses traditional features (e.g., genre tags, time of day, aggregated listening counts) to capture explicit and repeatable patterns.

### **Technology Stack**

* **Framework:** Implemented in **PyTorch** for flexibility and performance.
* **Training Optimization:** Employs **mixed-precision training** for faster, memory-efficient model updates.
* **Experiment Management:** Utilizes **MLflow** for robust tracking of experiments, parameters, metrics, and models, ensuring reproducibility.

### **Outputs**

The result is a **scalable, context-aware recommender system** capable of delivering emotionally relevant music recommendations in real time, moving beyond static mood tags and purely historical preferences.

---

## 🚀 Get Started (Conceptual)

1.  **Input:** A user provides a free-form text statement about their feelings (e.g., "I just had a great day at work!").
2.  **Inference:** The transformer model processes the text and maps it to a point in the Valence–Arousal space.
3.  **Recommendation:** The engine queries memory tables and finds songs/artists with a high emotional and personal relevance score near that VA point.
4.  **Output:** A personalized list of emotionally resonant song recommendations is delivered.