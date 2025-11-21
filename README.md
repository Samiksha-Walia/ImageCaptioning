
# 🧠 Intelligent Image Captioning System – BLIP + Fuzzy Logic + Dragonfly Optimization

A powerful **AI-based image captioning system** integrating **BLIP (Bootstrapping Language-Image Pre-training)** with **fuzzy logic evaluation**, **BLEU & BERTScore metrics**, and a **Dragonfly Optimization Algorithm** to generate *high-quality, optimized captions* for both single images and image sequences.

This project uses **Transformer-based vision-language models**, **linguistic evaluation**, and **nature-inspired optimization** to push captioning quality beyond standard deterministic pipelines.

> 🎓 **Research Project**: Developed as an advanced AI system integrating ML, NLP, fuzzy inference, and evolutionary optimization—ideal for research, publications, and multimedia analytics.



## 🚀 Key Features

* 🖼️ **Image Captioning using BLIP + Swin Transformer**
* 🤖 **Parameter optimization** using the **Dragonfly Algorithm**
* 🧮 **Real evaluation metrics**: BLEU & BERTScore
* fuzzy **Fuzzy Logic Quality System** for robust scoring
* 📁 Sequential captioning for entire **image folders**
* 🔧 Adjustable decoding parameters (beams, temperature, top-k/p)
* ⚙️ **Gradio UI** for interactive testing
* 📊 Automatic comparison of reference vs generated caption
* 🔍 NLP tokenization + linguistic scoring via NLTK
* ⚡ GPU-accelerated Transformer inference (PyTorch)
* 🔣 Fully automated pipeline from caption → evaluate → optimize → caption


## 🛠️ Tech Stack

| Component          | Technology / Libraries                           |
| ------------------ | ------------------------------------------------ |
| **Model**          | BLIP Image Captioning (HuggingFace Transformers) |
| **Optimization**   | Dragonfly Algorithm                              |
| **Evaluation**     | BLEU, BERTScore, Fuzzy Logic                     |
| **NLP**            | NLTK                                             |
| **UI**             | Gradio                                           |
| **Backend Engine** | PyTorch                                          |
| **Math / Logic**   | NumPy, scikit-fuzzy                              |



## ⚙️ How It Works

### 1️⃣ **Load BLIP Model**

* Pretrained on large-scale vision-language datasets
* Generates initial captions given an image

### 2️⃣ **Generate Caption**

Uses configurable decoding parameters:

* `num_beams`
* `max_length`
* `temperature`
* `top_k`, `top_p`

### 3️⃣ **Evaluate Caption**

Two complementary metrics:

* **BLEU Score** – syntactic similarity
* **BERTScore** – semantic similarity

### 4️⃣ **Fuzzy Logic Quality Estimation**

Inputs:

* BLEU
* Similarity (BERTScore)

Output:

* **Quality score** (0–1)

### 5️⃣ **Dragonfly Optimization**

Repeatedly:

1. Randomly sample decoding parameters
2. Generate caption
3. Evaluate & fuzzy-score
4. Keep best parameters

### 6️⃣ **Gradio App**

* **Single Image Mode**
* **Folder Sequence Mode**
* Fully interactive captions + evaluations



## 📂 Directory Support

Supports:

* Single images
* Entire folders (for video frames, datasets, surveillance data, drone images, etc.)



## 🧪 Testing Workflow

1. Upload an image
2. Provide a *reference caption*
3. System optimizes BLIP parameters
4. Generates the best caption
5. Shows:

   * Optimized params
   * BLEU score
   * BERTScore
   * Fuzzy Quality (%)

For folder mode:

* Upload a zipped/unzipped folder
* Captions are generated for each image sequentially



## 📦 Installation & Setup

### Prerequisites

* Python 3.8+
* PyTorch (CPU/GPU)
* Transformers
* scikit-fuzzy
* NLTK
* Gradio

### Install Dependencies

```bash
pip install torch transformers gradio scikit-fuzzy nltk bert-score pillow
```

### Run the App

```bash
python app.py
```

Gradio UI will appear at:

```
http://localhost:7860
```



## 🔍 Example Output

For each image, you get:

* 🖼️ **Generated Caption**
* 🔧 **Optimal Parameters**
* 📊 **BLEU Score**
* 🤖 **BERTScore**
* 🧠 **Fuzzy Quality Score**



## 🔮 Future Enhancements

* 🧬 Genetic Algorithm + PSO comparison
* 📈 Visualization dashboards for evaluation metrics
* 📽️ Video captioning using frame batching
* 🌐 API deployment (FastAPI / Flask)
* 🛠️ ONNX Runtime acceleration




## 👩‍💻 Author

**Samiksha Walia**
[GitHub](https://github.com/Samiksha-Walia) • [LinkedIn](https://linkedin.com/in/samiksha-walia)



## ⭐ Show Your Support

If this project supports your research or learning, ⭐ the repository and share your experience!

> *A fusion of vision-language models, NLP quality metrics, fuzzy intelligence, and swarm optimization—designed for next-gen AI captioning systems.*

