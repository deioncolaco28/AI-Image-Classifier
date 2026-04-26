# AI Image Classifier 📸

AI Image Classifier is an advanced, high-performance vision intelligence platform that uses a powerful **Weighted Neural Ensemble** (EfficientNet-V2-Large + Swin-Transformer-B) to classify objects with world-class accuracy. Built with a robust FastAPI backend and a stunning, interactive **Aurora Glass** frontend.

By combining the local feature-extraction power of Convolutional Neural Networks with the global relationship-mapping of Vision Transformers, AI Image Classifier achieves significantly higher reliability than standard single-model systems.

## 🚀 Key Features
**Dual-Brain Neural Ensemble:** Combines the state-of-the-art `EfficientNet-V2-L` (CNN Expert) with `Swin-Transformer-B` (Global Expert). This ensemble approach cancels out individual model errors, resulting in ~87% top-1 accuracy.
**High-Speed Batch Processing:** Classify up to **30 images** simultaneously with optimized asynchronous inference and multi-threaded data handling.
**Aurora Glass Interactive UI:** Features a completely custom HTML5 Canvas background with a **Magnetic Neural Web** that reacts to your mouse, paired with beautiful **Aurora Blobs** and modern glassmorphism styling.
**Numerical Precision Data View:** Provides clean, data-focused classification results showing exact confidence percentages and sub-category tagging (e.g., Animal → Dog → Husky) for professional-grade analysis.
**Comprehensive Pro Test Suite:** Comes pre-loaded with **30+ verified, high-definition images** across diverse categories like Animals, Food, Places, and Vehicles for immediate validation.

## 🛠️ Technology Stack
**Backend:** Python 3, FastAPI, Uvicorn (High-performance ASGI)
**Machine Learning Models:**
- PyTorch & Torchvision
- EfficientNet-V2-Large (60% weight)
- Swin-Transformer-B (40% weight)
**Data Handling:** PIL (Pillow), NumPy, Pathlib
**Frontend:** HTML5 Canvas, Vanilla CSS3 (Custom Glassmorphism), JavaScript (ES6+)

## 📁 Project Structure
AI Image Classifier/
├── app.py                 # Core FastAPI application & Neural Ensemble logic
├── requirements.txt       # Python dependency list (Torch 2.5.1, FastAPI 0.110)
├── index.html             # Stunning Aurora Glass dashboard & interactive UI
├── README.md              # Comprehensive project documentation
├── uploads/               # Temporary storage (Auto-cleared after inference)
└── test_images/           # 30+ High-definition verified images for testing
    ├── lion_animal.jpg
    ├── pizza_food.jpg
    └── ... (30+ diverse categories)

## ⚙️ Setup and Installation
Clone the Repository and navigate to the project root.
Install Dependencies:
Make sure you have Python 3.9+ installed. Use pip to install the required high-performance libraries.
```bash
pip install -r requirements.txt
```

## ▶️ Running the Application
Start the local FastAPI server:
```bash
python app.py
```
Open your web browser and navigate to:
```
http://127.0.0.1:8000/
```
**Note on Selection:** Drag and drop up to 30 images from the `test_images` folder and click **Select Classification** to see the dual-brain AI in action!

## 🧠 How it Works
**Convolutional Expert (EfficientNet-V2-L):** The model focuses on the fine textures, colors, and local features of the image (e.g., the fur texture of a dog or the seeds on a strawberry).
**Transformer Expert (Swin-B):** The Vision Transformer uses a shift-window mechanism to understand the global structure and shape of objects, ensuring the AI isn't "tricked" by complex backgrounds.
**Weighted Ensemble:** The final classification is a weighted average (60/40) of both models. This provides a robust "double-check" system that significantly reduces misidentifications.

## ⚠️ Disclaimer
This application is designed for educational and informational purposes. While the Neural Ensemble is highly accurate, AI classification can occasionally produce errors depending on image quality, lighting, and occlusions. Always verify critical results manually.
