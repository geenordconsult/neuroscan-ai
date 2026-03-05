# NeuroScan AI

**AI-Powered Alzheimer's Disease MRI Classification System**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An web application for classifying Alzheimer's disease stages from structural brain MRI scans using deep learning. Features explainable AI with Grad-CAM++ visualizations and comprehensive clinical insights.

**Developed by:** Uday Islam & Kritanu Chattopadhyay
**Additional Contributor:** Michael Mabayomije Jinadu
---

## 🎯 Overview

NeuroScan AI is a research-grade medical imaging classification system that leverages state-of-the-art ConvNeXt architecture to classify Alzheimer's disease stages from MRI scans. The system provides:

- **Automated Classification**: Four-stage classification (No Impairment, Very Mild, Mild, Moderate)
- **Explainable AI**: Grad-CAM++ visualizations showing model attention regions
- **Clinical Insights**: Risk assessment and evidence-based recommendations
- **Production-Ready**: Dockerized, scalable, and cloud-deployable

## ✨ Features

- 🧠 **Deep Learning Model**: ConvNeXt-based architecture with TTA + Ensemble achieving 99.84% accuracy
- 🤗 **Hugging Face Integration**: Automatic model download from [udayislam/alzheimer-mri-convnext-classifier](https://huggingface.co/udayislam/alzheimer-mri-convnext-classifier)
- 🔍 **Explainable AI**: Grad-CAM++ heatmaps for model interpretability
- 📊 **Comprehensive Analysis**: Probability distributions and confidence scores
- 🏥 **Clinical Insights**: Risk assessment and clinical recommendations
- 🌐 **Modern Web Interface**: Responsive, professional UI/UX
- 🐳 **Docker Support**: Containerized for easy deployment
- ☁️ **Cloud Ready**: Deployable on AWS, GCP, Azure, Render, Heroku
- 🔒 **Production-Safe**: Non-blocking inference, proper error handling, security best practices

## 🏗️ Architecture

### Model Architecture
- **Base Model**: ConvNeXt-Base (timm)
- **Input Resolution**: 384×384 pixels
- **Classes**: 4 (Mild Impairment, Moderate Impairment, No Impairment, Very Mild Impairment)
- **Training**: 5-fold cross-validation with ensemble
- **Augmentation**: Horizontal flip, rotation, brightness/contrast adjustment

### Technology Stack
- **Backend**: Flask 3.0, Gunicorn
- **Deep Learning**: PyTorch 2.1, timm
- **Model Hub**: Hugging Face Hub
- **Image Processing**: OpenCV, Albumentations, PIL
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Explainability**: Grad-CAM++

## 📋 Prerequisites

- Python 3.11+
- pip or conda
- Docker (optional, for containerized deployment)
- CUDA-capable GPU (optional, for faster inference)

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   
   git clone https://github.com/yourusername/neuroscan-ai.git
   cd neuroscan-ai
   2. **Create virtual environment**
   
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   3. **Install dependencies**
 
   pip install --upgrade pip
   pip install -r requirements.txt
   4. **Model Setup**
   
   The trained model is automatically downloaded from Hugging Face Hub on first run:
   - **Repository**: [udayislam/alzheimer-mri-convnext-classifier](https://huggingface.co/udayislam/alzheimer-mri-convnext-classifier)
   - **Model File**: `best_model.pth` (352MB)
   - **Download Location**: `backend/models/`
   
   The model will be downloaded automatically when you first start the application. This is a one-time download that takes 30-60 seconds depending on your network speed.

5. **Configure environment variables** (optional)
   # Create .env file
   SECRET_KEY=your-secret-key-here
   MODEL_PATH=models/best_model.pth
   PORT=5000
   FLASK_ENV=development
   6. **Run the application**
   
   python app.py
      The application will be available at `http://localhost:5000`

### Docker Deployment

1. **Build the Docker image**
   docker build -t neuroscan-ai .
   2. **Run the container**
  
   docker run -d \
     -p 5000:5000 \
     -e SECRET_KEY=your-secret-key \
     -v $(pwd)/models:/app/models:ro \
     -v $(pwd)/instance:/app/instance \
     neuroscan-ai
   ### Docker Compose

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down## 

Download model from :  [udayislam/alzheimer-mri-convnext-classifier](https://huggingface.co/udayislam/alzheimer-mri-convnext-classifier)

📁 Project Structure

```
neuroscan-ai/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose configuration
├── Procfile              # Process file for cloud deployment
├── .dockerignore         # Docker ignore patterns
├── .gitignore            # Git ignore patterns
│
├── models/               # Model files
│   ├── best_model.pth   # Trained model weights
│   └── class_names.json # Class name mappings            Model Link : [udayislam/alzheimer-mri-convnext-classifier](https://huggingface.co/udayislam/alzheimer-mri-convnext-classifier)
│
├── utils/                # Utility modules
│   ├── model_loader.py   # Model loading utilities
│   ├── preprocessor.py   # Image preprocessing
│   └── grad_cam.py       # Grad-CAM++ implementation
│
├── templates/            # Jinja2 templates
│   ├── base.html        # Base template
│   ├── index.html       # Home page
│   ├── upload.html      # Upload page
│   ├── results.html     # Results page
│   └── ...
│
├── static/               # Static assets
│   ├── css/             # Stylesheets
│   ├── js/              # JavaScript files
│   └── images/          # Images and logos
│
└── instance/            # Instance-specific files
    ├── uploads/         # Uploaded images
    └── results/         # Generated results
```

## 🔌 API Endpoints

### Web Routes
- `GET /` - Home page
- `GET /upload` - Upload page
- `POST /upload` - Upload and analyze MRI image
- `GET /results/<result_id>` - View analysis results
- `GET /methodology` - Model methodology page
- `GET /disclaimer` - Disclaimer and limitations
- `GET /about` - About page

### API Endpoints
- `POST /api/predict` - Lightweight prediction API (JSON response)
- `GET /api/health` - Health check endpoint
- `GET /api/statistics` - Model performance statistics

### Example API Usage

```bash
# Predict endpoint
curl -X POST http://localhost:5000/api/predict \
  -F "file=@mri_scan.jpg"

# Health check
curl http://localhost:5000/api/health
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SECRET_KEY` | Flask secret key for sessions | `alzheimer-mri-classification-secret-key-2024` |
| `MODEL_PATH` | Path to model file | `models/best_model.pth` |
| `PORT` | Port to run application | `5000` |
| `FLASK_ENV` | Flask environment | `production` |
| `USE_CUDA` | Enable CUDA if available | `true` |

### Model Configuration

Edit `app.py` to modify:
- Image size (default: 384×384)
- Model architecture (ConvNeXt/DenseNet)
- Class names and descriptions
- Cache size limits

## 🧪 Testing

```bash
# Run health check
curl http://localhost:5000/api/health

# Test prediction API
curl -X POST http://localhost:5000/api/predict \
  -F "file=@test_mri.jpg" \
  -H "Content-Type: multipart/form-data"
```

## 🚢 Deployment

### Cloud Platforms

#### Render
1. Connect your GitHub repository
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `gunicorn app:app`
4. Add environment variables

#### Heroku
```bash
heroku create neuroscan-ai
heroku config:set SECRET_KEY=your-secret-key
git push heroku main
```

#### AWS/GCP/Azure
- Use Docker containers with ECS/GKE/AKS
- Configure load balancers and auto-scaling
- Set up persistent storage for uploads

### Production Checklist

- [ ] Set strong `SECRET_KEY` environment variable
- [ ] Configure proper CORS settings
- [ ] Set up SSL/TLS certificates
- [ ] Configure logging and monitoring
- [ ] Set up backup for model files
- [ ] Configure rate limiting
- [ ] Set up health checks and alerts
- [ ] Review and update security settings

## 📊 Model Performance

- Accuracy: 99.84%

- Precision: 99.85%

- Recall: 99.84%

- F1-Score: 99.84%

- AUC-ROC: 100.0%

### Per-Class Performance

| Class | Accuracy |
|-------|----------|
| Mild Impairment | 97.6% |
| Moderate Impairment | 98.8% |
| No Impairment | 98.4% |
| Very Mild Impairment | 97.1% |

## ⚠️ Important Disclaimers

- **Research Tool**: This system is intended for research and educational purposes only
- **Not Diagnostic**: Outputs are not intended for clinical diagnosis
- **No Medical Advice**: Results should not replace professional medical consultation
- **Data Privacy**: Ensure compliance with HIPAA/GDPR when handling medical data
- **Model Limitations**: Performance may vary with different MRI scanners and protocols

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Uday Islam** - *Initial work*
- **Kritanu Chattopadhyay** - *Initial work*

## 🙏 Acknowledgments

- ConvNeXt architecture by Meta AI Research
- Grad-CAM++ implementation based on original research
- Medical imaging community for datasets and feedback

## 📧 Contact

For questions, issues, or collaboration inquiries, please open an issue on GitHub.

---

**⚠️ Medical Disclaimer**: This software is provided for research purposes only. It is not intended for use in clinical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical advice.
```
