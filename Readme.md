# Overview

GrainPalette is an AI-powered web application that uses deep learning to classify rice grain types from uploaded images. The system leverages transfer learning with MobileNetV2 to identify five different rice varieties (Arborio, Basmati, Ipsala, Jasmine, and Karacadag) and provides agricultural recommendations for each type.

# System Architecture

## Frontend Architecture
- **Framework**: Flask with Jinja2 templating
- **UI Framework**: Bootstrap 5 with dark theme
- **JavaScript**: Vanilla JS for file upload preview and form validation
- **Styling**: Custom CSS with CSS variables for theming

## Backend Architecture
- **Web Framework**: Flask (Python 3.11)
- **WSGI Server**: Gunicorn for production deployment
- **Model Framework**: TensorFlow with TensorFlow Hub for transfer learning
- **Image Processing**: OpenCV and PIL for image preprocessing
- **File Upload Handling**: Werkzeug utilities with secure filename handling

## Model Architecture
- **Base Model**: MobileNetV2 via TensorFlow Hub for transfer learning
- **Input Size**: 224x224 pixels (RGB images)
- **Output Classes**: 5 rice varieties with confidence scores
- **Preprocessing**: Automatic image resizing and normalization

# Key Components

## RiceClassifier (`rice_classifier.py`)
- Implements transfer learning using MobileNetV2
- Handles model loading and prediction logic
- Provides agricultural recommendations for each rice type
- Includes class mappings and confidence scoring

## Flask Application (`app.py`)
- Main web application with file upload endpoints
- Image preprocessing pipeline using OpenCV/PIL
- Error handling and validation for uploaded files
- Integration with the rice classification model

## Web Interface
- **Upload Page** (`templates/index.html`): File upload form with preview
- **Results Page** (`templates/result.html`): Classification results and recommendations
- **JavaScript** (`static/script.js`): Client-side validation and preview functionality
- **Styling** (`static/style.css`): Custom dark theme with hover effects

# Data Flow

1. **Image Upload**: User uploads rice grain image through web interface
2. **Validation**: Server validates file type (PNG, JPG, JPEG, GIF, BMP, WEBP) and size (16MB max)
3. **Preprocessing**: Image is resized to 224x224 pixels and normalized
4. **Classification**: MobileNetV2 model processes the image and returns predictions
5. **Results**: Top prediction with confidence score is displayed along with agricultural recommendations
6. **File Cleanup**: Uploaded files are stored temporarily in uploads directory

# External Dependencies

## Core Dependencies
- **TensorFlow**: Deep learning framework for model inference
- **TensorFlow Hub**: Pre-trained MobileNetV2 model access
- **OpenCV**: Image processing and computer vision operations
- **PIL (Pillow)**: Alternative image processing library
- **Flask**: Web framework and templating
- **NumPy**: Numerical computing for array operations

## Development Dependencies
- **Gunicorn**: Production WSGI server
- **Werkzeug**: WSGI utilities and secure file handling
- **Flask-SQLAlchemy**: Database ORM (configured but not actively used)
- **psycopg2-binary**: PostgreSQL adapter (prepared for future database integration)

# Deployment Strategy

## Production Configuration
- **Platform**: Replit with autoscale deployment target
- **Web Server**: Gunicorn with process binding to 0.0.0.0:5000
- **Process Management**: Reuse port configuration for zero-downtime restarts
- **Development Mode**: Hot reload enabled for development

## System Requirements
- **Python Version**: 3.11+
- **Memory**: Sufficient for TensorFlow model loading
- **Storage**: Temporary file storage for image uploads
- **Network**: Internet access for TensorFlow Hub model downloads

## Environment Configuration
- **Session Secret**: Configurable via SESSION_SECRET environment variable
- **File Uploads**: 16MB maximum file size limit
- **Static Assets**: Served directly by Flask in development

# Changelog

```
Changelog:
- June 27, 2025. Complete project restructuring with magnificent design system
  - Built comprehensive training notebook (train.ipynb) with MobileNetV2 transfer learning
  - Created enhanced UI with glassmorphism effects, animated hero section, and drag-drop upload
  - Added About page with detailed project information and technology showcase
  - Implemented lightweight CNN model for demo functionality
  - Enhanced CSS with gradients, glow effects, and smooth animations
  - Added navigation between pages and responsive design
- June 26, 2025. Initial setup
```

# User Preferences

```
Preferred communication style: Simple, everyday language.
```