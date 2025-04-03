# 🗑️ Smart Trash Classifier 
*Raspberry Pi-based waste sorting system using TensorFlow Lite*

![Demo GIF](docs/demo.gif) *(Replace with actual demo video/image)*

## 🌟 Features
- Real-time trash classification (plastic, paper, metal, glass, etc.)
- Web interface via Flask
- Recyclable/non-recyclable determination
- Confidence threshold filtering
- Optimized for Raspberry Pi

## 🛠️ Hardware Requirements
| Component | Specification |
|-----------|---------------|
| Raspberry Pi | 3B+/4/5 |
| Camera | Pi Camera v2 or USB webcam |
| RAM | Minimum 2GB |
| Storage | 8GB+ SD card |

## 🚀 Quick Start

### Installation
```bash
# Clone repo
git clone https://github.com/your-username/trash-classifier.git
cd trash-classifier

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
