# 16 - Deployment & Runtime Guide

## 1. Environment Options

### Option A: Windows + WSL2 Ubuntu (Recommended for GPU)
1. Open PowerShell and verify Ubuntu distribution:
   ```powershell
   wsl -d Ubuntu
   ```
2. Activate your Python GPU virtual environment:
   ```bash
   source ~/chronus-gpu/bin/activate
   ```
3. Navigate to the project root:
   ```bash
   cd /mnt/d/PROJECT
   ```
4. Run the Streamlit web interface:
   ```bash
   python3 -m streamlit run main.py --server.address 0.0.0.0 --server.port 8501
   ```
   Or double-click `run_web_ui.bat` from Windows.

### Option B: Native Windows (CPU / Direct)
1. Ensure dependencies from `requirements.txt` are installed in your active environment:
   ```powershell
   pip install -r requirements.txt
   ```
2. Start the application:
   ```powershell
   streamlit run main.py
   ```

## 2. Model Training Procedures
If you ever wish to re-train the models from scratch:
- Leaf Disease Model:
  ```powershell
  python src/train_image_model.py
  ```
- Soil Suitability Model:
  ```powershell
  python src/train_soil_model.py
  ```
- Or execute `run_training.bat`.
