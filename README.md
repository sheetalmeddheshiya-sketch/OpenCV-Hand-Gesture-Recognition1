# OpenCV + ML Hand Gesture Recognition

यह project `dataset/leapGestRecog` dataset से hand gestures train करता है और webcam से real-time prediction दिखाता है।

## Gestures

Dataset की class folders (`01_palm`, `02_l`, `03_fist`, `04_fist_moved`, `05_thumb`, `06_index`, `07_ok`, `08_palm_moved`, `09_c`, `10_down`) अपने-आप labels के रूप में पढ़ी जाती हैं।

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Train

Default training हर class से 500 images लेता है, इसलिए पहली run जल्दी पूरी होती है:

```powershell
python train_model.py
```

पूरे dataset पर train करने के लिए:

```powershell
python train_model.py --max-per-class 0
```

Model `models/gesture_model.joblib` और validation metadata `models/gesture_model.json` में save होंगे।

## Run webcam app

```powershell
python app.py
```

अगर दूसरा camera इस्तेमाल करना हो:

```powershell
python app.py --camera 1
```

Camera के बीच में दिखने वाले square के अंदर हाथ रखें। `Q` या `Esc` दबाकर app बंद करें।
Windows पर camera के लिए `app.py` DirectShow backend इस्तेमाल करता है, जो कई webcams में default backend से अधिक reliable है।

## Run dashboard

```powershell
streamlit run dashboard.py
```

Browser में खुलने वाले dashboard से model metrics देख सकते हैं, PNG/JPG hand image upload कर सकते हैं और dashboard के अंदर webcam capture से prediction चला सकते हैं। अब live gesture testing के लिए `app.py` अलग terminal में चलाने की जरूरत नहीं है।

## Deploy on Streamlit Community Cloud

1. इस project को GitHub repository में push करें।
2. Streamlit Cloud में **New app** चुनकर repository और `dashboard.py` को main file चुनें।
3. Python version `3.10` या `3.11` रखें।
4. `requirements.txt`, `dashboard.py`, `gesture_utils.py` और `models/gesture_model.joblib` repository में मौजूद होने चाहिए।
5. Deploy के बाद browser camera permission allow करें। Cloud पर continuous OpenCV webcam window (`app.py`) नहीं चलती; dashboard का `st.camera_input` browser camera capture के लिए है।

Deployment के लिए `requirements.txt` जानबूझकर minimal रखा गया है। `opencv-python-headless` cloud server के लिए इस्तेमाल होता है क्योंकि वहां desktop OpenCV window उपलब्ध नहीं होती।

## Approach

OpenCV preprocessing image को grayscale, resize, Otsu threshold और morphology से silhouette में बदलती है। HOG features को `StandardScaler` के बाद `LinearSVC` classifier में train किया जाता है। यही preprocessing training और webcam inference दोनों में reuse होती है।
