# TinyML for Gait Event Detection in Parkinson's
Freezing of Gait (FoG) is a common and disruptive motor symptom in patients with Parkinson’s disease (PD), often described as a temporary inability to move the feet forward despite the intention to walk. These episodes significantly impair mobility and increase the risk of falls. Accurate, real-time detection of FoG is essential for providing timely interventions and improving patient safety and independence.

Traditional clinical assessments or patient self-reports are limited in their ability to capture FoG episodes that occur unpredictably in daily life. To address this, Inertial Measurement Units (IMUs)—wearable sensors capable of capturing multi-axis acceleration—have gained popularity as a non-invasive and practical solution for continuous gait monitoring. Data collected from IMUs worn on the ankle, thigh, and trunk provides detailed insights into a patient's movement patterns.

This project leverages such IMU data and proposes a lightweight 1D Convolutional Neural Network (CNN) model to detect FoG events. The model is trained on a publicly available dataset released by Baechlin et al. (2010), which includes labeled recordings from 10 Parkinson’s patients performing gait-related tasks. A sliding window approach is used to segment the time-series data into model-friendly samples.

This work contributes to the development of efficient, on-device FoG detection systems that can function in real-time and outside clinical settings.

# Project Structure
`````

TinyFog-CNN/
├── Models/
│   ├── ONNX Model/
│   │   └── gait_model.onnx
│   ├── Optimized TensorFlow Lite Model/
│   │   └── gait_cnn_model_optimized.tflite
│   ├── Pytorch Model/
│   │   └── gait_fog_model_traced.pt
│   ├── TensorFlow Lite Model/
│   │   └── gait_cnn_model.tflite
│   └── TensorFlow Model/
│       └── gait_cnn_model.h5
│
├── csv_converted_files/
│   ├── Generalization Subjects/
│   │   └── [S09R01.csv, S10R01.csv]
│   ├── Train_Test_Subjects/
│   │   └── [S01R01.csv to S08R01.csv]
│   ├── csv_converted_all/
│   │   └── [S01R01.csv to S10R01.csv]
│   └── Xy_train_combined_1-8.csv
│
├── dataset_fog_release/
│   ├── dataset/
│   │   └── [S01R01.txt to S10R01.txt]
│   └── doc/
│       ├── baechlin_TITB_2010.pdf
│       ├── documentation.html
│       └── img/
│           ├── annotation.jpg
│           ├── device.jpg
│           └── sketchOfPaths_landscape.png
│
├── notebooks/
│   ├── TinyML_for_Gait_Event_Detection_in_Parkinson's_main.ipynb
│   └── text_to_csv_conversion.ipynb
│
├── results/                # [To store plots, confusion matrices, metrics]
├── scripts/                # [Optional: Python scripts for training/inference]
├── LICENSE                 # MIT License
├── README.md               # Project overview
├── requirements.txt        # Python dependencies
└── .gitignore              # Ignore models, cache, etc.

`````

# Dataset Description
This project uses the publicly available Freezing of Gait (FoG) dataset, originally developed to support research on wearable assistance for Parkinson’s disease patients experiencing gait disturbances.

###  Dataset Access

This project uses the publicly available **Daphnet Freezing of Gait Dataset**, accessible through the UCI Machine Learning Repository:

> 🔗 [Daphnet Freezing of Gait Dataset – UCI ML Repository](https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait)

The dataset was introduced in the study:

> **Baechlin et al.**, _"Wearable Assistant for Parkinson’s Disease Patients With the Freezing of Gait Symptom,"_ IEEE Transactions on Information Technology in Biomedicine, 2010.

📎 A preprint of this publication is included in the repository under `dataset_fog_release/doc/`.

---

###  Data Characteristics

Each file (e.g., `S01R01.txt`) contains time-series IMU signals recorded at **64 Hz**, gathered from three body locations:

- **Ankle**  
- **Thigh**  
- **Trunk**  

Each IMU sensor provides tri-axial acceleration, resulting in a total of **9 features** per time step.

---

###  File Format Overview

| Index | Feature Name     | Description                                 |
|-------|------------------|---------------------------------------------|
| 1     | `Time`           | Timestamp in seconds                        |
| 2–4   | `ankle_x/y/z`    | 3-axis ankle acceleration                   |
| 5–7   | `thigh_x/y/z`    | 3-axis thigh acceleration                   |
| 8–10  | `trunk_x/y/z`    | 3-axis trunk acceleration                   |
| 11    | `label`          | 1 = Freezing, 0 = No Freezing               |
| 12    | `task`           | Type of activity (e.g., Walk, Turn, Stand)  |

FoG annotations were manually labeled by clinical experts based on synchronized video recordings.

---

###  Data Split Strategy

To ensure model robustness and assess cross-subject generalization:

- **Training & Validation Subjects:**  
  `S01` through `S08`

- **Generalization Subjects (held out):**  
  `S09`, `S10`

These generalization subjects are used to simulate real-world deployment on unseen patients.

##  text_to_csv_conversion.ipynb

This notebook converts the raw `.txt` files from the Daphnet FoG dataset into structured `.csv` files. It assigns meaningful column names, organizes the data by subject, and saves cleaned outputs for training and evaluation. This step enables consistent preprocessing and segmentation across all models.

##  Data Preprocessing & Pipeline

To prepare the raw IMU sensor data for training, the following preprocessing steps were applied:

---

###  1. Sliding Window Segmentation

The time-series data from each subject file (`.csv` format) was segmented using a **sliding window** approach:

- **Window Size:** 128 time steps (2 seconds of data at 64Hz)
- **Stride:** 64 time steps (50% overlap)

Each window was treated as one training sample. The **label** for each window was assigned as:

- `1` if **any** time step in the window had FoG (`label=1`)
- `0` if **all** time steps in the window had `label=0`

---

###  2. Normalization

All IMU features were **z-score normalized**:

- Mean and standard deviation were computed **across all training samples**
- The same values were used to normalize both training and generalization subjects

This ensures consistent feature scaling across subjects and time periods.

---

###  3. Dataset Preparation

- The segmented and normalized data was reshaped to **(samples, 128, 9)** to fit the expected CNN input format
- Corresponding binary labels (`y`) were stored for classification

>  The final training dataset is saved as `Xy_train_combined_1-8.csv` for reuse and reproducibility.

---

###  Generalization Subject Evaluation

Two subject recordings (`S09`, `S10`) were held out during training and used as **generalization subjects**. Their data was:

- Preprocessed using the **same pipeline**
- Evaluated post-training to test model **generalization** capability

This setup simulates deployment on **unseen patients** in real-world use cases.

##  Results

The CNN-based model was trained on 8 subjects using a stratified 85–15% train-validation split. Hyperparameter tuning was performed over a grid of filter sizes, kernel sizes, dropout rates, and learning rates.

###  Best Validation Performance

- **F1 Score:** 0.9029
- **Accuracy:** 90.31%
- **Precision:** 89.44%
- **Recall:** 91.12%
- **AUC:** 0.9614

###  Generalization Evaluation

To assess cross-subject generalization, the final model was tested on subject `S10` (completely unseen during training). The results were as follows:

- **Generalization Accuracy:** 77.61%
- **Generalization F1 Score:** 0.764
- **Confusion Matrix** and **ROC/PR Curves** included in the notebook.

This validates the model’s robustness for real-world applications where data from new patients is encountered.

---

##  Model Variants & Exported Formats

The final model architecture is a lightweight 1D CNN designed for time-series classification with TinyML deployment in mind. To support deployment in real-world wearable devices, the model was exported in multiple formats including PyTorch, ONNX, TensorFlow, and TensorFlow Lite (TFLite and optimized TFLite) for TinyML compatibility.

###  Model Architecture

- 2 × Conv1D + BatchNorm + MaxPooling
- Global Average Pooling
- Dropout (0.3)
- Dense(64) + Output sigmoid layer

###  Exported Model Formats

| Format                   | File                                             | Purpose                                 |
|--------------------------|--------------------------------------------------|------------------------------------------|
| **PyTorch**              | `gait_fog_model_traced.pt`                       | TorchScript for PyTorch inference        |
| **ONNX**                 | `gait_model.onnx`                                | Interoperability across platforms        |
| **TensorFlow (.h5)**     | `gait_cnn_model.h5`                              | Keras model checkpoint                   |
| **TensorFlow Lite**      | `gait_cnn_model.tflite`                          | Lightweight mobile deployment            |
| **TFLite (Optimized)**   | `gait_cnn_model_optimized.tflite`               | TinyML edge deployment                   |

>  All models are located in the `Models/` directory of this repository.


## Installation and Usage

1. Clone the repository:
   
  ```bash
  git clone https://github.com/raadsr15/TinyML-for-Gait-Event-Detection-in-Parkinson's.git
  cd TinyML-for-Gait-Event-Detection-in-Parkinson's
  ```
2. Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```
3. Convert raw .txt data to .csv:

  ```bash
  jupyter notebook text_to_csv_conversion.ipynb
  ```
4. Train the model and export formats
   
```bash
jupyter notebook TinyML_for_Gait_Event_Detection_in_Parkinson's_main.ipynb
```





