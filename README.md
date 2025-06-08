# TinyML for Gait Event Detection in Parkinson's
Freezing of Gait (FoG) is a common and disruptive motor symptom in patients with Parkinson’s disease (PD), often described as a temporary inability to move the feet forward despite the intention to walk. These episodes significantly impair mobility and increase the risk of falls. Accurate, real-time detection of FoG is essential for providing timely interventions and improving patient safety and independence.

Traditional clinical assessments or patient self-reports are limited in their ability to capture FoG episodes that occur unpredictably in daily life. To address this, Inertial Measurement Units (IMUs)—wearable sensors capable of capturing multi-axis acceleration—have gained popularity as a non-invasive and practical solution for continuous gait monitoring. Data collected from IMUs worn on the ankle, thigh, and trunk provides detailed insights into a patient's movement patterns.

This project leverages such IMU data and proposes a lightweight 1D Convolutional Neural Network (CNN) model to detect FoG events. The model is trained on a publicly available dataset released by Baechlin et al. (2010), which includes labeled recordings from 10 Parkinson’s patients performing gait-related tasks. A sliding window approach is used to segment the time-series data into model-friendly samples.

The final model achieved an F1-score of 0.9029 on the validation set and 77.61% accuracy on an unseen generalization subject—demonstrating its robustness across subjects. To support deployment in real-world wearable devices, the model was exported in multiple formats including PyTorch, ONNX, TensorFlow, and TensorFlow Lite (TFLite and optimized TFLite) for TinyML compatibility.

This work contributes to the development of efficient, on-device FoG detection systems that can function in real-time and outside clinical settings.

`````
## Project Structure
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

## Source
The dataset was introduced in the study:

Baechlin et al., "Wearable Assistant for Parkinson’s Disease Patients With the Freezing of Gait Symptom," IEEE TITB, 2010.
Link to paper (preprint included in repo)

The dataset is included in the dataset_fog_release/ directory of this repository.


