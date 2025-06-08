# TinyML for Gait Event Detection in Parkinson's
TinyFog-CNN is a lightweight 1D Convolutional Neural Network (CNN) model designed for real-time detection of Freezing of Gait (FoG) in Parkinson’s Disease patients using ankle and hip worn IMU sensors. Built for TinyML deployment with TensorFlow Lite.

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
