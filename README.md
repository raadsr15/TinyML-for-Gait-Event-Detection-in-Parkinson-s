# TinyML for Gait Event Detection in Parkinson's
TinyFog-CNN is a lightweight 1D Convolutional Neural Network (CNN) model designed for real-time detection of Freezing of Gait (FoG) in Parkinson’s Disease patients using ankle and hip worn IMU sensors. Built for TinyML deployment with TensorFlow Lite.


## Project Structure
├── Models
│   ├── ONNX Model
│   │   └── gait_model.onnx
│   ├── Optimized TensorFlow Lite Model
│   │   └── gait_cnn_model_optimized.tflite
│   ├── Pytorch Model
│   │   └── gait_fog_model_traced.pt
│   ├── TensorFlow Lite Model
│   │   └── gait_cnn_model.tflite
│   └── TensorFlow Model
│       └── gait_cnn_model.h5
├── TinyML_for_Gait_Event_Detection_in_Parkinson's_main.ipynb
├── csv_converted_files
│   ├── Generalization Subjects
│   │   ├── S09R01.csv
│   │   └── S10R01.csv
│   ├── Train_Test_Subjects
│   │   ├── S01R01.csv
│   │   ├── S01R02.csv
│   │   ├── S02R01.csv
│   │   ├── S02R02.csv
│   │   ├── S03R01.csv
│   │   ├── S03R02.csv
│   │   ├── S03R03.csv
│   │   ├── S04R01.csv
│   │   ├── S05R01.csv
│   │   ├── S05R02.csv
│   │   ├── S06R01.csv
│   │   ├── S06R02.csv
│   │   ├── S07R01.csv
│   │   ├── S07R02.csv
│   │   └── S08R01.csv
│   ├── Xy_train_combined_1-8.csv
│   └── csv_converted_all
│       ├── S01R01.csv
│       ├── S01R02.csv
│       ├── S02R01.csv
│       ├── S02R02.csv
│       ├── S03R01.csv
│       ├── S03R02.csv
│       ├── S03R03.csv
│       ├── S04R01.csv
│       ├── S05R01.csv
│       ├── S05R02.csv
│       ├── S06R01.csv
│       ├── S06R02.csv
│       ├── S07R01.csv
│       ├── S07R02.csv
│       ├── S08R01.csv
│       ├── S09R01.csv
│       └── S10R01.csv
├── dataset_fog_release
│   ├── README
│   ├── dataset
│   │   ├── S01R01.txt
│   │   ├── S01R02.txt
│   │   ├── S02R01.txt
│   │   ├── S02R02.txt
│   │   ├── S03R01.txt
│   │   ├── S03R02.txt
│   │   ├── S03R03.txt
│   │   ├── S04R01.txt
│   │   ├── S05R01.txt
│   │   ├── S05R02.txt
│   │   ├── S06R01.txt
│   │   ├── S06R02.txt
│   │   ├── S07R01.txt
│   │   ├── S07R02.txt
│   │   ├── S08R01.txt
│   │   ├── S09R01.txt
│   │   └── S10R01.txt
│   └── doc
│       ├── baechlin_TITB_2010 - Wearable Assistant for Parkinsons Disease Patients With the Freezing of Gait Symptom (preprint).pdf
│       ├── documentation.html
│       └── img
│           ├── annotation.jpg
│           ├── device.jpg
│           └── sketchOfPaths_landscape.png
└── text_to_csv_conversion.ipynb
