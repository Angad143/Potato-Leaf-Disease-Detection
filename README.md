# **Tools and Libraries Used in Our Project**  

<div style="display: flex; flex-wrap: wrap; gap: 10px;">  
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white" alt="Streamlit" style="flex: 1 1 30%;">  
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white" alt="TensorFlow" style="flex: 1 1 30%;">  
  <img src="https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white" alt="Keras" style="flex: 1 1 30%;">  
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white" alt="OpenCV" style="flex: 1 1 30%;">  
  <img src="https://img.shields.io/badge/Numpy-013243?style=flat&logo=numpy&logoColor=white" alt="Numpy" style="flex: 1 1 30%;">  
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white" alt="Pandas" style="flex: 1 1 30%;">  
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=python&logoColor=white" alt="Matplotlib" style="flex: 1 1 30%;">  
  <img src="https://img.shields.io/badge/Seaborn-3776AB?style=flat&logo=python&logoColor=white" alt="Seaborn" style="flex: 1 1 30%;">  
  <img src="https://img.shields.io/badge/Pillow-3C1A74?style=flat&logo=pillow&logoColor=white" alt="Pillow" style="flex: 1 1 30%;">  
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white" alt="Jupyter" style="flex: 1 1 30%;">  
</div> 

-------------
# 🥔🌿 Potato Leaf Disease Detection System 🥔🌿

## 📌 Project Overview
This project focuses on detecting potato leaf diseases using deep learning techniques. A Convolutional Neural Network (CNN) model is trained on a dataset of potato leaf images to classify leaves as **Healthy**, **Early Blight**, or **Late Blight**. This system aims to help farmers and agricultural experts diagnose diseases early and take preventive measures.

## 📂 Dataset
The dataset consists of images of potato leaves categorized into three classes:
- **Healthy**
- **Early Blight**
- **Late Blight**

The dataset was provided by Edunet Foundation and preprocessed for model training.

## 🛠️ Implementation Steps
1. **Data Collection**: Collected images of potato leaves from the Kaggle dataset.
2. **Data Preprocessing**: Applied image augmentation and resizing.
3. **Model Selection**: Used a CNN architecture based on TensorFlow and Keras.
4. **Training & Validation**: Trained the model.
5. **Evaluation**: Assessed the model's accuracy using metrics like precision and recall.
6. **Deployment**: Implemented the model in a web application using Streamlit.

## 🏆 Results & Findings
- Achieved **97.8% accuracy** on train data.
- Successfully classified leaf images into three categories.
- Provided an easy-to-use interface for users to upload and analyze potato leaf images.

## 🚀 How to Run the Project
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Angad143/Potato-Leaf-Disease-Detection-System.git
cd potato-leaf-disease-detection
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Run the Streamlit Web App
```bash
streamlit run app.py
```

## **Conclusion**  
This project demonstrates how AI can be leveraged for precision agriculture by accurately detecting potato leaf diseases. By integrating deep learning techniques, it provides a practical solution for early disease diagnosis, helping farmers take timely action to prevent crop loss and improve yield.  

📹 **Video demonstration:** [Project Demo Video](https://github.com/Angad143/Potato-Leaf-Disease-Detection-System/blob/main/Output%20Images%20and%20Video/Potatoes%20Leaf%20Disease%20Detections.mp4)   


## 📖 References
1. Shruthi, U., Nagaveni, V., & Raghavendra, B. K. (2019, March). A Review on Machine Learning Classification Techniques for Plant Disease Detection. In 2019 5th International Conference on Advanced Computing & Communication Systems (ICACCS) (pp. 281-284). IEEE.

2. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444.

3. Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for image-based plant disease detection. Frontiers in Plant Science, 7, 1419.

4. Sladojevic, S., Arsenovic, M., Anderla, A., Culibrk, D., & Stefanovic, D. (2016). Deep neural networks based recognition of plant diseases by leaf image classification. Computational Intelligence and Neuroscience, 2016, 1-11.4.
 
5. TensorFlow Documentation for Image Classification. and many more.......

## 🔥 Future Enhancements
- Improve model accuracy with transfer learning techniques.
- Extend the project to detect diseases in other crops.
- Deploy on cloud platforms for real-time detection.
