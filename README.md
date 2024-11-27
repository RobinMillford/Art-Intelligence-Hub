---

# AI vs Real Art Image Classification

This project is a **college capstone project** designed to classify images as **AI-generated art** or **Real art** using a fine-tuned MobileNetV2 model. The model utilizes transfer learning to achieve high accuracy and interpretability through Grad-CAM visualizations. The app has been deployed on **Streamlit Cloud** for public access, enabling users to upload images and receive predictions directly.

---

## 🔥 Features

- **AI Art vs Real Art Classification**  
  Predicts whether an image is AI-generated or real art.
  
- **Grad-CAM Heatmap Visualization**  
  Highlights the image regions that influenced the model's prediction.

- **Interactive Pagination**  
  Allows users to upload and navigate multiple images seamlessly.

- **Custom Background and Dynamic UI**  
  User-friendly interface with enhanced visuals and smooth interactions.

---

## 🧠 Model Details

- **Base Model**: MobileNetV2 (Transfer Learning)  
- **Training Accuracy**: 94.50%  
- **Validation Accuracy**: 95.88%  
- **Dataset**: The model was trained on a dataset of **27,266 images** and validated on **3,370 images**.

---

## 📓 Kaggle Notebook

In this project, a Kaggle Notebook was used extensively for training, testing, and analyzing multiple models for classifying AI-generated and real art. Key points include:

- **Dataset Preparation:**
  - The dataset consisted of **27,266 training images** and **3,370 validation images**.
  - Divided into two sets: **AI-generated art** and **real art**.
  - Addressed class imbalance by experimenting with **class weights**.

- **Problem Formulation:**
  - The task was formulated as a **binary classification problem**: AI-generated vs. real art.
  - Leveraged **transfer learning** to enable cost-effective use of deep CNN architectures.

- **Model Experimentation:**
  - Experimented with **six CNN architectures**, including:
    - MobileNetV1 (focus model)
    - MobileNetV2
    - DenseNet121
    - InceptionV3
    - ResNet50
  - Fine-tuned models with optimized parameters such as learning rate, batch size, and class weights.

- **Key Insights:**
  - MobileNetV1 was prioritized for its reputation in transfer learning applications.
  - Both **MobileNetV1** and **MobileNetV2** performed exceptionally well, with closely comparable results.
  - MobileNetV2 was ultimately selected for deployment in the **Streamlit app** due to its **lightweight and scalable design**.

- **Training & Evaluation:**
  - Achieved a **training accuracy of 94.50%** and a **validation accuracy of 97.65%** with the best model.

- **Model Saving:**
  - Saved the fine-tuned **MobileNetV2 model** (`MobileNetV2_finetuned_model.keras`) for deployment.

- **Visualization & Interpretation:**
  - Used **Grad-CAM heatmaps** to interpret model predictions by highlighting image areas most relevant to classification.

This combination of methods showcases the project's potential to provide a robust solution for distinguishing between AI-generated and real artworks.

---

The project leverages **MobileNetV2's lightweight architecture** to ensure efficient predictions without sacrificing accuracy.

---

## 🌟 Deployment

The app is deployed on **Streamlit Cloud**, making it accessible to users anywhere.

**Streamlit App Link**: [Access Here](https://classify-ai-image-or-realart.streamlit.app/)

**APP**

![Alt Text](https://github.com/RobinMillford/Classifying-AI-Generated-and-Real-Art/blob/main/app.png)

---

## 🖥️ How to Use

### From GitHub

1. **Fork the Repository**:  
   - Click the **Fork** button at the top-right corner of the repository.
2. **Clone the Repository**:  
   ```bash
   git clone https://github.com/RobinMillford/Classifying-AI-Generated-and-Real-Art.git
   ```
3. **Navigate to the Project Folder**:  
   ```bash
   cd Classifying-AI-Generated-and-Real-Art
   ```
4. **Install Dependencies**:  
   ```bash
   pip install -r requirements.txt
   ```
5. **Run the App**:  
   ```bash
   streamlit run app.py
   ```
6. Open the local URL (http://localhost:8501) in your browser to access the app.

### From Streamlit Cloud

1. **Visit the Deployed App**:  
   Open the [Streamlit App](https://classify-ai-image-or-realart.streamlit.app/) in your browser.

2. **Upload Images**:  
   Drag and drop images or select files to classify as AI-generated or real art.

3. **Navigate Results**:  
   Use the **Next** and **Previous** buttons to explore predictions and Grad-CAM visualizations for each uploaded image.

---

## 💻 How to Contribute

1. **Fork the Repository** on GitHub.  
2. **Clone Your Forked Repository**:  
   ```bash
   git clone https://github.com/RobinMillford/Classifying-AI-Generated-and-Real-Art.git
   ```
3. **Create a New Branch**:  
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make Your Changes and Commit**:  
   ```bash
   git commit -m "Add your message here"
   ```
5. **Push Changes to Your Fork**:  
   ```bash
   git push origin feature/your-feature-name
   ```
6. **Create a Pull Request**:  
   Open a pull request from your branch to the original repository's `main` branch.

---

## 📚 Project Details

This project is part of our **college capstone project**, aimed at exploring the practical applications of **deep learning** and **computer vision** in art classification. The goal was to develop a deployable app that combines efficient classification and intuitive user interaction.

### Technologies Used

- **TensorFlow/Keras**: For building and training the MobileNetV2 model.  
- **Streamlit**: For app development and deployment.  
- **OpenCV**: For image processing.  
- **Grad-CAM**: For interpretability through heatmaps.  

---

## 📄 License

This project is licensed under the [AGPL-3.0 license](LICENSE).

---

## 🌟 Acknowledgments

- **Dataset Sources**: We used a combination of AI-generated art datasets and real art image collections [Source 1](https://www.kaggle.com/datasets/superpotato9/dalle-recognition-dataset) and [Source 2](https://www.kaggle.com/datasets/sankarmechengg/art-images-clear-and-distorted).  
- **Faculty Advisors**: Thanks to our professors for their invaluable guidance throughout this project.  
- **Streamlit Community**: For resources and support in app deployment.  

---
