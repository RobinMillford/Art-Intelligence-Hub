# Art Intelligence Hub: An Ensemble Deep Learning Project

This project is a **college capstone project** featuring a multi-page Streamlit application that showcases advanced deep learning models for art analysis. It includes two main tools: an **Artwork Authenticity** classifier to distinguish between AI-generated and human-made art, and an **Art Style Ensemble Classifier** that identifies the artistic style of a painting.

The app is deployed on **Streamlit Cloud** for public access, enabling users to upload images and receive robust, interpretable predictions.

---

## 🔥 Features

- **Dual Classifier App**: A multi-page interface with two distinct art analysis tools.
- **Artwork Authenticity Classification**:
  - Predicts whether an image is AI-generated or real art using a fine-tuned `MobileNetV2`.
  - Includes **Grad-CAM heatmap visualization** to highlight the image regions that influenced the model's prediction.
- **Art Style Ensemble Classification**:
  - Identifies an artwork's style from 10 categories (e.g., _Impressionism_, _Surrealism_).
  - Combines the predictions of two powerful models (`EfficientNetV2-B2` and `ConvNeXt-Small`) for a more accurate and reliable result.
- **Interpretability**:
  - The "Art Style" page visualizes each model's top 3 predictions in a bar chart, showing its confidence and alternative considerations.
- **Modern, Interactive UI**:
  - A clean, dark-themed interface for a professional user experience.

---

## 🧠 Model Details

This project leverages three powerful, fine-tuned models:

1.  **Artwork Authenticity Model**:
    - **Base Model**: `MobileNetV2` (Transfer Learning)
    - **Task**: Binary classification (AI Art or Real Art).
2.  **Art Style Ensemble Models**:
    - **Model A**: `EfficientNetV2-B2`
    - **Model B**: `ConvNeXt-Small`
    - **Task**: Multi-class classification (10 art styles).
    - **Ensemble Strategy**: The final prediction is the averaged probability (soft voting) from both models.

---

## 📓 Kaggle Notebook

The model training and experimentation were conducted in a Kaggle Notebook. Key highlights include:

- **Dataset Preparation**:
  - Combined multiple datasets, including real art and AI-generated art (`Stable Diffusion` & `Midjourney`), totaling over **150,000 images**.
  - The final dataset was split into **80% training**, **10% validation**, and **10% test** sets.
- **Model Experimentation**:
  - Fine-tuned three separate deep learning architectures (`MobileNetV2`, `EfficientNetV2-B2`, `ConvNeXt-Small`) using a two-stage transfer learning approach.
  - Utilized modern training techniques such as the `AdamW` optimizer, learning rate scheduling, and advanced data augmentation.
- **Key Insights**:
  - The ensemble approach for style classification leverages the architectural diversity of `EfficientNetV2` (a powerful CNN) and `ConvNeXt-Small` (a modern hybrid) to achieve higher accuracy than either model alone.
  - `MobileNetV2` was chosen for the authenticity task due to its lightweight and efficient design, making it ideal for a fast, responsive app.

**Kaggle Notebook Link**: [Access Here](https://www.kaggle.com/code/yaminh/ai-vs-real-project)

---

## 🌟 Deployment

The app is deployed on **Streamlit Cloud**, making it accessible to users anywhere.

**Streamlit App Link**: [Access Here](https://classify-ai-image-or-realart.streamlit.app/)

---

## 🖥️ How to Use

### From GitHub

1.  **Fork the Repository**.
2.  **Clone the Repository**:
    ```bash
    git clone [https://github.com/RobinMillford/Art-Intelligence-Hub.git](https://github.com/RobinMillford/Art-Intelligence-Hub.git)
    ```
3.  **Navigate to the Project Folder**:
    ```bash
    cd Art-Intelligence-Hub
    ```
4.  **Install Git LFS**: Your models are large files. You must have Git LFS installed to download them.
    ```bash
    git lfs install
    git lfs pull
    ```
5.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
6.  **Run the App**:
    ```bash
    streamlit run app.py
    ```
7.  Open the local URL (http://localhost:8501) in your browser.

### From Streamlit Cloud

1.  **Visit the Deployed App**:
    Open the [Streamlit App](https://classify-ai-image-or-realart.streamlit.app/) in your browser.
2.  **Choose a Classifier**:
    Select either "Artwork Authenticity" or "Art Style Classifier" from the sidebar.
3.  **Upload an Image**:
    Drag and drop an image or select a file to classify.
4.  **View Results**:
    See the model's prediction and explore the "Individual Model Predictions" expander for more details.

---

## 💻 How to Contribute

1.  **Fork the Repository** on GitHub.
2.  **Clone Your Forked Repository**.
3.  **Create a New Branch**:
    ```bash
    git checkout -b feature/your-feature-name
    ```
4.  **Make Your Changes and Commit**.
5.  **Push Changes to Your Fork**.
6.  **Create a Pull Request**.

---

## 📚 Project Details

This project is part of our **college capstone project**, aimed at exploring the practical applications of **deep learning** and **computer vision** in art analysis. The goal was to develop a deployable app that combines efficient classification and intuitive user interaction.

### Technologies Used

- **TensorFlow/Keras**: For building and training the deep learning models.
- **Streamlit**: For app development and deployment.
- **OpenCV**: For image processing.
- **Git LFS**: For managing large model files.

---

## 📄 License

This project is licensed under the [AGPL-3.0 license](LICENSE).

---

## 🌟 Acknowledgments

- **Dataset Sources**: We used a combination of AI-generated art datasets and real art image collections from Kaggle.
- **Faculty Advisors**: Thanks to our professors for their invaluable guidance throughout this project.
- **Streamlit Community**: For resources and support in app deployment.

---
