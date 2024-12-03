```markdown
# 📰 Classification of News Articles Using Machine Learning Techniques with PySpark

## 📚 Overview
This project focuses on building and evaluating a multi-class classification model using PySpark. The workflow includes data preprocessing, feature extraction, model training, and performance evaluation using various metrics precision,recall, and confusion matrix.. Additionally, a **Flask-based web interface** is provided for users to interact with the model.

---

## 🔧 Features

- **Data Preprocessing**:
  - Tokenization, stopword removal, and feature extraction using TF-IDF.

- **Model Training**:
  - Implementation of multi-class classification models using PySpark's MLlib.

- **Evaluation Metrics**:
  - Precision, Recall, and F1-score for detailed model assessment.
  - Confusion Matrix for analyzing classification errors.

- **Visualization**:
  - Visualize model performance using Matplotlib and Seaborn.

- **Web Interface**:
  - Flask-based web application to interact with the trained model.
  - Accepts text input for classification and displays the predicted category.
  - Frontend designed with **HTML/CSS** for a seamless user experience.

---

## 🛠️ Prerequisites

Ensure the following are installed on your system:
- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)

---

## 🚀 Getting Started

### Build and Run the Application
To build and run the containerized application:
```bash
docker-compose up
```

### Access the Flask Web Interface
Once the application is running, open your browser and navigate to:
[http://127.0.0.1:5000](http://127.0.0.1:5000)

You will see the web interface where you can input a news article for classification.

### Access the Container
To access the running Python container:
```bash
docker exec -it container_name_or_id bash
```

### Stop and Remove the Container
To stop and remove the container:
```bash
docker compose down
```

---

## 🛡️ License

This project is licensed under the **MIT License**. See the `LICENSE` file for more details.

---

## 🙌 Acknowledgments

- **Dataset**: Sourced from [Kaggle](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset).
```