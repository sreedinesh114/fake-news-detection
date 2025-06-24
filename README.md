📰 Fake News Detection using Machine Learning + NLP

This project detects fake news using Natural Language Processing (NLP) and machine learning (Logistic Regression + TF-IDF). Includes a Streamlit web UI for user interaction.

🚀 Features

- Preprocessing with NLTK
- TF-IDF vectorization
- Logistic Regression classifier
- Accuracy ~99%
- Interactive Streamlit interface

📁 Folder Structure

fake-news-detection/
├── dataset/ # Contains Fake.csv and True.csv
├── model/ # Saved ML model and vectorizer
├── main.py # Training and evaluation script
├── app.py # Streamlit web app
├── requirements.txt # Project dependencies
└── README.md

🔧 How to Run

1. Install dependencies
bash
pip install -r requirements.txt
2. Train the model
bash
python main.py
3. Launch the web app
bash
streamlit run app.py