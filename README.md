# 📰 Fake News Detector

This is my **Fake News Detector** — a fully functional web application that uses **Machine Learning** to classify news articles as **Real** or **Fake**.  
I developed it completely from scratch, starting from **data preprocessing** and **model training**, to **API creation**, **web interface**, and **full production deployment** on my VPS with **Gunicorn + Apache + HTTPS**.

---


## 🚀 Live Demo
**Web Interface:** [https://haseebsagheer.com/fake-news-detector](https://haseebsagheer.com/fake-news-detector)  
**API Endpoint:** `/predict`


---

## 📌 Project Motivation

Misinformation spreads faster than ever, and it's not just a social issue — it's a data problem.  
As a Data Scientist in training, I wanted to create something that:
- Tackles a real-world problem
- Involves a full **end-to-end ML pipeline**
- Challenges me beyond Jupyter Notebooks, pushing into **production deployment**
- Strengthens my **Flask, API design, server administration, and ML deployment skills**

This was also a way to **add a strong real-world AI project to my portfolio**, something that potential employers and clients can interact with live.

---

## 📊 Dataset Details

I used the **Fake News Detection Dataset** from Kaggle by **Emine Yetim**.

Dataset link: [Kaggle – Fake News Detection Datasets](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)

It consists of two CSV files:
- **`fake.csv`** → Articles labeled as "Fake"
- **`true.csv`** → Articles labeled as "Real"

Each file has:
- **title** — headline/title of the news article
- **text** — the body/content of the news article
- **subject** — category/subject of the article
- **date** — publication date

---

## 🛠 Technologies Used

### **Backend**
- Python 3.10
- Flask (API and Web Interface)
- Gunicorn (WSGI application server)
- Apache (Reverse Proxy + SSL)
- Systemd (service manager for auto-start)

### **Machine Learning**
- scikit-learn
- pandas, numpy
- joblib (model persistence)
- LinearSVC (Support Vector Machine with linear kernel)
- TF-IDF Vectorizer

### **Frontend**
- HTML, CSS (custom, responsive design)
- Simple, minimal interface for usability

### **Deployment & Security**
- Ubuntu VPS
- Apache VirtualHost configurations for multiple apps
- Let's Encrypt SSL for HTTPS
- Reverse Proxy with ProxyPass/ProxyPassReverse

---

## 🔄 Development Workflow

### 1️⃣ **Data Preprocessing**
I started by loading both CSV files into pandas DataFrames and:
- Merged them with labels: `1` for real, `0` for fake
- Removed duplicates and empty rows
- Lowercased all text
- Removed punctuation and special characters
- Tokenized words
- Removed stopwords
- Lemmatized tokens for normalization
- Merged `title` and `text` into a single column for better context

---

### 2️⃣ **Feature Extraction**
- Used **TF-IDF Vectorization** to convert text into numerical features
- Tuned vectorizer parameters (`max_df`, `min_df`, `ngram_range`) to balance dimensionality and context capture

---

### 3️⃣ **Model Selection & Training**
- Tried several models: Logistic Regression, Random Forest, Naive Bayes, LinearSVC
- Selected **LinearSVC** for:
  - Speed
  - High accuracy
  - Low resource usage in production
- Trained with TF-IDF vectors
- Saved entire pipeline (`TF-IDF + LinearSVC`) with `joblib`

---

### 4️⃣ **Building the Flask App**
The app has **two main components**:
1. **Web UI** — user-friendly interface where users paste a news article
2. **API Endpoint (`/predict`)** — accepts POST JSON requests and returns predictions

**Example API Request:**
```bash
curl -X POST https://haseebsagheer.com/fake-news-detector/predict \
  -H "Content-Type: application/json" \
  -d '{"title":"Breaking News","text":"Some news content..."}'
```
# Fake News Detector

## Example API Response
```json
{
  "label": "Real",
  "len_chars": 150,
  "model": "LinearSVC_TFIDF",
  "score": 0.85
}
```

---

## 🌟 Features
- ✅ **Real-Time Classification** — returns prediction instantly  
- ✅ **API Access** — usable in external applications  
- ✅ **User-Friendly Interface** — paste text and click predict  
- ✅ **Secure (HTTPS)** — fully encrypted connections  
- ✅ **Multiple Apps on One VPS** — CV Generator + Fake News Detector  
- ✅ **Error Handling** — proper JSON responses for invalid requests  

---

## 🚀 Live Demo
**Web Interface:** [https://haseebsagheer.com/fake-news-detector](https://haseebsagheer.com/fake-news-detector)  
**API Endpoint:** `/predict`

---

## ⚔️ Challenges I Overcame

### Multiple Flask Apps on Same VPS
My CV Generator was already deployed. Setting up a second Flask app without breaking the first required careful Apache and Gunicorn configuration.

### Apache Reverse Proxy Issues
At first, the frontend got `SyntaxError: Unexpected token '<'...` errors.  
I fixed it by correcting `ProxyPassMatch` rules and ensuring JSON was passed correctly.

### Persistent Service
Initially, I had to manually start Gunicorn after every reboot.  
Learned how to use `systemd` to create a permanent service.

### CORS & SSL
Configured HTTPS for secure API calls and avoided mixed-content errors in browsers.

---

## 📚 What I Learned
- End-to-End ML Deployment — from data preprocessing to production API  
- Flask Best Practices for building maintainable apps  
- Gunicorn & Apache Reverse Proxy integration  
- Linux Server Management — virtual environments, services, ports, logs  
- Model Optimization for real-time predictions  
- Debugging server-side JSON parsing issues  

---

## 📂 Project Structure
```plaintext
fake-news-detector/
│
├── app/                   
│   ├── app.py             # Flask app
│   ├── templates/         # HTML UI
│   ├── static/            # CSS files
│
├── artifacts/
│   ├── best_pipeline.joblib  # Trained ML model
│
├── requirements.txt
└── README.md
```

---

## 🏅 Certifications
- **Credly Profile:** [https://www.credly.com/users/haseeb-sagheer](https://www.credly.com/users/haseeb-sagheer)  
- **Coursera Profile:** [https://www.coursera.org/learner/haseeb-sagheer](https://www.coursera.org/learner/haseeb-sagheer)

---

## 🙌 Acknowledgments
- Dataset by [Emine Yetim – Kaggle](https://www.kaggle.com)  
- ChatGPT — only used for HTML & CSS help for the UI. All backend, ML, API, and deployment were coded by me.

---

## 📞 Contact
- 📧 **Email:** engrhaseebsagheer@gmail.com  
- 📱 **Phone/WhatsApp:** +92 308 2496103  
- 💼 **LinkedIn:** [https://linkedin.com/in/haseeb-sagheer](https://linkedin.com/in/haseeb-sagheer)  
- 🐙 **GitHub:** [https://github.com/engrhaseebsagheer](https://github.com/engrhaseebsagheer)  
- 🌐 **Portfolio:** [https://haseebsagheer.com](https://haseebsagheer.com)  

---


