# From Classical Search Engine Methods to Transformers

Description: This project explores various search engine models, ranging from classical approaches to modern transformer-based techniques.

## 📌 Overview

This project is developed to compare different search engine models. The key models used include:

- ✅ Term Frequency (TF)
- ✅ Term Frequency-Inverse Document Frequency (TF-IDF)
- ✅ FastText
- ✅ FastText-IDF
- ✅ E5
- ✅ LaBSE

## 🚀 Data Used

The dataset consists of integrated content from four different Wikipedia articles.

## 📂 Folder Structure

```
SearchEngine/
│── data/
│   ├── preprocessor/
│   │   ├── get_bigram.py
│   │   ├── initial_files.py
│   │   ├── process_fasttext_vector.py
│   ├── processed/
│   │   ├── json files
│   ├── raw_files/
│   │   ├── collections.in
│   │   ├── requests.in
│   │   ├── stop_words.in
│   ├── load_processed_data.py
│── models/
│   ├── __init__.py
│   ├── e5.py
│   ├── fasttext.py
│   ├── fasttext_idf.py
│   ├── LaBSE.py
│   ├── model.py
│   ├── tf_idf.py
│   ├── tf.py
│── load_data.py
│── main.py
│── out_result.py
```