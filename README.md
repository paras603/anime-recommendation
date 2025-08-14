# 🎌 Anime Recommendation System

This is a **Content-Based Anime Recommendation System** built with Python. It recommends anime based on genre similarity using **TF-IDF vectorization** and **cosine similarity** via the **k-Nearest Neighbors** algorithm.

---

## 📂 Project Overview

This project loads and processes a dataset of anime series, cleans the data, and uses **text-based genre information** to find similar shows. The goal is to recommend anime titles that are similar to a user’s input based on genre composition.

---

## 📊 Dataset

The dataset used is `anime.csv`, which includes metadata such as:

- `Name`
- `Genres`
- `Score`
- `Episodes`
- `Studios`
- `Source`
- `Type`
- `Rating`
- `Members`
- `Premiered`
- `MAL_ID`

> **Note**: You must have a valid `anime.csv` file in the same directory for the script to work. You can get datasets like this from sources such as [Kaggle](https://www.kaggle.com/datasets).

---

## 🧪 Features

- Handles missing and unknown data
- Processes multi-label genres and studios
- Converts genre data into TF-IDF vectors
- Uses cosine similarity to find nearest anime neighbors
- Recommends top 10 similar anime titles based on genre

---

## 🛠️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/anime-recommendation-system.git
cd anime-recommendation-system
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 📁 Add the Dataset
Place your anime.csv file in the project root directory.

## 🧾 Usage
Run the script from your terminal:
```bash
python anime_recommender.py
```

You will be prompted to enter an anime name:
```bash
Enter an anime name: Naruto
```

The system will then print 10 similar anime recommendations based on genre similarity.

## 🧹 Data Preprocessing Highlights
- Removes unnecessary columns (MAL_ID, Members, Premiered)
- Converts multi-label genre fields into clean lists
- Replaces "Unknown" values with appropriate defaults
- Transforms genre data using TfidfVectorizer

## 🤖 Recommendation Engine
Uses:
- TfidfVectorizer from scikit-learn to vectorize genre text
- NearestNeighbors to find the closest items using cosine similarity

## 📦 Requirements
The following libraries are required:
- pandas
- numpy
- scikit-learn
- scipy

Add this to your requirements.txt:
```bash
pandas
numpy
scikit-learn
scipy
```

## 🧠 Example Output
```bash
Enter an anime name: Naruto

             Name         Score  ...  Source     Rating
0    Bleach             7.94     ...  Manga     PG-13
1    One Piece          8.50     ...  Manga     PG-13
2    Fairy Tail         7.74     ...  Manga     PG-13
...
```

## 🧾 Notes
Input is case-sensitive. Make sure to type the anime name exactly as it appears.
You can modify the code to support fuzzy matching or search suggestions.

## 📬 License
This project is licensed under the MIT License.

## 👨‍💻 Author
Paras Mani Rai
