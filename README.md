Run in Colab

📚 Book Recommendation Engine

This project is a solution to the Book Recommendation Engine project
 from the Data Analysis with Python Certification by freeCodeCamp.

The goal is to create a book recommendation system that suggests books similar to a given book based on user ratings.

🧠 Project Overview

The system uses the Book-Crossings dataset, which contains book details and user ratings.

Data is filtered to include:

Books with at least 100 ratings

Users who have rated at least 200 books

A user-item matrix is created with books as rows, users as columns, and ratings as values.

🧰 Features

✅ Data preprocessing and filtering

✅ Sparse user-book matrix creation

✅ k-Nearest Neighbors (kNN) algorithm for book recommendations

✅ Returns top 5 recommended books with similarity distances

📂 File Structure
.
├── BX-Books.csv              # Book metadata
├── BX-Book-Ratings.csv       # Ratings dataset
├── book_recommendation.py    # Main Python script
└── README.md

📈 How It Works

Load and filter data

book_counts = df_ratings['isbn'].value_counts()
valid_books = book_counts[book_counts >= 100].index

user_counts = df_ratings['user'].value_counts()
valid_users = user_counts[user_counts >= 200].index
df_ratings_filtered = df_ratings[
    df_ratings['isbn'].isin(valid_books) &
    df_ratings['user'].isin(valid_users)
]


Create book-user matrix

book_user_matrix = merged_df.pivot_table(
    index='title',
    columns='user',
    values='rating',
    fill_value=0
)


Train kNN model

model_knn = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')
model_knn.fit(book_user_matrix.values)


Get recommendations

def get_recommends(book_title):
    book_vector = book_user_matrix.loc[book_title].values.reshape(1, -1)
    distances, indices = model_knn.kneighbors(book_vector)
    rec_books = [[book_user_matrix.index[idx], float(dist)] for dist, idx in zip(distances.flatten()[1:], indices.flatten()[1:])]
    return [book_title, rec_books]

🔍 Evaluation

Example usage:

get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")


Returns a list of recommended books similar in user ratings to the input book.

The model passes the FCC challenge when it correctly identifies books that are highly similar.

📝 Notes

Uses Python, Pandas, NumPy, and scikit-learn.

Recommendation is based on user rating similarity (cosine distance).

Demonstrates data cleaning, filtering, and machine learning for recommendation systems.

How to Run

Open the notebook in Google Colab.

Run all cells sequentially.

No local environment setup required (Colab includes TensorFlow 2.x).

👤 Author

BondAlex-maker
🔗 GitHub Repository
