from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np
import bcrypt


# Charger les données
app = Flask(__name__)
app.secret_key = 'a3f8d09e6b274eafbeef1234c9a86b27'


# Charger le CSV
movies_df = pd.read_csv('popular_movies_cleaned.csv')
movies_df.fillna("", inplace=True)
movies_df['combined_text'] = (
    movies_df['title'] + " " + movies_df['genres'] + " " + movies_df['overview'] +
    " " + movies_df['cast'] + " " + movies_df['director']
)

# Prétraitement des données pour le système de recommandation
texts = movies_df['combined_text'].tolist()
vect = TfidfVectorizer()
tfidf_mat = vect.fit_transform(texts).toarray()

svd = TruncatedSVD(n_components=50, random_state=42)
data_encoded = svd.fit_transform(tfidf_mat)

# Base de données utilisateur (simple pour la démo)
users = {}
user_history = {}

# Fonction de recommandation
def recommend_movies(user_id, n_recommendations=5):
    # Obtenir les films vus par l'utilisateur
    user_movies = user_history.get(user_id, {})
    if not user_movies:
        print("Aucun historique trouvé pour cet utilisateur.")
        return []

    # Calculer la similarité pour tous les films
    similarity = np.dot(data_encoded, data_encoded.T)

    # Ajuster les scores de similarité avec les notes utilisateur
    user_weights = np.zeros(len(movies_df))
    for movie_id, rating in user_movies.items():
        if movie_id in movies_df['id'].values:
            movie_index = movies_df.index[movies_df['id'] == movie_id][0]
            user_weights += similarity[movie_index] * rating  # Pondération par la note

    # Diviser par le nombre de films notés pour une moyenne pondérée
    user_weights /= len(user_movies)

    # Exclure les films déjà vus
    watched_indices = [movies_df.index[movies_df['id'] == movie_id][0] for movie_id in user_movies if movie_id in movies_df['id'].values]
    user_weights[watched_indices] = -1  # Assigner un score négatif pour exclure ces films

    # Trier les recommandations
    top_indices = np.argsort(user_weights)[-n_recommendations:][::-1]

    # Retourner les recommandations
    recommendations = []
    for idx in top_indices:
        if user_weights[idx] > 0:  # S'assurer que le score est positif
            recommendations.append({
                "title": movies_df.iloc[idx]['title'],
                "genres": movies_df.iloc[idx]['genres'],
                "overview": movies_df.iloc[idx]['overview'],
                "poster": movies_df.iloc[idx]['poster_path'],  # Ajouter l'affiche si disponible
                "score": movies_df.iloc[idx]['vote_average'],
                "runtime": movies_df.iloc[idx]['runtime'],
                "release_date": movies_df.iloc[idx]['release_date'],
                "similarity": user_weights[idx]

            })
    return recommendations


@app.route('/')
def home():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users:
            return "Utilisateur existant !"

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        users[username] = hashed_password
        user_history[username] = {}

        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username not in users or not bcrypt.checkpw(password.encode('utf-8'), users[username]):
            return "Nom d'utilisateur ou mot de passe incorrect !"

        session['user'] = username
        return redirect(url_for('dashboard'))
    return render_template('login.html')


@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))

    username = session['user']

    if request.method == 'POST':
        movie_id = int(request.form['movie_id'])
        rating = int(request.form['rating'])
        user_history[username][movie_id] = rating
        return redirect(url_for('dashboard'))

    # Préparer les données des films vus avec plus d'informations
    watched_movies = [
        {
            "id": movie_id,
            "rating": rating
        } for movie_id, rating in user_history[username].items()
    ]

    return render_template('dashboard.html',
                           movies_df=movies_df,
                           watched_movies=watched_movies)

@app.route('/recommendations', methods=['GET', 'POST'])
def recommendations():
    if 'user' not in session:
        return redirect(url_for('login'))

    username = session['user']

    # Récupérer les films recommandés
    recs = recommend_movies(username, n_recommendations=12)

    # Récupérer les genres uniques à partir des films recommandés
    unique_genres = sorted(set(genre.strip() for rec in recs for genre in rec['genres'].split(',')))

    # Récupérer les années uniques à partir des films recommandés
    unique_years = sorted(set(rec['release_date'][:4] for rec in recs))

    # Appliquer les filtres si des valeurs sont fournies
    selected_genre = request.args.get('genre', "")
    selected_year = request.args.get('year', "")

    # Appliquer les filtres sur les films recommandés
    filtered_recs = recs
    if selected_genre:
        filtered_recs = [rec for rec in filtered_recs if selected_genre.lower() in rec['genres'].lower()]
    if selected_year:
        filtered_recs = [rec for rec in filtered_recs if rec['release_date'][:4] == selected_year]

    return render_template('recommendations.html',
                           recommendations=filtered_recs,
                           genres=unique_genres,
                           years=unique_years,
                           selected_genre=selected_genre,
                           selected_year=selected_year)


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=8080)
    app.run(debug=True)
