from collections import defaultdict
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify


data =pd.read_table('./data.csv',delimiter=',')
scaler = StandardScaler()

number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

'''
Finds song details from spotify dataset. If song is unavailable in dataset, it returns none.
'''
def find_song(name, year):
    song_data = defaultdict()
    results = sp.search(q= 'track: {} year: {}'.format(name,year), limit=1)
    if results['tracks']['items'] == []:
        return None

    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = [name]
    song_data['year'] = [year]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)

'''
Fetches song details from dataset. If info is unavailable in dataset, it will search details from the spotify dataset.
'''
def get_song_data(song, spotify_data):
    try:
        song_data = spotify_data[(spotify_data['name'] == song['name']) 
                                & (spotify_data['year'] == song['year'])].iloc[0]
        print('Fetching song information from local dataset')
        return song_data
    
    except IndexError:
        print('Fetching song information from spotify dataset')
        return find_song(song['name'], song['year'])
'''
Fetches song info from dataset and does the mean of all numerical features of the song-data.
'''
def get_mean_vector(song_list, spotify_data):
    song_vectors = []
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)  
    
    song_matrix = np.array(list(song_vectors))#nd-array where n is number of songs in list. It contains all numerical vals of songs in sep list.
    #print(f'song_matrix {song_matrix}')
    return np.mean(song_matrix, axis=0) # mean of each ele in list, returns 1-d array


# Load the pre-trained deep learning model
model = load_model("./song_recommender_model.h5")

'''
Flattenning the dictionary by grouping the key and forming a list of values for respective key.
'''
def flatten_dict_list(dict_list):
    flattened_dict = defaultdict()
    for key in dict_list[0].keys(): 
        flattened_dict[key] = [] # 'name', 'year'
    for dic in dict_list:
        for key,value in dic.items():
            flattened_dict[key].append(value) # creating list of values
    return flattened_dict

def recommend_songs_dl(song_list, spotify_data, model, scaler, n_songs=10):
    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)

    song_center = get_mean_vector(song_list, spotify_data)
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))

    # Predict cluster labels using the deep learning model
    cluster_probs = model.predict(scaled_song_center)
    cluster_label = np.argmax(cluster_probs, axis=1)[0]

    # Get the n_songs closest to the cluster center
    data_cluster = spotify_data[spotify_data['cluster_label'] == cluster_label]
    distances = cdist(scaled_song_center, scaled_data[data_cluster.index], 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])  # Fixed the typo here

    rec_songs = data_cluster.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[metadata_cols].to_dict(orient='records')


def get_recommendations_dl(song_list, n_songs=10):
    # Get recommendations using the recommend_songs_dl() function
    recommendations = recommend_songs_dl(song_list, data, model, scaler, n_songs)

    # Print the recommendations
    print("Recommended songs:")
    for i, song in enumerate(recommendations, 1):
        print(f"{i}. {song['name']} ({song['year']}) by {song['artists']}")

    return recommendations


app = Flask(__name__)

@app.route('/api/recommend', methods=['POST'])
def api_predict():
    
    recommendation = get_recommendations_dl(song_list, n_songs=10)
    response = jsonify({'recommendation': recommendation})
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)