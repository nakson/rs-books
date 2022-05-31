from typing import Optional, List
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import os
import csv
from sklearn.cluster import estimate_bandwidth
from surprise.model_selection import train_test_split
from surprise.prediction_algorithms.random_pred import NormalPredictor
from utils import map_genre
import json
from surprise import dump
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import Dataset
from surprise import Reader
import pandas as pd
from surprise import accuracy



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================DATA=========================

# Loading data file path ...
file_path_users = './data/BX-Users.csv'
file_path_ratings = './data/BX-Book-Ratings.csv'
file_path_books = './data/BX-Books.csv'

# Loading .csv data into pandas df
book_df = pd.read_csv(file_path_books, names = ['iid', 'title', 'author', 'year', 'publisher','image'], sep=',',skiprows=1, dtype={'year':int})
rating_df = pd.read_csv(file_path_ratings, names = ['uid', 'iid','rating'], sep=',',skiprows=1)
common = pd.merge(rating_df, book_df, how ='inner', on =['iid'])
rating_df = rating_df.loc[rating_df['iid'].isin(common['iid'])] # only keep the iid that book_df has
user_df = pd.read_csv(file_path_users, names = ['uid', 'age'], sep=',',skiprows=1)
user_df = user_df.dropna()

# df -> surprise Dataset
#data_rating = Dataset.load_from_df(rating_df[["uid", "iid", "rating"]].sample(n=10), Reader(rating_scale=(1, 5)))


# method 1
sim_options_1 = {'name': 'cosine',
                    'user_based': False,  # compute  similarities between items
                    'min_support': 1  # minimum number of common items for two items
               }
algo = KNNBasic(k=20, sim_options=sim_options_1)
# medhod 2
sim_options_2 = {'name': 'cosine',
                    'user_based': True,  # compute  similarities between users
                    'min_support': 1  # minimum number of common items for two user
               }
algo_means = KNNWithMeans(k=20, sim_options=sim_options_2)

selected_data = rating_df.sample(n=20)
rating_list = []
"""
=================== Body =============================
"""


# == == == == == == == == == API == == == == == == == == == == =

# show four genres
@app.get("/api/genre")
def get_genre():
    return {'genre': ["Action", "Adventure", "Animation", "Children"]}

# show all generes
'''
@app.get("/api/genre")
def get_genre():
    return {'genre': ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
                      "Documentary", "Drama", "Fantasy", "Film_Noir", "Horror", "Musical", "Mystery",
                      "Romance", "Sci_Fi", "Thriller", "War", "Western"]}
'''

@app.post("/api/books")
def books_in_age(form: dict):
    print("----User selected----")
    print(form['age'])
    print(form['name'])
    selected_books, selected_iid, selected_uid = filter_data_by_age(form['age'], user_df, rating_df, book_df)
    # update data for training
    global selected_data
    selected_data = rating_df.loc[rating_df['uid'].isin(selected_uid)].sample(n=8000)
    data_rating = Dataset.load_from_df(selected_data[["uid", "iid", "rating"]], Reader(rating_scale=(1, 10)))
    # fit algo 1
    trainset = data_rating.build_full_trainset()
    global algo
    algo.fit(trainset)

    sample_books = selected_books.loc[selected_books['iid'].isin(selected_data['iid'])]
    books_in_age = sample_books.sample(n=30)
    books_in_age.loc[:, 'score'] = None
    return books_in_age.to_json(orient='index')



@app.post("/api/rate_recommend")
async def get_recommend(books: List, n=200):
    trainset_2 = list_add(books)
    global algo_means
    algo_means.fit(trainset_2)

    res = []
    all_results = {}
    books_list = [iid for iid in selected_data.iid]
    for i in books_list:
        uid = 300000
        iid = i
        pred = algo_means.predict(uid,iid).est
        all_results[iid] = pred
    sorted_list = sorted(all_results.items(), key = lambda d:d[1], reverse=True)
    global rating_list
    rating_list = sorted_list
    # sorted_list
    print("+++++++++++++++++sorted_list size:",len(sorted_list))
    for i in range(n):
        print(sorted_list[i])
        res.append(sorted_list[i][0])

    print("+++++++++++++++++++res",res)

    rec_books = book_df.loc[book_df['iid'].isin(res)]
    return json.loads(rec_books.to_json(orient="records"))


@app.get("/api/like_recommend/{item_id}")
async def like_recommend(item_id):
    print("-=========================item_id:",item_id)
    # get_similar
    res = get_similar_items(str(item_id), n=5) # similar items
    res = [i for i in res]
    print("res:",res)
    rec_books = book_df.loc[book_df['iid'].isin(res)]
    print("res_books:",rec_books)

    rec_books.loc[:, 'like_eval'] = 0
    return json.loads(rec_books.to_json(orient="records"))
    


@app.post("/api/like_eval")
async def get_like_rmse(books: List):
    #print("===========like eval books",books)
    bad_rec = []
    good_rec = 0
    uid = []
    iid = []
    eval = []

    for book in books:
        if book['like_eval'] < 0:
            bad_rec.append(book)
            eval.append(0)
            iid.append(book['iid'])
        if book['like_eval'] > 0:
            good_rec += 1
            eval.append(8)
            iid.append(book['iid'])
    print("===================like_evaL iid: ",iid)
    print("===================like_evaL eval: ",eval)
    for i in range(len(iid)):
        uid.append(300000)

    test_dict = {
        "uid": uid,
        "iid": iid,
        "rating": eval,
    }

    df = pd.DataFrame(test_dict)
    reader = Reader(rating_scale=(1, 10))

    test_data = Dataset.load_from_df(df[["uid", "iid", "rating"]], reader)
    testset = test_data.build_full_trainset().build_testset()

    global algo
    pred = algo.test(testset)
    rmse = accuracy.rmse(pred, verbose=True)
    print("==============rate rmse:",rmse)

    return rmse


@app.post("/api/rate_eval")
async def get_rate_rmse(books: List):
    #print(books[0])
    iid = []
    eval = []
    uid = []
    for book in books:
        if book['rate_eval'] > 0:
            iid.append(book['iid'])
            eval.append(book['rate_eval']*2)
    print("===================rate_evaL iid: ",iid)
    print("===================rate_evaL eval: ",eval)
    for i in range(len(iid)):
        uid.append(300000)

    test_dict = {
    "uid": uid,
    "iid": iid,
    "rating": eval,
    }

    df = pd.DataFrame(test_dict)
    print(df)

    reader = Reader(rating_scale=(1, 10))
    test_data = Dataset.load_from_df(df[["uid", "iid", "rating"]], reader)
    testset = test_data.build_full_trainset().build_testset()

    global algo_means
    pred = algo_means.test(testset)
    acc = accuracy.rmse(pred, verbose=True)
    print("==============acc:",acc)
    return acc

def user_add(iid, score):
    user = '944'
    # simulate adding a new user into the original data file
    df = pd.read_csv('./u.data')
    df.to_csv('new_' + 'u.data')
    with open(r'new_u.data',mode='a',newline='',encoding='utf8') as cfa:
        wf = csv.writer(cfa,delimiter='\t')
        data_input = []
        s = [user,str(iid),int(score),'0']
        data_input.append(s)
        for k in data_input:
            wf.writerow(k)

def get_initial_items(iid, score, n=12):
    res = []
    user_add(iid, score)
    file_path = os.path.expanduser('new_u.data')
    reader = Reader(line_format='user item rating timestamp', sep='\t')
    data = Dataset.load_from_file(file_path, reader=reader)
    trainset = data.build_full_trainset()
    algo = KNNBasic(sim_options={'name': 'pearson', 'user_based': False})
    algo.fit(trainset)
    dump.dump('./model',algo=algo,verbose=1)
    all_results = {}
    for i in range(1682):
        uid = str(944)
        iid = str(i)
        pred = algo.predict(uid,iid).est
        all_results[iid] = pred
    sorted_list = sorted(all_results.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
    for i in range(n):
        print(sorted_list[i])
        res.append(sorted_list[i][0])
    return res


# filtering books data from selected age range
def filter_data_by_age(age, u_df, r_df, b_df):
    if age == '0-18':
        get_uid = u_df.loc[(u_df['age'] < 18)]['uid']
    elif age == '18-30':
        get_uid = u_df.loc[(u_df['age'] >= 18) & (u_df['age'] < 30)]['uid']
    elif age == '30-40':
        get_uid = u_df.loc[(u_df['age'] >= 30) & (u_df['age'] < 40)]['uid']
    elif age == '40-50':
        get_uid = u_df.loc[(u_df['age'] >= 40) & (u_df['age'] < 50)]['uid']
    else:
        get_uid = u_df.loc[u_df['age'] >= 50]['uid']
    
    get_iid = r_df.loc[r_df['uid'].isin(get_uid)]['iid']
    get_books = b_df.loc[(b_df['iid'].isin(get_iid)) & (b_df['year']>1995)]
        
    return get_books.drop_duplicates(), get_iid, get_uid


# Method 1 - like
def get_similar_items(iid, n=5):
    global algo
    inner_id = algo.trainset.to_inner_iid(str(iid))
    print("=================inner_id",inner_id)
    neighbors = algo.get_neighbors(inner_id, k=n)
    neighbors_iid = [algo.trainset.to_raw_iid(x) for x in neighbors]
    print(neighbors_iid)
    return neighbors_iid


# Method 2 - rate
def list_add(books: List):
    print("===========selected_data size:",len(selected_data))
    new_data = selected_data.copy(deep=True)
    for book in books:
        if book['score'] > 0:
            new_data = new_data.append({'uid':300000,'iid':book['iid'], 'rating':book['score']*2}, ignore_index=True)
    update_data = Dataset.load_from_df(new_data[["uid", "iid", "rating"]], Reader(rating_scale=(1, 10)))
    trainset_2 = update_data.build_full_trainset()
    return trainset_2



 