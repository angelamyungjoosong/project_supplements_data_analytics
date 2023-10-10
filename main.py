# conda install -c conda-forge fastapi uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from konlpy.tag import Okt  
import numpy as np
app = FastAPI()
okt = Okt()

# No 'Access-Control-Allow-Origin'
# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영 환경에서는 접근 가능한 도메인만 허용하는 것이 좋습니다.
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the 'resources' directory as '/images' endpoint
from fastapi.staticfiles import StaticFiles
# app.mount("/images", StaticFiles(directory="resources"), name="images") 
# EX) http://127.0.0.1:8000/images/thermometer.png
# <img src='http://127.0.0.1:8000/images/thermometer.png' /> # view
# <a href='http://127.0.0.1:8000/images/thermometer.png'> thermometer.png</a> # download

@app.get("/")
async def root():
    return {"message": "Hello World"}

import pickle

# /api_v1/mlmodelwithregression with dict params
# method : post
# {
#     "texture_mean": 18.5,
#     "perimeter_mean": 102.1
# }
@app.post('/api_v1/mlmodelwithregression') 
def mlmodelwithregression(data:dict) : # json
    print('data with dict {}'.format(data))

    # data dict to 변수 활당
    comment = data['comment']

    def tokenize_sentence(comment):
        tokens = okt.morphs(comment)
        return ' '.join(tokens)
    
    tokenized_comment = tokenize_sentence(comment)

    
    # 벡터화시키기 
    feature_vectorized = 0
    with open('data/web_sentimental_tfidfVectorizer.pkl', 'rb') as tfidfVectorizer_file:
        tfidfVectorizer = pickle.load(tfidfVectorizer_file)
        input_labels = [tokenized_comment] # 학습했던 설명변수 형식 맞게 적용
        feature_vectorized = tfidfVectorizer.transform(input_labels) # 설명변수 백터화 
        np_feature_vectorized = np.array(feature_vectorized.toarray()) 
        print(np_feature_vectorized)
        pass


    # 학습 모델 불러와 예측
    with open('data/web_sentimental_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
        result_predict = model.predict(np_feature_vectorized)
        # print(result_predict)
        pass


    # 예측값 리턴
    result = {result_predict[0]}
    print(result)
    return result
