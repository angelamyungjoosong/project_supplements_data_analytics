
import numpy as np
import pandas as pd
import warnings # 경고 메시지 무시

from konlpy.tag import Okt
okt = Okt()
from mecab import MeCab
mecab = MeCab()


df_replace_list = pd.read_csv('datasets/replace_list.csv')
#print(df_replace_list[:3])


def replace_word(tokenized_review):
    replaced_tokens = []  # 치환된 단어들을 저장할 리스트
    for i in range(len(df_replace_list['before_replacement'])):
        try:
            # 치환할 단어가 있는 경우에만 데이터 치환 수행
            if df_replace_list['before_replacement'][i] in tokenized_review:
                replaced_token = tokenized_review.replace(df_replace_list['before_replacement'][i], df_replace_list['after_replacement'][i])
                pass
                replaced_tokens.append(replaced_token)
        except Exception as e:
            print(f"Error 발생 / 에러명: {e}")
    return replaced_tokens

## 확인용 
print(replace_word('다욧트 stess 살이 빠지다 피치 달달 단맛'))
# In[61]:

#print(replaced_review)



