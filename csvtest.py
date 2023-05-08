import pandas as pd
from konlpy.tag import Komoran
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# csv
df = pd.read_csv('csv/sk하이닉스.csv')

# KoELECTRA 모델 로드
model_name = "hyunwoongko/kobart"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 형태소 분석 함수
komoran = Komoran()
def tokenize(text):
    return komoran.morphs(text)

# 제목과 내용을 합쳐서 content 열 생성
df['content'] = df['title']

# content 열에 대해 형태소 분석 적용
df['content'] = df['content'].apply(tokenize)

# 감성 분석을 위한 전처리 함수
def preprocess(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    inputs.to(device)
    return inputs

# 예측 함수
def predict(inputs):
    outputs = model(**inputs)
    logits = outputs.logits
    probs = logits.softmax(dim=-1)
    return probs[0].detach().cpu().numpy()

# content 열에 대해 예측 수행
df['sentiment'] = df['content'].apply(lambda x: predict(preprocess(' '.join(x))))

# 감성 분석 결과를 문자 형식으로 변환
def convert_sentiment(probs):
    if probs[0] <= 0.49:
        return '부정'
    else:
        return '긍정'

df['sentiment'] = df['sentiment'].apply(convert_sentiment)

# csv 파일로 저장
df.to_csv('csv/sk하이닉스_주가뉴스(KoBart season2).csv', index=False)

# 결과 확인
print(df['content'])
print(df['sentiment'])
