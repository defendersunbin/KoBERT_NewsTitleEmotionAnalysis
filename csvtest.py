import pandas as pd
from konlpy.tag import Kkma

# csv
df = pd.read_csv('csv/sk하이닉스.csv')

# 꼬꼬마 객체 생성
kkma = Kkma()

# 형태소 분석 함수
def tokenize(text):
    return kkma.morphs(text)

# content 열에 대해 형태소 분석 적용
df['content'] = df['content'].apply(tokenize)

# csv 파일로 저장
df.to_csv('csv/sk하이닉스_주가뉴스.csv', index=False)

# 결과 확인
print(df['content'])
