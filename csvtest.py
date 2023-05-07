import pandas as pd
from konlpy.tag import Hannanum

# csv
df = pd.read_csv('csv/sk하이닉스_20230506_03시34분09초.csv')

# 한나눔 객체 생성
hannanum = Hannanum()

# 형태소 분석 함수
def tokenize(text):
    return hannanum.morphs(text)

# content 열에 대해 형태소 분석 적용
df['content'] = df['content'].apply(tokenize)

# csv 파일로 저장
df.to_csv('csv/sk하이닉스_형태소분석_20230506_03시34분09초.csv', index=False)

# 결과 확인
print(df['content'])
