import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def extract_quarter(date_str):
    date = pd.to_datetime(date_str).normalize()
    return f"{date.year}Q{date.quarter}"

def get_embedding(article_df):
    article_df['writedAt_n'] = article_df['writedAt'].apply(extract_quarter)
    article_df['scrapedAt_n'] = article_df['scrapedAt'].apply(extract_quarter)
    article_df['contents'] = article_df['title'] + article_df['summarized']

    article_df.drop(['url','origin','writer','title','writedAt','scrapedAt','summarized'],axis=1,inplace=True)

    # 원-핫 인코더 초기화 및 범주형 데이터 변환
    encoder = OneHotEncoder(sparse=False)
    encoded_categories = encoder.fit_transform(article_df[['writedAt_n', 'scrapedAt_n']])
    df_encoded = pd.DataFrame(encoded_categories, columns=encoder.get_feature_names_out(['writedAt_n', 'scrapedAt_n']))
    article_df_emb = pd.concat([article_df, df_encoded], axis=1)
    article_df_emb.drop(['writedAt_n','scrapedAt_n'],axis=1,inplace=True)

    return article_df_emb
