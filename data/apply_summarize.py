import pandas as pd
from utils import summarize_text


def preprocess(article_df, save_to_csv = False):

    article_df['summarized'] = None
    article_df['origin_len'] = article_df['content'].apply(len)

    for i, row in article_df.iterrows():
        result = summarize_text(row['content'])
        article_df.at[i, 'summarized'] = result

    article_df.drop('content',axis=1,inplace=True)
    
    if save_to_csv:
        article_df.to_csv('summarized_article_final.csv', index=False, encoding='utf-8-sig')
    return article_df
