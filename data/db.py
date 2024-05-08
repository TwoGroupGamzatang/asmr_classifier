from pymongo import MongoClient
import pandas as pd

def getdata(url, save_to_csv=False):
    # MongoDB 클라이언트 
    client = MongoClient(url)
    db = client['scraper']
    collection = db['scrapedcontents']

    # 데이터 조회 
    documents = list(collection.find())

    client.close()

    # DataFrame
    df = pd.DataFrame(documents)
    df.drop(['_id', 'userId', '__v'], axis=1, inplace=True)

    # save
    if save_to_csv:
        df.to_csv("scraped_contents.csv", index=False, encoding='utf-8-sig')
        print("Data saved to 'scraped_contents.csv'")

    return df
