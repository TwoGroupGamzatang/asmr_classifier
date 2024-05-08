import os
from openai import OpenAI

client = OpenAI(
    
    api_key="", #사용할 API키 입력
)

def summarize_text(document):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "You are an expert in article summarization. When you receive the article content, you can analyze the article content, summarize it, and show it to the user. Now, take the article as input from me and perform the following. Please answer in Korean. 1) Title Article Title 2) Tag extraction Extract up to 3 tags to determine which category the article falls into 3) Overall summary Summarize it into a length that takes about minutes to read.",
            },
            {
                "role": "assistant",
                "content": "물론이죠, 요약할 아티클을 알려주세요."
            },
            
            {
                "role": "user",
                "content": document +'내가 제시한 형식으로 1분 요약해줘'
            }
        ],
        model="gpt-3.5-turbo",
    )
    answer = chat_completion.choices[0].message.content
    return answer
