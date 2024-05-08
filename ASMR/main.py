from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from torch.optim import Adam
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
from model import ArticleClassifier, ArticleDataset
from data import get_embedding

# 장치 설정(GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

article_df = pd.read_excel('summarized_article_final.xlsx')

article_df_emb = get_embedding(article_df)
# 데이터셋 및 DataLoader 초기화
tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
# 데이터 분할
train_val, test_data = train_test_split(article_df_emb, test_size=10, random_state=42)
train_data, val_data = train_test_split(train_val, test_size=5, random_state=42)

# 각 데이터셋 별 DataLoader 생성
train_dataset = ArticleDataset(train_data,tokenizer)
val_dataset = ArticleDataset(val_data,tokenizer)
test_dataset = ArticleDataset(test_data,tokenizer)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

# 모델 및 옵티마이저 초기화
model = ArticleClassifier(num_classes=4, num_date_features=7+1).to(device)  # +1 for scrapedAt_n_2024Q2
optimizer = Adam(model.parameters(), lr=1e-5)

# 학습
for epoch in range(100):
    model.train()
    total_loss = 0
    for text, lengths, writedAt, scrapedAt, targets in train_loader:
        # 텍스트 토크나이징
        encoded_texts = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        input_ids = encoded_texts['input_ids'].to(device)
        attention_mask = encoded_texts['attention_mask'].to(device)

        # 기타 데이터를 GPU로 이동
        lengths = lengths.float().to(device)  # Assuming lengths is already a tensor
        writedAt = writedAt.float().to(device)  # Assuming writedAt is already a tensor
        scrapedAt = scrapedAt.float().to(device)  # Assuming scrapedAt is already a tensor
        targets = targets.to(device, dtype=torch.long)

        # 모델의 zero_grad 호출
        optimizer.zero_grad()

        # 모델 실행
        outputs = model(input_ids, attention_mask, lengths, writedAt, scrapedAt)

        # 손실 계산 및 역전파
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")


    # 검증 및 테스트 루프 예시
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for text, lengths, writedAt, scrapedAt, targets in val_loader:
            # 텍스트 토크나이징 및 GPU로 이동
            encoded_texts = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            input_ids = encoded_texts['input_ids'].to(device)
            attention_mask = encoded_texts['attention_mask'].to(device)

            # 기타 데이터를 float32로 변환하고 GPU로 이동
            lengths = lengths.float().to(device)  # Assuming lengths is already a tensor
            writedAt = writedAt.float().to(device)  # Assuming writedAt is already a tensor
            scrapedAt = scrapedAt.float().to(device)  # Assuming scrapedAt is already a tensor
            targets = targets.to(device, dtype=torch.long)

            # 모델 실행
            outputs = model(input_ids, attention_mask, lengths, writedAt, scrapedAt)
            predicted_classes = torch.argmax(F.softmax(outputs, dim=1), dim=1)
            correct += (predicted_classes == targets).sum().item()
            total += targets.size(0)
        print(f"Validation Accuracy: {correct / total:.2f}")


# 테스트
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for text, lengths, writedAt, scrapedAt, targets in test_loader:
        # 데이터를 GPU로 이동
        encoded_texts = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        input_ids = encoded_texts['input_ids'].to(device)
        attention_mask = encoded_texts['attention_mask'].to(device)

        lengths = lengths.float().to(device)  # Assuming lengths is already a tensor
        writedAt = writedAt.float().to(device)  # Assuming writedAt is already a tensor
        scrapedAt = scrapedAt.float().to(device)  # Assuming scrapedAt is already a tensor
        targets = targets.to(device, dtype=torch.long)

        outputs = model(input_ids, attention_mask, lengths, writedAt, scrapedAt)
        predicted_classes = torch.argmax(F.softmax(outputs, dim=1), dim=1)
        correct += (predicted_classes == targets).sum().item()
        total += targets.size(0)
    print(f"Test Accuracy: {correct / total:.2f}")

