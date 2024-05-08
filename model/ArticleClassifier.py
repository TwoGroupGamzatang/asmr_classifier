import torch
import torch.nn as nn
from transformers import BertModel

class ArticleClassifier(nn.Module):
    def __init__(self, num_classes, num_date_features):
        super(ArticleClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("monologg/kobert")
        hidden_size = self.bert.config.hidden_size
        
        # 최종 분류
        self.classifier = nn.Linear(hidden_size + 1 + num_date_features, num_classes)

    def forward(self, input_ids, attention_mask, origin_len, writedAt, scrapedAt):
        # BERT 모델을 통해 텍스트를 처리
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.pooler_output
        
        # 아티클 길이를 tensor로 처리
        log_length_features = torch.log(origin_len.float() + 1).unsqueeze(1)
        
        # 날짜 정보 처리
        date_features = torch.cat((writedAt, scrapedAt), dim=1)
        
        # 모든 특징을 결합
        combined_features = torch.cat((sequence_output, log_length_features, date_features), dim=1)
        
        # 분류기를 통해 최종 예측 수행
        logits = self.classifier(combined_features)
        
        return logits
