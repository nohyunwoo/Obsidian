### ✅ **KoBERT 7 감정 분류**

- **장점**:
    
    - 리소스 매우 적음 (랩탑/로컬 서버 가능)
        
    - 실시간 채팅 감정 분류에 적합
        
    - 배포/서버 운영 시 안정적
        
- **단점**: 대화 맥락을 깊이 이해하진 못함

https://complexoftaste.tistory.com/2

https://github.com/SKTBrain/KoBERT

-> 라이브러리 충돌로 코랩에서 환경세팅이 되지 않았음
-> 다른 모델을 모색

----
## **DistilBERT** 모델
**에크만의 6가지 기본 감정**: 기쁨, 슬픔, 분노, 공포, 놀람, 혐오 + 중립
-> 7가지
https://huggingface.co/docs/transformers/main/ko/tasks/sequence_classification

# DistilBERT 기초부터 확장까지 완전 가이드

## 1. DistilBERT란 무엇인가?

### 1.1 기본 개념 이해하기

**DistilBERT**는 원래 BERT(Bidirectional Encoder Representations from Transformers)라는 모델의 경량화 버전입니다. 쉽게 말해, BERT의 성능은 거의 유지하면서 크기와 속도를 개선한 모델입니다.

- **BERT**: 구글에서 개발한 자연어 처리 모델로, 문맥을 이해하는 양방향(Bidirectional) 학습 방식을 사용합니다.
- **DistilBERT**: Hugging Face에서 개발한 모델로, 원래 BERT보다 40% 작고 60% 빠르면서도 성능은 97% 수준을 유지합니다.

### 1.2 DistilBERT의 장점

1. **가벼움**: 더 적은 컴퓨팅 자원으로 실행 가능
2. **빠름**: 학습과 추론 속도가 모두 빠름
3. **효율적**: 적은 자원으로 좋은 성능 제공
4. **다양한 언어 지원**: 다국어 버전 제공

## 2. 자연어 처리(NLP) 기초 개념

### 2.1 텍스트 처리의 기본 단계

1. **토큰화(Tokenization)**: 텍스트를 작은 단위(토큰)로 나누는 과정
    
    - 예: "안녕하세요" → ["안녕", "하세요"] 또는 ["안", "녕", "하", "세", "요"]
2. **인코딩(Encoding)**: 토큰을 숫자로 변환하는 과정
    
    - 컴퓨터는 텍스트를 직접 이해할 수 없어 숫자로 변환합니다.
3. **임베딩(Embedding)**: 단어를 의미있는 벡터 공간으로 매핑
    
    - 의미적으로 비슷한 단어는 벡터 공간에서 가깝게 위치합니다.

### 2.2 언어 모델의 개념

언어 모델은 텍스트의 패턴을 학습하여 다음에 올 단어를 예측하거나 텍스트를 이해하는 AI 시스템입니다. DistilBERT도 이러한 언어 모델 중 하나입니다.

## 3. 실전: Google Colab으로 시작하기

### 3.1 Google Colab이란?

Google Colab은 브라우저에서 Python 코드를 작성하고 실행할 수 있는 무료 서비스입니다. GPU를 무료로 사용할 수 있어 딥러닝 작업에 적합합니다.

### 3.2 Colab 접속 및 설정

1. [Google Colab](https://colab.research.google.com/) 접속
2. 새 노트북 생성
3. GPU 설정: 메뉴 → 런타임 → 런타임 유형 변경 → 하드웨어 가속기: GPU 선택

## 4. DistilBERT 사용을 위한 환경 설정

### 4.1 필요한 라이브러리 설치

아래 코드를 Colab에 입력하고 실행합니다:

```python
# Transformers 라이브러리와 기타 필요한 패키지 설치
!pip install transformers datasets torch scikit-learn pandas matplotlib seaborn
```

### 4.2 기본 라이브러리 임포트

```python
import numpy as np
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
```

## 5. 감정 분석의 기초

### 5.1 감정 분석이란?

감정 분석(Sentiment Analysis)은 텍스트에서 작성자의 감정, 태도, 의견 등을 파악하는 NLP 기술입니다. 흔히 텍스트가 긍정적인지, 부정적인지, 중립적인지 등을 분류합니다.

### 5.2 감정 분류 방법

1. **이진 분류**: 긍정 vs 부정
2. **삼중 분류**: 긍정 vs 중립 vs 부정
3. **다중 분류**: 기쁨, 슬픔, 분노, 공포, 놀람, 혐오 등 여러 감정으로 분류

### 5.3 간단한 감정 분석 예제 데이터

```python
# 샘플 데이터 생성 (실제 프로젝트에서는 실제 데이터셋을 사용하세요)
texts = [
    "오늘은 정말 행복한 하루였어요. 모든 일이 잘 풀렸습니다.",
    "친구가 떠나버렸습니다. 너무 슬프네요.",
    "왜 약속을 지키지 않는 거야? 정말 화가 나!",
    "밤에 혼자 집에 있는데 이상한 소리가 들려서 무서워요.",
    "갑자기 뒤에서 소리가 나서 깜짝 놀랐어요!",
    "이 음식 맛이 너무 이상해서 먹을 수가 없어요."
]

# 감정 라벨 (0: 기쁨, 1: 슬픔, 2: 분노, 3: 공포, 4: 놀람, 5: 혐오)
labels = [0, 1, 2, 3, 4, 5]

# 감정 매핑 딕셔너리
emotions = {
    0: "기쁨",
    1: "슬픔",
    2: "분노", 
    3: "공포",
    4: "놀람",
    5: "혐오"
}

# 데이터프레임 생성
df = pd.DataFrame({
    'text': texts,
    'label': labels
})

print(df)
```

## 6. DistilBERT 모델 파헤치기

### 6.1 모델 종류와 선택

DistilBERT에는 여러 버전이 있습니다:

1. **distilbert-base-uncased**: 영어 전용, 소문자만 처리
2. **distilbert-base-cased**: 영어 전용, 대소문자 구분
3. **distilbert-base-multilingual-cased**: 다국어 지원(한국어 포함)

한국어를 처리하려면 다국어 버전을 사용하거나 한국어 특화 모델(KoELECTRA, KoBERT 등)을 고려할 수 있습니다.

### 6.2 토크나이저와 모델 로드

```python
# 다국어 DistilBERT 모델 로드 (한국어 처리 가능)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-multilingual-cased',
    num_labels=len(emotions),  # 분류할 감정 개수
)

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

### 6.3 토큰화 과정 살펴보기

```python
# 토큰화 예제
text = "안녕하세요. DistilBERT를 배워봅시다!"
encoded = tokenizer.encode_plus(
    text,
    add_special_tokens=True,
    max_length=128,
    padding='max_length',
    truncation=True,
    return_attention_mask=True,
    return_tensors='pt'
)

print("입력 ID:", encoded['input_ids'])
print("어텐션 마스크:", encoded['attention_mask'])
print("디코딩된 토큰:", tokenizer.convert_ids_to_tokens(encoded['input_ids'][0]))
```

## 7. 데이터 전처리 핵심 이해하기

### 7.1 데이터 분할

```python
# 훈련 및 검증 세트 분리
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].values, 
    df['label'].values, 
    test_size=0.2,  # 20%는 검증용
    random_state=42,  # 결과 재현성을 위한 시드 설정
    stratify=df['label'].values  # 각 클래스 비율 유지
)
```

### 7.2 텍스트 전처리 함수

```python
# 텍스트를 모델 입력 형식으로 변환하는 함수
def preprocess_data(texts, labels, tokenizer, max_len=128):
    input_ids = []
    attention_masks = []
    
    for text in texts:
        # 텍스트를 토큰화하고 모델 입력 형식으로 변환
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # [CLS], [SEP] 같은 특수 토큰 추가
            max_length=max_len,       # 최대 길이 설정
            padding='max_length',     # 패딩 추가
            truncation=True,          # 최대 길이 초과시 자르기
            return_attention_mask=True,  # 어텐션 마스크 반환
            return_tensors='pt',      # PyTorch 텐서 형식으로 반환
        )
        
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    
    # 텐서 형태로 변환
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    
    return input_ids, attention_masks, labels
```

### 7.3 데이터 로더 생성

```python
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# 데이터 전처리
train_input_ids, train_attention_masks, train_labels = preprocess_data(
    train_texts, train_labels, tokenizer
)
val_input_ids, val_attention_masks, val_labels = preprocess_data(
    val_texts, val_labels, tokenizer
)

# 배치 크기 설정
batch_size = 16

# 훈련 데이터 로더
train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
train_dataloader = DataLoader(
    train_dataset,
    sampler=RandomSampler(train_dataset),  # 데이터 무작위 샘플링
    batch_size=batch_size
)

# 검증 데이터 로더
val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)
val_dataloader = DataLoader(
    val_dataset,
    sampler=SequentialSampler(val_dataset),  # 순차적 샘플링
    batch_size=batch_size
)
```

## 8. 모델 학습 과정 이해하기

### 8.1 옵티마이저와 스케줄러 설정

```python
from transformers import AdamW, get_linear_schedule_with_warmup

# 옵티마이저 설정
optimizer = AdamW(model.parameters(),
                  lr=2e-5,  # 학습률
                  eps=1e-8   # 안정성을 위한 작은 값
                 )

# 에폭(전체 데이터셋을 학습하는 횟수) 설정
epochs = 4

# 총 학습 스텝 계산
total_steps = len(train_dataloader) * epochs

# 학습률 스케줄러 설정 (학습 과정에서 학습률을 조정)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,  # 워밍업 스텝 수
    num_training_steps=total_steps
)
```

### 8.2 간단한 학습 루프

```python
import time
from tqdm.notebook import tqdm

# 모델 훈련 함수
def train_model(model, train_dataloader, val_dataloader, epochs, optimizer, scheduler, device):
    # 시작 시간 기록
    total_start_time = time.time()
    
    # 각 에폭별 훈련
    for epoch in range(epochs):
        print(f'\n======== 에폭 {epoch + 1} / {epochs} ========')
        
        # 훈련 모드로 설정
        model.train()
        total_loss = 0
        
        # 훈련 데이터로더를 통한 배치 학습
        for batch in tqdm(train_dataloader, desc="훈련 중"):
            # 배치 데이터를 디바이스로 이동
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            
            # 그래디언트 초기화
            model.zero_grad()
            
            # 모델 포워드 패스
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # 역전파
            loss.backward()
            
            # 그래디언트 클리핑 (폭발 방지)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # 파라미터 업데이트
            optimizer.step()
            
            # 학습률 업데이트
            scheduler.step()
        
        # 평균 훈련 손실 계산
        avg_train_loss = total_loss / len(train_dataloader)
        print(f'에폭 {epoch + 1} 평균 훈련 손실: {avg_train_loss:.4f}')
        
        # 검증 수행
        model.eval()  # 평가 모드로 설정
        val_accuracy = evaluate_model(model, val_dataloader, device)
        print(f'검증 정확도: {val_accuracy:.4f}')
    
    print(f'\n훈련 완료! 총 훈련 시간: {(time.time() - total_start_time) / 60:.2f}분')
    return model

# 모델 평가 함수
def evaluate_model(model, dataloader, device):
    correct_predictions = 0
    total_predictions = 0
    
    # 그래디언트 계산 없이 실행
    with torch.no_grad():
        for batch in dataloader:
            # 배치 데이터를 디바이스로 이동
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            
            # 모델 출력
            outputs = model(
                input_ids,
                attention_mask=attention_mask
            )
            
            # 예측값 계산
            _, predictions = torch.max(outputs.logits, dim=1)
            
            # 정확도 계산
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
    
    return correct_predictions / total_predictions

# 모델 학습 실행
trained_model = train_model(
    model, 
    train_dataloader, 
    val_dataloader, 
    epochs, 
    optimizer, 
    scheduler, 
    device
)
```

## 9. 모델 평가 및 결과 시각화

### 9.1 혼동 행렬 생성

```python
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 예측 수행
def get_predictions(model, dataloader, device):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            
            predictions.extend(preds.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())
    
    return predictions, true_labels

# 예측 실행
predictions, true_labels = get_predictions(model, val_dataloader, device)

# 혼동 행렬 생성
cm = confusion_matrix(true_labels, predictions)

# 혼동 행렬 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=list(emotions.values()),
    yticklabels=list(emotions.values())
)
plt.title('혼동 행렬')
plt.ylabel('실제 레이블')
plt.xlabel('예측 레이블')
plt.show()
```

### 9.2 분류 보고서 생성

```python
from sklearn.metrics import classification_report

# 분류 보고서 출력
print("분류 보고서:")
print(classification_report(
    true_labels, 
    predictions, 
    target_names=list(emotions.values())
))
```

## 10. 새로운 텍스트 예측하기

```python
# 예측 함수
def predict_emotion(text, model, tokenizer, emotions_dict, device):
    # 모델을 평가 모드로 설정
    model.eval()
    
    # 텍스트 토큰화
    encoded_dict = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    # 텐서를 디바이스로 이동
    input_ids = encoded_dict['input_ids'].to(device)
    attention_mask = encoded_dict['attention_mask'].to(device)
    
    # 예측 실행
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    # 예측 결과 추출
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    # 감정 확률 계산
    probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
    probs_dict = {emotions_dict[i]: float(probabilities[i]) * 100 for i in range(len(emotions_dict))}
    
    return emotions_dict[predicted_class], probs_dict

# 예측 테스트
test_text = "오늘 시험에서 좋은 성적을 받아서 너무 기분이 좋아요!"
emotion, probs = predict_emotion(test_text, model, tokenizer, emotions, device)

print(f"텍스트: {test_text}")
print(f"예측된 감정: {emotion}")
print("감정별 확률:")
for emotion, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
    print(f"  - {emotion}: {prob:.2f}%")
```

## 11. 모델 저장 및 불러오기

```python
# 모델 저장
model_save_path = 'distilbert_emotion_model'
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f"모델이 '{model_save_path}' 경로에 저장되었습니다.")

# 모델 불러오기
loaded_model = DistilBertForSequenceClassification.from_pretrained(model_save_path)
loaded_tokenizer = DistilBertTokenizer.from_pretrained(model_save_path)

loaded_model.to(device)  # GPU로 이동
```

## 12. 확장 방법: 실제 프로젝트로 발전시키기

### 12.1 더 많은 데이터로 확장하기

실제 프로젝트에서는 더 많은 데이터가 필요합니다. 한국어 감정 분석에 사용할 수 있는 데이터셋:

- **AI Hub**: 한국어 감성 대화 말뭉치
- **NSMC**: 네이버 영화 리뷰 데이터셋
- **Korean HateSpeech Dataset**: 혐오 표현 데이터셋

```python
# CSV 파일에서 데이터 로드하는 예시
df = pd.read_csv('your_dataset.csv')
texts = df['text'].tolist()
labels = df['label'].tolist()
```

### 12.2 하이퍼파라미터 튜닝

모델 성능을 향상시키는 주요 방법입니다:

```python
# 하이퍼파라미터 변경 예시
batch_size = 32  # 배치 크기 증가
learning_rate = 5e-5  # 학습률 조정
epochs = 6  # 에폭 수 증가
```

### 12.3 교차 검증 적용하기

```python
from sklearn.model_selection import KFold

# 5-폴드 교차 검증 예시
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(texts)):
    print(f"===== 폴드 {fold + 1}/5 =====")
    
    # 폴드별 데이터 분할
    fold_train_texts = [texts[i] for i in train_idx]
    fold_train_labels = [labels[i] for i in train_idx]
    fold_val_texts = [texts[i] for i in val_idx]
    fold_val_labels = [labels[i] for i in val_idx]
    
    # 이 폴드에 대한 모델 학습 및 평가
    # (앞서 정의한 코드 활용)
```

### 12.4 다양한 언어 모델 시도하기

다양한 언어 모델을 시도하여 성능을 비교할 수 있습니다:

- **KoBERT**: 한국어에 최적화된 BERT 모델
- **KoELECTRA**: 한국어 특화 ELECTRA 모델
- **KoGPT2**: 한국어 GPT-2 모델

```python
# 다른 모델 사용 예시 (KoBERT)
# !pip install git+https://git@github.com/SKTBrain/KoBERT.git
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

# KoBERT 모델 로드
bertmodel, vocab = get_pytorch_kobert_model()
```

### 12.5 감정 분석 웹 애플리케이션 만들기

학습한 모델을 웹 애플리케이션으로 배포할 수 있습니다:

- **Streamlit**: 간단한 웹 애플리케이션 생성
- **Flask/Django**: 더 복잡한 웹 서비스 개발
- **API 서비스**: REST API로 모델 서비스 제공

```python
# Streamlit을 이용한 간단한 웹 앱 예시
# !pip install streamlit
import streamlit as st

st.title("감정 분석 앱")
user_input = st.text_area("분석할 텍스트를 입력하세요", "여기에 텍스트를 입력하세요.")

if st.button("감정 분석하기"):
    emotion, probs = predict_emotion(user_input, model, tokenizer, emotions, device)
    st.write(f"예측된 감정: **{emotion}**")
    
    # 확률 막대 차트 표시
    for emotion, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        st.write(f"{emotion}: {prob:.2f}%")
        st.progress(int(prob / 100))
```

## 13. 실전 팁과 주의사항

### 13.1 데이터 품질 관리

- **균형 잡힌 데이터셋**: 각 감정 클래스가 균등하게 분포되어야 합니다
- **데이터 증강**: 부족한 클래스의 데이터를 증강하는 기법 활용
- **정제**: 노이즈 데이터 제거 및 전처리

### 13.2 모델 성능 최적화

- **학습률 스케줄링**: 학습 과정에서 학습률을 동적으로 조절
- **조기 종료(Early Stopping)**: 과적합 방지를 위해 검증 성능이 향상되지 않으면 학습 중단
- **앙상블 기법**: 여러 모델의 예측을 결합하여 성능 향상

### 13.3 배포 고려사항

- **모델 압축**: 추론 속도 향상을 위한 모델 양자화/가지치기
- **배치 처리**: 대량 텍스트 처리 시 배치 단위로 처리
- **캐싱**: 반복적인 요청에 대한 결과 캐싱

## 14. 추가 학습 자료

### 14.1 추천 도서 및 온라인 강의

- **책**: "Natural Language Processing with Transformers"
- **강의**:
    - 스탠포드 대학교 CS224N (Natural Language Processing with Deep Learning)
    - Hugging Face 코스 (https://huggingface.co/course)

### 14.2 유용한 라이브러리 및 도구

- **Transformers**: 다양한 트랜스포머 모델 제공
- **Datasets**: 데이터셋 관리 및 처리
- **Weights & Biases**: 실험 추적 및 시각화
- **TensorBoard**: 학습 과정 모니터링

## 15. 결론

DistilBERT는 효율적이면서도 강력한 자연어 처리 모델로, 감정 분석과 같은 분류 작업에 탁월한 성능을 발휘합니다. 이 가이드를 통해 기본 개념부터 실전 구현까지 배웠으며, 이제 자신만의 감정 분석 프로젝트를 시작할 준비가 되었습니다!

실습을 통해 개념을 확실히 이해하고, 점진적으로 모델과 데이터셋을 확장해 나가면서 성능을 향상시켜 보세요. 자연어 처리의 세계는 광범위하고 빠르게 발전하고 있으며, DistilBERT는 그 여정을 시작하기에 완벽한 모델입니다.



{'loss': 1.73, 'grad_norm': 5.929318904876709, 'learning_rate': 9e-07, 'epoch': 0.01}     
{'loss': 1.7015, 'grad_norm': 5.57500696182251, 'learning_rate': 1.9e-06, 'epoch': 0.02}  
{'loss': 1.6544, 'grad_norm': 5.874943733215332, 'learning_rate': 2.9e-06, 'epoch': 0.03}  
{'loss': 1.5608, 'grad_norm': 6.219969749450684, 'learning_rate': 3.9e-06, 'epoch': 0.04}  
{'loss': 1.3226, 'grad_norm': 7.1138410568237305, 'learning_rate': 4.9000000000000005e-06, 'epoch': 0.05}
{'loss': 0.923, 'grad_norm': 6.102278232574463, 'learning_rate': 5.9e-06, 'epoch': 0.06}   
{'loss': 0.568, 'grad_norm': 4.2255377769470215, 'learning_rate': 6.900000000000001e-06, 'epoch': 0.07}
{'loss': 0.2971, 'grad_norm': 2.5310873985290527, 'learning_rate': 7.9e-06, 'epoch': 0.08} 
{'loss': 0.1829, 'grad_norm': 1.5549311637878418, 'learning_rate': 8.9e-06, 'epoch': 0.09} 
{'loss': 0.0811, 'grad_norm': 0.9621895551681519, 'learning_rate': 9.900000000000002e-06, 'epoch': 0.1}
{'loss': 0.0451, 'grad_norm': 0.5688076019287109, 'learning_rate': 1.09e-05, 'epoch': 0.11}
{'loss': 0.0271, 'grad_norm': 0.36860567331314087, 'learning_rate': 1.19e-05, 'epoch': 0.12}
{'loss': 0.0178, 'grad_norm': 0.25229892134666443, 'learning_rate': 1.29e-05, 'epoch': 0.13}
{'loss': 0.0127, 'grad_norm': 0.2077009677886963, 'learning_rate': 1.3900000000000002e-05, 'epoch': 0.14}
{'loss': 0.0472, 'grad_norm': 0.16825085878372192, 'learning_rate': 1.49e-05, 'epoch': 0.15}
{'loss': 0.0082, 'grad_norm': 0.13425493240356445, 'learning_rate': 1.59e-05, 'epoch': 0.16}
{'loss': 0.0064, 'grad_norm': 0.12296118587255478, 'learning_rate': 1.69e-05, 'epoch': 0.17}
{'loss': 0.0055, 'grad_norm': 0.09465190023183823, 'learning_rate': 1.79e-05, 'epoch': 0.18}
{'loss': 0.0047, 'grad_norm': 0.09078194200992584, 'learning_rate': 1.8900000000000002e-05, 'epoch': 0.19}
{'loss': 0.0039, 'grad_norm': 0.07758123427629471, 'learning_rate': 1.9900000000000003e-05, 'epoch': 0.2}
{'loss': 0.0034, 'grad_norm': 0.061946555972099304, 'learning_rate': 2.09e-05, 'epoch': 0.21}
{'loss': 0.0029, 'grad_norm': 0.053750596940517426, 'learning_rate': 2.19e-05, 'epoch': 0.22}
{'loss': 0.0024, 'grad_norm': 0.05377698689699173, 'learning_rate': 2.29e-05, 'epoch': 0.23}
{'loss': 0.0021, 'grad_norm': 0.04732121154665947, 'learning_rate': 2.39e-05, 'epoch': 0.24}
{'loss': 0.0019, 'grad_norm': 0.035849329084157944, 'learning_rate': 2.4900000000000002e-05, 'epoch': 0.25}
{'loss': 0.0509, 'grad_norm': 0.03485892340540886, 'learning_rate': 2.5900000000000003e-05, 'epoch': 0.26}
{'loss': 0.0017, 'grad_norm': 0.03606022149324417, 'learning_rate': 2.6900000000000003e-05, 'epoch': 0.27}
{'loss': 0.0017, 'grad_norm': 0.03672884404659271, 'learning_rate': 2.7900000000000004e-05, 'epoch': 0.28}
{'loss': 0.0015, 'grad_norm': 0.0311703123152256, 'learning_rate': 2.8899999999999998e-05, 'epoch': 0.29}
{'loss': 0.0013, 'grad_norm': 0.026362774893641472, 'learning_rate': 2.9900000000000002e-05, 'epoch': 0.3}
{'loss': 0.0011, 'grad_norm': 0.02500220388174057, 'learning_rate': 3.09e-05, 'epoch': 0.31}
{'loss': 0.001, 'grad_norm': 0.022560181096196175, 'learning_rate': 3.19e-05, 'epoch': 0.32}
{'loss': 0.0009, 'grad_norm': 0.023742325603961945, 'learning_rate': 3.29e-05, 'epoch': 0.33}
{'loss': 0.1102, 'grad_norm': 0.035488273948431015, 'learning_rate': 3.3900000000000004e-05, 'epoch': 0.34}
{'loss': 0.0019, 'grad_norm': 0.04240161180496216, 'learning_rate': 3.49e-05, 'epoch': 0.35}
{'loss': 0.0017, 'grad_norm': 0.03292503207921982, 'learning_rate': 3.59e-05, 'epoch': 0.36}
{'loss': 0.0012, 'grad_norm': 0.026468027383089066, 'learning_rate': 3.69e-05, 'epoch': 0.37}
{'loss': 0.001, 'grad_norm': 0.020617667585611343, 'learning_rate': 3.79e-05, 'epoch': 0.38}
{'loss': 0.0008, 'grad_norm': 0.020711081102490425, 'learning_rate': 3.8900000000000004e-05, 'epoch': 0.39}
{'loss': 0.0007, 'grad_norm': 0.016900381073355675, 'learning_rate': 3.99e-05, 'epoch': 0.4}
{'loss': 0.0006, 'grad_norm': 0.013731124810874462, 'learning_rate': 4.09e-05, 'epoch': 0.41}
{'loss': 0.0005, 'grad_norm': 0.01412891410291195, 'learning_rate': 4.19e-05, 'epoch': 0.42}
{'loss': 0.0005, 'grad_norm': 0.01095452532172203, 'learning_rate': 4.29e-05, 'epoch': 0.43}
{'loss': 0.0004, 'grad_norm': 0.010435562580823898, 'learning_rate': 4.39e-05, 'epoch': 0.44}
{'loss': 0.0004, 'grad_norm': 0.011186657473444939, 'learning_rate': 4.49e-05, 'epoch': 0.46}
{'loss': 0.0004, 'grad_norm': 0.009442798793315887, 'learning_rate': 4.5900000000000004e-05, 'epoch': 0.47}
{'loss': 0.0603, 'grad_norm': 0.012440331280231476, 'learning_rate': 4.69e-05, 'epoch': 0.48}
{'loss': 0.0005, 'grad_norm': 0.014737531542778015, 'learning_rate': 4.79e-05, 'epoch': 0.49}
{'loss': 0.0569, 'grad_norm': 2.239046335220337, 'learning_rate': 4.89e-05, 'epoch': 0.5}  
{'loss': 0.0009, 'grad_norm': 0.02613440528512001, 'learning_rate': 4.99e-05, 'epoch': 0.51}
{'loss': 0.001, 'grad_norm': 0.021564818918704987, 'learning_rate': 4.9817592217267936e-05, 'epoch': 0.52}
{'loss': 0.0007, 'grad_norm': 0.015882669016718864, 'learning_rate': 4.96149169031212e-05, 'epoch': 0.53}
{'loss': 0.0005, 'grad_norm': 0.014196294359862804, 'learning_rate': 4.9412241588974466e-05, 'epoch': 0.54}
{'loss': 0.0004, 'grad_norm': 0.00909552350640297, 'learning_rate': 4.920956627482773e-05, 'epoch': 0.55}
{'loss': 0.0003, 'grad_norm': 0.008126534521579742, 'learning_rate': 4.900689096068099e-05, 'epoch': 0.56}
{'loss': 0.0003, 'grad_norm': 0.007794531993567944, 'learning_rate': 4.8804215646534255e-05, 'epoch': 0.57}
{'loss': 0.0002, 'grad_norm': 0.005364495795220137, 'learning_rate': 4.860154033238752e-05, 'epoch': 0.58}
{'loss': 0.0693, 'grad_norm': 0.005861700512468815, 'learning_rate': 4.839886501824078e-05, 'epoch': 0.59}
{'loss': 0.0003, 'grad_norm': 0.011378929950296879, 'learning_rate': 4.8196189704094045e-05, 'epoch': 0.6}
{'loss': 0.0005, 'grad_norm': 0.014733795076608658, 'learning_rate': 4.799351438994731e-05, 'epoch': 0.61}
{'loss': 0.0527, 'grad_norm': 0.02738226391375065, 'learning_rate': 4.779083907580057e-05, 'epoch': 0.62}
{'loss': 0.0011, 'grad_norm': 0.02634316496551037, 'learning_rate': 4.7588163761653834e-05, 'epoch': 0.63}
{'loss': 0.0007, 'grad_norm': 0.019840188324451447, 'learning_rate': 4.73854884475071e-05, 'epoch': 0.64}
{'loss': 0.0506, 'grad_norm': 0.06158585101366043, 'learning_rate': 4.718281313336036e-05, 'epoch': 0.65}
{'loss': 0.0013, 'grad_norm': 0.029528619721531868, 'learning_rate': 4.698013781921362e-05, 'epoch': 0.66}
{'loss': 0.0007, 'grad_norm': 0.016287654638290405, 'learning_rate': 4.677746250506688e-05, 'epoch': 0.67}
{'loss': 0.0004, 'grad_norm': 0.012722354382276535, 'learning_rate': 4.657478719092015e-05, 'epoch': 0.68}
{'loss': 0.0003, 'grad_norm': 0.009253652766346931, 'learning_rate': 4.637211187677341e-05, 'epoch': 0.69}
{'loss': 0.0003, 'grad_norm': 0.006261990871280432, 'learning_rate': 4.616943656262667e-05, 'epoch': 0.7}
{'loss': 0.0002, 'grad_norm': 0.006419126410037279, 'learning_rate': 4.596676124847994e-05, 'epoch': 0.71}
{'loss': 0.0617, 'grad_norm': 0.007021169178187847, 'learning_rate': 4.57640859343332e-05, 'epoch': 0.72}
{'loss': 0.0004, 'grad_norm': 0.012534063309431076, 'learning_rate': 4.556141062018646e-05, 'epoch': 0.73}
{'loss': 0.0005, 'grad_norm': 0.008515633642673492, 'learning_rate': 4.535873530603973e-05, 'epoch': 0.74}
{'loss': 0.0003, 'grad_norm': 0.007950382307171822, 'learning_rate': 4.515605999189299e-05, 'epoch': 0.75}
{'loss': 0.0003, 'grad_norm': 0.008659182116389275, 'learning_rate': 4.495338467774625e-05, 'epoch': 0.76}
{'loss': 0.1008, 'grad_norm': 0.027935044839978218, 'learning_rate': 4.4750709363599514e-05, 'epoch': 0.77}
{'loss': 0.0031, 'grad_norm': 0.05819813162088394, 'learning_rate': 4.454803404945278e-05, 'epoch': 0.78}
{'loss': 0.0481, 'grad_norm': 0.020698126405477524, 'learning_rate': 4.4345358735306045e-05, 'epoch': 0.79}
{'loss': 0.0012, 'grad_norm': 0.03365820273756981, 'learning_rate': 4.41426834211593e-05, 'epoch': 0.8}
{'loss': 0.048, 'grad_norm': 0.02586285211145878, 'learning_rate': 4.394000810701257e-05, 'epoch': 0.81}
{'loss': 0.0017, 'grad_norm': 0.06441807746887207, 'learning_rate': 4.3737332792865834e-05, 'epoch': 0.82}
{'loss': 0.0013, 'grad_norm': 0.02513931505382061, 'learning_rate': 4.353465747871909e-05, 'epoch': 0.83}
{'loss': 0.0007, 'grad_norm': 0.01887134090065956, 'learning_rate': 4.333198216457236e-05, 'epoch': 0.84}
{'loss': 0.0004, 'grad_norm': 0.011729791760444641, 'learning_rate': 4.312930685042562e-05, 'epoch': 0.85}
{'loss': 0.0003, 'grad_norm': 0.009116088040173054, 'learning_rate': 4.292663153627888e-05, 'epoch': 0.86}
{'loss': 0.0003, 'grad_norm': 0.0069864788092672825, 'learning_rate': 4.272395622213215e-05, 'epoch': 0.87}
{'loss': 0.0002, 'grad_norm': 0.006383085157722235, 'learning_rate': 4.252128090798541e-05, 'epoch': 0.88}
{'loss': 0.0518, 'grad_norm': 2.275294542312622, 'learning_rate': 4.231860559383867e-05, 'epoch': 0.89}
{'loss': 0.0004, 'grad_norm': 0.012878527864813805, 'learning_rate': 4.2115930279691936e-05, 'epoch': 0.9}
{'loss': 0.0007, 'grad_norm': 0.012145205400884151, 'learning_rate': 4.19132549655452e-05, 'epoch': 0.91}
{'loss': 0.0006, 'grad_norm': 0.018439456820487976, 'learning_rate': 4.171057965139846e-05, 'epoch': 0.92}
{'loss': 0.0608, 'grad_norm': 0.01992461457848549, 'learning_rate': 4.1507904337251725e-05, 'epoch': 0.93}
{'loss': 0.0009, 'grad_norm': 0.07102948427200317, 'learning_rate': 4.1305229023104983e-05, 'epoch': 0.94}
{'loss': 0.0008, 'grad_norm': 0.016921788454055786, 'learning_rate': 4.1102553708958255e-05, 'epoch': 0.95}
{'loss': 0.0006, 'grad_norm': 0.020931703969836235, 'learning_rate': 4.0899878394811514e-05, 'epoch': 0.96}
{'loss': 0.0004, 'grad_norm': 0.012226645834743977, 'learning_rate': 4.069720308066477e-05, 'epoch': 0.97}
{'loss': 0.0003, 'grad_norm': 0.011743330396711826, 'learning_rate': 4.0494527766518045e-05, 'epoch': 0.98}
{'loss': 0.0003, 'grad_norm': 0.008351798169314861, 'learning_rate': 4.02918524523713e-05, 'epoch': 0.99}
 33%|████████████████▋                                 | 989/2967 [32:40<55:51,  1.69s/it]C:\Users\nohyunwoo\AppData\Local\Programs\Python\Python313\Lib\site-packages\torch\utils\data\dataloader.py:665: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
  warnings.warn(warn_msg)
{'loss': 0.0003, 'grad_norm': 0.009627996943891048, 'learning_rate': 4.008917713822457e-05, 'epoch': 1.0}
{'loss': 0.0002, 'grad_norm': 0.005694256629794836, 'learning_rate': 3.988650182407783e-05, 'epoch': 1.01}
{'loss': 0.0002, 'grad_norm': 0.005790052469819784, 'learning_rate': 3.968382650993109e-05, 'epoch': 1.02}
{'loss': 0.0002, 'grad_norm': 0.004581131972372532, 'learning_rate': 3.948115119578436e-05, 'epoch': 1.03}
{'loss': 0.0002, 'grad_norm': 0.003942248411476612, 'learning_rate': 3.9278475881637616e-05, 'epoch': 1.04}
{'loss': 0.0002, 'grad_norm': 0.005777942948043346, 'learning_rate': 3.907580056749088e-05, 'epoch': 1.05}
{'loss': 0.0001, 'grad_norm': 0.005663580261170864, 'learning_rate': 3.887312525334415e-05, 'epoch': 1.06}
{'loss': 0.0001, 'grad_norm': 0.004087391309440136, 'learning_rate': 3.8670449939197405e-05, 'epoch': 1.07}
{'loss': 0.0001, 'grad_norm': 0.004718040116131306, 'learning_rate': 3.846777462505067e-05, 'epoch': 1.08}
{'loss': 0.0001, 'grad_norm': 0.00392565131187439, 'learning_rate': 3.8265099310903936e-05, 'epoch': 1.09}
{'loss': 0.0001, 'grad_norm': 0.0032761436887085438, 'learning_rate': 3.8062423996757194e-05, 'epoch': 1.1}
{'loss': 0.0001, 'grad_norm': 0.0030942028388381004, 'learning_rate': 3.785974868261046e-05, 'epoch': 1.11}
{'loss': 0.0001, 'grad_norm': 0.0035988660529255867, 'learning_rate': 3.7657073368463725e-05, 'epoch': 1.12}
{'loss': 0.0001, 'grad_norm': 0.0029355608858168125, 'learning_rate': 3.7454398054316983e-05, 'epoch': 1.13}
{'loss': 0.0001, 'grad_norm': 0.002896358259022236, 'learning_rate': 3.725172274017025e-05, 'epoch': 1.14}
{'loss': 0.0001, 'grad_norm': 0.0028437122236937284, 'learning_rate': 3.7049047426023514e-05, 'epoch': 1.15}
{'loss': 0.0645, 'grad_norm': 0.0027552121318876743, 'learning_rate': 3.684637211187677e-05, 'epoch': 1.16}
{'loss': 0.0572, 'grad_norm': 0.02824368327856064, 'learning_rate': 3.664369679773004e-05, 'epoch': 1.17}
{'loss': 0.001, 'grad_norm': 0.01887322962284088, 'learning_rate': 3.6441021483583296e-05, 'epoch': 1.18}
{'loss': 0.0004, 'grad_norm': 0.012518520466983318, 'learning_rate': 3.623834616943657e-05, 'epoch': 1.19}
{'loss': 0.0004, 'grad_norm': 0.011071452870965004, 'learning_rate': 3.603567085528983e-05, 'epoch': 1.2}
{'loss': 0.0003, 'grad_norm': 0.009490208700299263, 'learning_rate': 3.5832995541143086e-05, 'epoch': 1.21}
{'loss': 0.0003, 'grad_norm': 0.007565722800791264, 'learning_rate': 3.563032022699636e-05, 'epoch': 1.22}
{'loss': 0.0562, 'grad_norm': 0.007862173020839691, 'learning_rate': 3.5427644912849616e-05, 'epoch': 1.23}
{'loss': 0.0004, 'grad_norm': 0.00949922576546669, 'learning_rate': 3.522496959870288e-05, 'epoch': 1.24}
{'loss': 0.0525, 'grad_norm': 0.025156598538160324, 'learning_rate': 3.502229428455615e-05, 'epoch': 1.25}
{'loss': 0.001, 'grad_norm': 0.029864812269806862, 'learning_rate': 3.4819618970409405e-05, 'epoch': 1.26}
{'loss': 0.001, 'grad_norm': 0.01766064390540123, 'learning_rate': 3.461694365626267e-05, 'epoch': 1.27}
{'loss': 0.0459, 'grad_norm': 0.03446286916732788, 'learning_rate': 3.441426834211593e-05, 'epoch': 1.28}
{'loss': 0.0011, 'grad_norm': 0.02294689416885376, 'learning_rate': 3.4211593027969194e-05, 'epoch': 1.29}
{'loss': 0.0451, 'grad_norm': 0.011026696301996708, 'learning_rate': 3.400891771382246e-05, 'epoch': 1.3}
{'loss': 0.0014, 'grad_norm': 0.029947757720947266, 'learning_rate': 3.380624239967572e-05, 'epoch': 1.31}
{'loss': 0.0015, 'grad_norm': 0.039991654455661774, 'learning_rate': 3.3603567085528983e-05, 'epoch': 1.32}
{'loss': 0.0007, 'grad_norm': 0.012387235648930073, 'learning_rate': 3.340089177138225e-05, 'epoch': 1.33}
{'loss': 0.0004, 'grad_norm': 0.013510618358850479, 'learning_rate': 3.319821645723551e-05, 'epoch': 1.34}
{'loss': 0.0003, 'grad_norm': 0.009218805469572544, 'learning_rate': 3.299554114308877e-05, 'epoch': 1.35}
{'loss': 0.0003, 'grad_norm': 0.006130189169198275, 'learning_rate': 3.279286582894204e-05, 'epoch': 1.37}
{'loss': 0.0002, 'grad_norm': 0.008726980537176132, 'learning_rate': 3.2590190514795296e-05, 'epoch': 1.38}
{'loss': 0.0567, 'grad_norm': 0.013453085906803608, 'learning_rate': 3.238751520064856e-05, 'epoch': 1.39}
{'loss': 0.0006, 'grad_norm': 0.022187311202287674, 'learning_rate': 3.218483988650183e-05, 'epoch': 1.4}
{'loss': 0.0006, 'grad_norm': 0.013327263295650482, 'learning_rate': 3.1982164572355086e-05, 'epoch': 1.41}
{'loss': 0.0005, 'grad_norm': 0.010667928494513035, 'learning_rate': 3.177948925820835e-05, 'epoch': 1.42}
{'loss': 0.0004, 'grad_norm': 0.006664025131613016, 'learning_rate': 3.1576813944061616e-05, 'epoch': 1.43}
{'loss': 0.0003, 'grad_norm': 0.009030776098370552, 'learning_rate': 3.137413862991488e-05, 'epoch': 1.44}
{'loss': 0.0003, 'grad_norm': 0.010148894041776657, 'learning_rate': 3.117146331576814e-05, 'epoch': 1.45}
{'loss': 0.0002, 'grad_norm': 0.007179900072515011, 'learning_rate': 3.09687880016214e-05, 'epoch': 1.46}
{'loss': 0.0478, 'grad_norm': 0.009314306080341339, 'learning_rate': 3.076611268747467e-05, 'epoch': 1.47}
{'loss': 0.0003, 'grad_norm': 0.01082802377641201, 'learning_rate': 3.056343737332793e-05, 'epoch': 1.48}
{'loss': 0.0005, 'grad_norm': 0.01174898911267519, 'learning_rate': 3.036076205918119e-05, 'epoch': 1.49}
{'loss': 0.048, 'grad_norm': 0.039404865354299545, 'learning_rate': 3.015808674503446e-05, 'epoch': 1.5}
{'loss': 0.001, 'grad_norm': 0.01995680294930935, 'learning_rate': 2.9955411430887718e-05, 'epoch': 1.51}
{'loss': 0.0007, 'grad_norm': 0.016175102442502975, 'learning_rate': 2.975273611674098e-05, 'epoch': 1.52}
{'loss': 0.0004, 'grad_norm': 0.009052020497620106, 'learning_rate': 2.9550060802594242e-05, 'epoch': 1.53}
{'loss': 0.0003, 'grad_norm': 0.010535222478210926, 'learning_rate': 2.934738548844751e-05, 'epoch': 1.54}
{'loss': 0.0003, 'grad_norm': 0.004514409229159355, 'learning_rate': 2.9144710174300773e-05, 'epoch': 1.55}
{'loss': 0.0003, 'grad_norm': 0.007551089394837618, 'learning_rate': 2.894203486015403e-05, 'epoch': 1.56}
{'loss': 0.0002, 'grad_norm': 0.004122275393456221, 'learning_rate': 2.87393595460073e-05, 'epoch': 1.57}
{'loss': 0.0002, 'grad_norm': 0.004633034113794565, 'learning_rate': 2.8536684231860562e-05, 'epoch': 1.58}
{'loss': 0.0001, 'grad_norm': 0.0036190184764564037, 'learning_rate': 2.8334008917713824e-05, 'epoch': 1.59}
{'loss': 0.0001, 'grad_norm': 0.004037955310195684, 'learning_rate': 2.813133360356709e-05, 'epoch': 1.6}
{'loss': 0.0781, 'grad_norm': 0.003790635848417878, 'learning_rate': 2.792865828942035e-05, 'epoch': 1.61}
{'loss': 0.0003, 'grad_norm': 0.008041470311582088, 'learning_rate': 2.7725982975273613e-05, 'epoch': 1.62}
{'loss': 0.0004, 'grad_norm': 0.017370181158185005, 'learning_rate': 2.7523307661126875e-05, 'epoch': 1.63}
{'loss': 0.0005, 'grad_norm': 0.0144822271540761, 'learning_rate': 2.732063234698014e-05, 'epoch': 1.64}
{'loss': 0.0003, 'grad_norm': 0.007018620613962412, 'learning_rate': 2.7117957032833402e-05, 'epoch': 1.65}
{'loss': 0.0002, 'grad_norm': 0.00883771013468504, 'learning_rate': 2.6915281718686664e-05, 'epoch': 1.66}
{'loss': 0.0002, 'grad_norm': 0.008199871517717838, 'learning_rate': 2.671260640453993e-05, 'epoch': 1.67}
{'loss': 0.0502, 'grad_norm': 0.0071488842368125916, 'learning_rate': 2.650993109039319e-05, 'epoch': 1.68}
{'loss': 0.0006, 'grad_norm': 0.07544813305139542, 'learning_rate': 2.6307255776246453e-05, 'epoch': 1.69}
{'loss': 0.0003, 'grad_norm': 0.024173492565751076, 'learning_rate': 2.6104580462099715e-05, 'epoch': 1.7}
{'loss': 0.0004, 'grad_norm': 0.007863149978220463, 'learning_rate': 2.590190514795298e-05, 'epoch': 1.71}
{'loss': 0.0644, 'grad_norm': 0.01604750007390976, 'learning_rate': 2.5699229833806242e-05, 'epoch': 1.72}
{'loss': 0.0498, 'grad_norm': 0.03377952799201012, 'learning_rate': 2.5496554519659504e-05, 'epoch': 1.73}
{'loss': 0.0011, 'grad_norm': 0.01974060758948326, 'learning_rate': 2.5293879205512773e-05, 'epoch': 1.74}
{'loss': 0.0544, 'grad_norm': 0.056814104318618774, 'learning_rate': 2.5091203891366035e-05, 'epoch': 1.75}
{'loss': 0.0541, 'grad_norm': 0.04302617162466049, 'learning_rate': 2.4888528577219296e-05, 'epoch': 1.76}
{'loss': 0.0016, 'grad_norm': 0.04189681261777878, 'learning_rate': 2.468585326307256e-05, 'epoch': 1.77}
{'loss': 0.001, 'grad_norm': 0.01437455229461193, 'learning_rate': 2.4483177948925824e-05, 'epoch': 1.78}
{'loss': 0.0007, 'grad_norm': 0.013397746719419956, 'learning_rate': 2.4280502634779086e-05, 'epoch': 1.79}
{'loss': 0.0005, 'grad_norm': 0.013235628604888916, 'learning_rate': 2.4077827320632347e-05, 'epoch': 1.8}
{'loss': 0.0003, 'grad_norm': 0.010654788464307785, 'learning_rate': 2.387515200648561e-05, 'epoch': 1.81}
{'loss': 0.1033, 'grad_norm': 0.01460549607872963, 'learning_rate': 2.3672476692338875e-05, 'epoch': 1.82}
{'loss': 0.0012, 'grad_norm': 0.05820357799530029, 'learning_rate': 2.346980137819214e-05, 'epoch': 1.83}
{'loss': 0.0016, 'grad_norm': 0.02500942163169384, 'learning_rate': 2.32671260640454e-05, 'epoch': 1.84}
{'loss': 0.0488, 'grad_norm': 0.04131266102194786, 'learning_rate': 2.3064450749898664e-05, 'epoch': 1.85}
{'loss': 0.002, 'grad_norm': 0.03244448080658913, 'learning_rate': 2.2861775435751926e-05, 'epoch': 1.86}
{'loss': 0.0011, 'grad_norm': 0.019001934677362442, 'learning_rate': 2.265910012160519e-05, 'epoch': 1.87}
{'loss': 0.0008, 'grad_norm': 0.0323227122426033, 'learning_rate': 2.2456424807458453e-05, 'epoch': 1.88}
{'loss': 0.0004, 'grad_norm': 0.01249295100569725, 'learning_rate': 2.2253749493311715e-05, 'epoch': 1.89}
{'loss': 0.0004, 'grad_norm': 0.01053493283689022, 'learning_rate': 2.205107417916498e-05, 'epoch': 1.9}
{'loss': 0.0004, 'grad_norm': 0.009186923503875732, 'learning_rate': 2.1848398865018242e-05, 'epoch': 1.91}
{'loss': 0.0004, 'grad_norm': 0.009820694103837013, 'learning_rate': 2.1645723550871504e-05, 'epoch': 1.92}
{'loss': 0.0003, 'grad_norm': 0.00843748264014721, 'learning_rate': 2.144304823672477e-05, 'epoch': 1.93}
{'loss': 0.0002, 'grad_norm': 0.004471865948289633, 'learning_rate': 2.124037292257803e-05, 'epoch': 1.94}
{'loss': 0.0619, 'grad_norm': 0.009668359532952309, 'learning_rate': 2.1037697608431296e-05, 'epoch': 1.95}
{'loss': 0.0003, 'grad_norm': 0.009009214118123055, 'learning_rate': 2.0835022294284555e-05, 'epoch': 1.96}
{'loss': 0.0004, 'grad_norm': 0.010920043103396893, 'learning_rate': 2.063234698013782e-05, 'epoch': 1.97}
{'loss': 0.0004, 'grad_norm': 0.011074681766331196, 'learning_rate': 2.0429671665991082e-05, 'epoch': 1.98}
{'loss': 0.0003, 'grad_norm': 0.0070615787990391254, 'learning_rate': 2.0226996351844347e-05, 'epoch': 1.99}
 67%|███████████████████████████████▎               | 1978/2967 [1:05:26<27:50,  1.69s/it]C:\Users\nohyunwoo\AppData\Local\Programs\Python\Python313\Lib\site-packages\torch\utils\data\dataloader.py:665: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
  warnings.warn(warn_msg)
{'loss': 0.0003, 'grad_norm': 0.006858270149677992, 'learning_rate': 2.002432103769761e-05, 'epoch': 2.0}
{'loss': 0.0003, 'grad_norm': 0.0096206059679389, 'learning_rate': 1.982164572355087e-05, 'epoch': 2.01}
{'loss': 0.0003, 'grad_norm': 0.007046884391456842, 'learning_rate': 1.9618970409404137e-05, 'epoch': 2.02}
{'loss': 0.0003, 'grad_norm': 0.007023043930530548, 'learning_rate': 1.94162950952574e-05, 'epoch': 2.03}
{'loss': 0.053, 'grad_norm': 0.007144239265471697, 'learning_rate': 1.921361978111066e-05, 'epoch': 2.04}
{'loss': 0.0004, 'grad_norm': 0.007984503172338009, 'learning_rate': 1.9010944466963926e-05, 'epoch': 2.05}
{'loss': 0.06, 'grad_norm': 0.020779188722372055, 'learning_rate': 1.8808269152817188e-05, 'epoch': 2.06}
{'loss': 0.0006, 'grad_norm': 0.020642293617129326, 'learning_rate': 1.8605593838670453e-05, 'epoch': 2.07}
{'loss': 0.0497, 'grad_norm': 0.025462467223405838, 'learning_rate': 1.840291852452371e-05, 'epoch': 2.08}
{'loss': 0.001, 'grad_norm': 0.029021622613072395, 'learning_rate': 1.8200243210376977e-05, 'epoch': 2.09}
{'loss': 0.0009, 'grad_norm': 0.029290396720170975, 'learning_rate': 1.7997567896230242e-05, 'epoch': 2.1}
{'loss': 0.0006, 'grad_norm': 0.01692507602274418, 'learning_rate': 1.7794892582083504e-05, 'epoch': 2.11}
{'loss': 0.0005, 'grad_norm': 0.016425857320427895, 'learning_rate': 1.7592217267936766e-05, 'epoch': 2.12}
{'loss': 0.0502, 'grad_norm': 0.019803358241915703, 'learning_rate': 1.7389541953790028e-05, 'epoch': 2.13}
{'loss': 0.0009, 'grad_norm': 0.034439802169799805, 'learning_rate': 1.7186866639643293e-05, 'epoch': 2.14}
{'loss': 0.0307, 'grad_norm': 0.022354884073138237, 'learning_rate': 1.6984191325496555e-05, 'epoch': 2.15}
{'loss': 0.0012, 'grad_norm': 0.02283117175102234, 'learning_rate': 1.6781516011349817e-05, 'epoch': 2.16}
{'loss': 0.0009, 'grad_norm': 0.01392267644405365, 'learning_rate': 1.6578840697203082e-05, 'epoch': 2.17}
{'loss': 0.0004, 'grad_norm': 0.00611873809248209, 'learning_rate': 1.6376165383056344e-05, 'epoch': 2.18}
{'loss': 0.0006, 'grad_norm': 0.012207097373902798, 'learning_rate': 1.617349006890961e-05, 'epoch': 2.19}
{'loss': 0.0003, 'grad_norm': 0.010779043659567833, 'learning_rate': 1.5970814754762868e-05, 'epoch': 2.2}
{'loss': 0.0003, 'grad_norm': 0.010781358927488327, 'learning_rate': 1.5768139440616133e-05, 'epoch': 2.21}
{'loss': 0.0003, 'grad_norm': 0.009725349023938179, 'learning_rate': 1.55654641264694e-05, 'epoch': 2.22}
{'loss': 0.0002, 'grad_norm': 0.007952923886477947, 'learning_rate': 1.536278881232266e-05, 'epoch': 2.23}
{'loss': 0.0334, 'grad_norm': 0.00860480684787035, 'learning_rate': 1.5160113498175924e-05, 'epoch': 2.24}
{'loss': 0.0003, 'grad_norm': 0.008125362917780876, 'learning_rate': 1.4957438184029184e-05, 'epoch': 2.25}
{'loss': 0.0007, 'grad_norm': 0.06367997080087662, 'learning_rate': 1.475476286988245e-05, 'epoch': 2.26}
{'loss': 0.0006, 'grad_norm': 0.009335863403975964, 'learning_rate': 1.4552087555735713e-05, 'epoch': 2.28}
{'loss': 0.0003, 'grad_norm': 0.007285625208169222, 'learning_rate': 1.4349412241588975e-05, 'epoch': 2.29}
{'loss': 0.0548, 'grad_norm': 0.006062877364456654, 'learning_rate': 1.4146736927442239e-05, 'epoch': 2.3}
{'loss': 0.0006, 'grad_norm': 0.05737612396478653, 'learning_rate': 1.39440616132955e-05, 'epoch': 2.31}
{'loss': 0.0004, 'grad_norm': 0.011678754352033138, 'learning_rate': 1.3741386299148764e-05, 'epoch': 2.32}
{'loss': 0.0003, 'grad_norm': 0.009265607222914696, 'learning_rate': 1.3538710985002026e-05, 'epoch': 2.33}
{'loss': 0.0403, 'grad_norm': 0.0406498983502388, 'learning_rate': 1.333603567085529e-05, 'epoch': 2.34}
{'loss': 0.0008, 'grad_norm': 0.04066990688443184, 'learning_rate': 1.3133360356708555e-05, 'epoch': 2.35}
{'loss': 0.0009, 'grad_norm': 0.006005753297358751, 'learning_rate': 1.2930685042561815e-05, 'epoch': 2.36}
{'loss': 0.0006, 'grad_norm': 0.02013816125690937, 'learning_rate': 1.272800972841508e-05, 'epoch': 2.37}
{'loss': 0.0004, 'grad_norm': 0.016042528674006462, 'learning_rate': 1.252533441426834e-05, 'epoch': 2.38}
{'loss': 0.0004, 'grad_norm': 0.004432631656527519, 'learning_rate': 1.2322659100121606e-05, 'epoch': 2.39}
{'loss': 0.0629, 'grad_norm': 0.006603694055229425, 'learning_rate': 1.2119983785974868e-05, 'epoch': 2.4}
{'loss': 0.0003, 'grad_norm': 0.011926287785172462, 'learning_rate': 1.1917308471828132e-05, 'epoch': 2.41}
{'loss': 0.0004, 'grad_norm': 0.007967524230480194, 'learning_rate': 1.1714633157681395e-05, 'epoch': 2.42}
{'loss': 0.0002, 'grad_norm': 0.006955114658921957, 'learning_rate': 1.1511957843534659e-05, 'epoch': 2.43}
{'loss': 0.0004, 'grad_norm': 0.0075476826168596745, 'learning_rate': 1.130928252938792e-05, 'epoch': 2.44}
{'loss': 0.0003, 'grad_norm': 0.005495982710272074, 'learning_rate': 1.1106607215241184e-05, 'epoch': 2.45}
{'loss': 0.0003, 'grad_norm': 0.008004318922758102, 'learning_rate': 1.0903931901094446e-05, 'epoch': 2.46}
{'loss': 0.0525, 'grad_norm': 0.007964500226080418, 'learning_rate': 1.070125658694771e-05, 'epoch': 2.47}
{'loss': 0.0004, 'grad_norm': 0.008026069030165672, 'learning_rate': 1.0498581272800973e-05, 'epoch': 2.48}
{'loss': 0.0005, 'grad_norm': 0.01912871189415455, 'learning_rate': 1.0295905958654237e-05, 'epoch': 2.49}
{'loss': 0.0003, 'grad_norm': 0.007544142659753561, 'learning_rate': 1.0093230644507499e-05, 'epoch': 2.5}
{'loss': 0.0005, 'grad_norm': 0.006918668281286955, 'learning_rate': 9.890555330360763e-06, 'epoch': 2.51}
{'loss': 0.0003, 'grad_norm': 0.007576568517833948, 'learning_rate': 9.687880016214024e-06, 'epoch': 2.52}
{'loss': 0.0004, 'grad_norm': 0.007525772787630558, 'learning_rate': 9.48520470206729e-06, 'epoch': 2.53}
{'loss': 0.0003, 'grad_norm': 0.006446697283536196, 'learning_rate': 9.282529387920552e-06, 'epoch': 2.54}
{'loss': 0.0003, 'grad_norm': 0.009233945049345493, 'learning_rate': 9.079854073773815e-06, 'epoch': 2.55}
{'loss': 0.0003, 'grad_norm': 0.0064387195743620396, 'learning_rate': 8.877178759627077e-06, 'epoch': 2.56}
{'loss': 0.0002, 'grad_norm': 0.006962988991290331, 'learning_rate': 8.67450344548034e-06, 'epoch': 2.57}
{'loss': 0.0002, 'grad_norm': 0.004421955440193415, 'learning_rate': 8.471828131333603e-06, 'epoch': 2.58}
{'loss': 0.0002, 'grad_norm': 0.0051404135301709175, 'learning_rate': 8.269152817186868e-06, 'epoch': 2.59}
{'loss': 0.0003, 'grad_norm': 0.007867894135415554, 'learning_rate': 8.06647750304013e-06, 'epoch': 2.6}
{'loss': 0.0603, 'grad_norm': 0.006604671943932772, 'learning_rate': 7.863802188893393e-06, 'epoch': 2.61}
{'loss': 0.0498, 'grad_norm': 0.012551522813737392, 'learning_rate': 7.661126874746655e-06, 'epoch': 2.62}
{'loss': 0.0004, 'grad_norm': 0.01613113470375538, 'learning_rate': 7.458451560599919e-06, 'epoch': 2.63}
{'loss': 0.0004, 'grad_norm': 0.012187233194708824, 'learning_rate': 7.255776246453182e-06, 'epoch': 2.64}
{'loss': 0.0005, 'grad_norm': 0.016111044213175774, 'learning_rate': 7.053100932306446e-06, 'epoch': 2.65}
{'loss': 0.0004, 'grad_norm': 0.02067507803440094, 'learning_rate': 6.850425618159709e-06, 'epoch': 2.66}
{'loss': 0.0003, 'grad_norm': 0.00973419938236475, 'learning_rate': 6.647750304012972e-06, 'epoch': 2.67}
{'loss': 0.0003, 'grad_norm': 0.006356060039252043, 'learning_rate': 6.4450749898662345e-06, 'epoch': 2.68}
{'loss': 0.0003, 'grad_norm': 0.005425754934549332, 'learning_rate': 6.242399675719498e-06, 'epoch': 2.69}
{'loss': 0.0002, 'grad_norm': 0.005486427806317806, 'learning_rate': 6.039724361572761e-06, 'epoch': 2.7}
{'loss': 0.0003, 'grad_norm': 0.008596209809184074, 'learning_rate': 5.837049047426024e-06, 'epoch': 2.71}
{'loss': 0.0454, 'grad_norm': 2.22688889503479, 'learning_rate': 5.634373733279287e-06, 'epoch': 2.72}
{'loss': 0.0003, 'grad_norm': 0.005728650838136673, 'learning_rate': 5.43169841913255e-06, 'epoch': 2.73}
{'loss': 0.0004, 'grad_norm': 0.008585246279835701, 'learning_rate': 5.229023104985813e-06, 'epoch': 2.74}
{'loss': 0.0003, 'grad_norm': 0.01630215533077717, 'learning_rate': 5.026347790839076e-06, 'epoch': 2.75}
{'loss': 0.0507, 'grad_norm': 0.007476779632270336, 'learning_rate': 4.823672476692339e-06, 'epoch': 2.76}
{'loss': 0.0556, 'grad_norm': 0.020366227254271507, 'learning_rate': 4.620997162545603e-06, 'epoch': 2.77}
{'loss': 0.0003, 'grad_norm': 0.006892255507409573, 'learning_rate': 4.4183218483988654e-06, 'epoch': 2.78}
{'loss': 0.0005, 'grad_norm': 0.00975730735808611, 'learning_rate': 4.215646534252128e-06, 'epoch': 2.79}
{'loss': 0.0006, 'grad_norm': 0.016257699579000473, 'learning_rate': 4.012971220105392e-06, 'epoch': 2.8}
{'loss': 0.0449, 'grad_norm': 0.011577652767300606, 'learning_rate': 3.8102959059586546e-06, 'epoch': 2.81}
{'loss': 0.0006, 'grad_norm': 0.009711001999676228, 'learning_rate': 3.6076205918119173e-06, 'epoch': 2.82}
{'loss': 0.0005, 'grad_norm': 0.020035061985254288, 'learning_rate': 3.404945277665181e-06, 'epoch': 2.83}
{'loss': 0.0004, 'grad_norm': 0.010536937043070793, 'learning_rate': 3.2022699635184437e-06, 'epoch': 2.84}
{'loss': 0.0005, 'grad_norm': 0.013872882351279259, 'learning_rate': 2.999594649371707e-06, 'epoch': 2.85}
{'loss': 0.0005, 'grad_norm': 0.013697258196771145, 'learning_rate': 2.7969193352249696e-06, 'epoch': 2.86}
{'loss': 0.0005, 'grad_norm': 0.01265893504023552, 'learning_rate': 2.594244021078233e-06, 'epoch': 2.87}
{'loss': 0.0005, 'grad_norm': 0.01118963398039341, 'learning_rate': 2.391568706931496e-06, 'epoch': 2.88}
{'loss': 0.0404, 'grad_norm': 0.01155104674398899, 'learning_rate': 2.188893392784759e-06, 'epoch': 2.89}
{'loss': 0.0005, 'grad_norm': 0.014765758998692036, 'learning_rate': 1.986218078638022e-06, 'epoch': 2.9}
{'loss': 0.0418, 'grad_norm': 0.009664991870522499, 'learning_rate': 1.7835427644912851e-06, 'epoch': 2.91}
{'loss': 0.0004, 'grad_norm': 0.01617024838924408, 'learning_rate': 1.5808674503445483e-06, 'epoch': 2.92}
 'epoch': 2.96}
{'loss': 0.0006, 'grad_norm': 0.016488539054989815, 'learning_rate': 5.674908796108635e-07, 'epoch': 2.97}
{'loss': 0.0005, 'grad_norm': 0.006110728718340397, 'learning_rate': 3.648155654641265e-07, 'epoch': 2.98}
{'loss': 0.0005, 'grad_norm': 0.017570458352565765, 'learning_rate': 1.6214025131738955e-07, 'epoch': 2.99}
{'train_runtime': 5891.7163, 'train_samples_per_second': 8.053, 'train_steps_per_second': 0.504, 'train_loss': 0.04419876692401412, 'epoch': 3.0}
100%|███████████████████████████████████████████████| 2967/2967 [1:38:11<00:00,  1.99s/it]