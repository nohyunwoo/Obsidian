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

