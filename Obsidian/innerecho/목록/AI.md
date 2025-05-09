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
->

----
## **DistilBERT** 모델
**에크만의 6가지 기본 감정**: 기쁨, 슬픔, 분노, 공포, 놀람, 혐오 + 중립
-> 7가지
https://huggingface.co/docs/transformers/main/ko/tasks/sequence_classification

-> 한국어는 KoELECTRA 정확성을 따라올 수 없음
-> KoELECTRA 할 예정

------
## **KoELECTRA 모델
![[Pasted image 20250508231019.png]]