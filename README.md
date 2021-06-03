# 2-Stage 

**[WRAP UP REPORT](https://hackmd.io/@cdll-lo-ol-lo-ol/rJ-DbnrLu)**

> WRAP UP 리포트에는 2주 간의 과정이 상세하게 적혀있습니다..! :smile:

## 문제 개요

**관계 추출(Relation Extraction)은** 문장의 <u>단어(Entity)에 대한 속성과 관계를 예측</u>하는 문제입니다. 

관계 추출은 지식 그래프 구축을 위한 핵심 구성 요소로, 구조화된 검색, 감정 분석, 질문 답변하기, 요약과 같은 자연어처리 응용 프로그램에서 중요합니다. 비구조적인 자연어 문장에서 구조적인 triple을 추출해 정보를 요약하고, 중요한 성분을 핵심적으로 파악할 수 있습니다.

## 프로젝트 구조

### 폴더 구조

- `database` : 데이터베이스 관련 파일들이 들어있습니다.
- `model_vocab` : Kobert Tokenizer에서 사용하는 Vocab이 들어있습니다.
- `submits`

### 모듈 설명 

- `config.py` : hyperparameter를 설정하는 모듈
- `database.py` : database 쿼리 관련 모듈
- `head_view.js` : BertViz 시각화 관련 모듈
- `hp_space.py` : 하이퍼파라미터 Search Space 관련 모듈
- `inference.py` : 모델 추론 관련 모듈
- `losses.py` : 로스 관련 모듈
- `make_data.py` : 데이터 전처리 관련 모듈
- `networks.py` : 모델이 구현된 모듈
- `optimizers.py` : 옵티마이저 관련 모듈
- `prepare.py` : 데이터셋 관련 모듈
- `run.py` : 하이퍼 파라미터 튜닝또는 학습 관련 모듈
- `slack.py` : 슬랙 알람 모듈
- `test.html` : 버트 비즈 시각화 Example
- `tokenization_kobert.py` : Kobert 토크나이저 모듈
- `train.py` : 학습 관련 모듈
- `utils.py` : 유틸 관련 모듈
- `visualize.py` : BertViz 시각화 관련 모듈

### 학습 & 추론 파이프라인

![auto-full-pipeline](https://i.imgur.com/qfVsut8.png)

1. Sampling
    - 전략을 샘플링 합니다.
2. Train or Tune
    - `model.train()`: (After HyperParameter Optimization) 모델의 학습을 진행합니다.
    - `model.tune()`: HyperParameter Search를 진행합니다.
3. Evaluate
    - `model.eval()`: Validation Loss와 성능 지표(Acc)를 계산합니다. Logging을 담당합니다.
4. Push Alarm and If Best Score

    슬랙으로 모델의 학습이 끝나다는 것을 알립니다.

    - 4-1. Inference
        - 지금까지 학습된 모델들과 비교해서 우수하다고 판단이 되면 Test Dataset에 대해 Inference를 수행하고 결과물을 생성합니다.
      - 4-2. Submission
        - 생성된 결과물을 제출하고 1단계로 되돌아갑니다.
