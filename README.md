# 🤖 성동구 소상공인 AI 비밀상담사

폐업 위험 진단 (LightGBM + SHAP) 및 맞춤형 경영 전략 제공 (RAG + Gemini 2.0 Flash)

📊 **데이터 출처**: 신한카드 성동구 소상공인 관련 데이터

---

## 🎯 주요 기능

### 1. 폐업 위험 진단 (AI 예측 모델)
- LightGBM 기반 폐업 위험도 점수 (0-100점)
- SHAP으로 위험 요인 및 강점 분석
- 성동구 소상공인 대비 경영 지표 비교

### 2. 맞춤형 경영 전략 (RAG + LLM)
- 6개 정부 정책 문서 기반 RAG 검색
- Gemini 2.0 Flash로 맞춤형 조언 생성
- 참고 자료 출처 명시

### 3. AI 상담 챗봇
- 진단 결과 기반 실시간 상담
- 대화 히스토리 관리

---

## 📊 데이터 및 모델

### 입력 데이터
- **데이터셋**: integrated_final_dataset.csv (86,590건)
- **주요 변수**: 매출, 고객수, 재방문율, 업종/상권 비교 등 30+ 변수

### AI 모델
- **알고리즘**: LightGBM (폐업 예측)
- **설명 가능성**: SHAP (피처 중요도 분석)

### 정책 문서 (RAG)
1. 경영안정지원사업 비교표
2. 2024년 소상공인 역량강화사업 우수사례집
3. 2024 기업가형 소상공인 우수사례집
4. 2025년 중소기업육성자금 융자지원 변경계획
5. 소상공인 부담경감 크레딧 지원사업 설명자료
6. 은행권 소상공인 지원방안

---

## 🚀 배포 방법 (Streamlit Community Cloud)

### 1. GitHub 저장소 준비

```bash
# 프로젝트 폴더 구조
📦 프로젝트명/
├── ai_agent_improved.py
├── requirements.txt
├── .gitignore
├── README.md
└── data/
    ├── lgbm_closure_predictor.pkl
    ├── integrated_final_dataset.csv
    └── pdfs/
        ├── (참고1) 경영안정지원사업...pdf
        ├── [eBook]2024년...pdf
        ├── ★★2024 기업가형...pdf
        ├── 2025년 중소기업육성자금...pdf
        ├── 250527_소상공인_부담경감...pdf
        └── 은행권 소상공인...pdf
```

### 2. GitHub에 업로드

```bash
# Git 초기화
git init
git add .
git commit -m "Initial commit: AI 비밀상담사"

# GitHub 저장소 생성 후
git remote add origin https://github.com/사용자명/저장소명.git
git push -u origin main
```

### 3. Streamlit Cloud 배포

1. [share.streamlit.io](https://share.streamlit.io) 접속
2. GitHub 계정으로 로그인
3. "New app" 클릭
4. 저장소, 브랜치, 파일 선택
5. **Advanced settings** 클릭 → **Secrets** 입력:
   ```toml
   GEMINI_API_KEY = "여기에_API_키_입력"
   ```
6. "Deploy!" 클릭

### 4. API 키 발급 (필수)

1. [Google AI Studio](https://aistudio.google.com/app/apikey) 접속
2. "Create API key" 클릭
3. 발급된 키를 Streamlit Secrets에 입력

---

## 💻 로컬 실행 방법

### 1. 환경 설정

```bash
# 가상환경 생성 (선택)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. API 키 설정

프로젝트 루트에 `.env` 파일 생성:
```
GEMINI_API_KEY=여기에_API_키_입력
```

### 3. 실행

```bash
streamlit run ai_agent_improved.py
```

브라우저에서 자동으로 열림 (기본: http://localhost:8501)

---

## 🎨 주요 기능 설명

### 📊 경영 지표 해석
원시 데이터를 사장님이 이해하기 쉬운 표현으로 변환:
- ❌ "1_10%이하" → ✅ "⭐⭐ 매우 좋음 (상위 10%)"
- ❌ "7.5%ile" → ✅ "⭐ 최상위 7.5% (상위권)"

### 📈 위험도 시각화
- 색상 그라디언트 바 (초록 → 노랑 → 빨강)
- 정확한 위치 마커 (점수 배지 + 원형 마커 + 화살표)
- 0-50-100 눈금 표시

### 📚 근거 기반 제안
- 모든 AI 조언에 **참고 자료 출처** 명시
- 정책 문서명을 카드 형태로 표시
- RAG 검색 결과 추적 가능

---

## 🛠️ 기술 스택

### AI/ML
- **LightGBM**: 폐업 예측 모델
- **SHAP**: 설명 가능한 AI
- **Gemini 2.0 Flash**: 맞춤형 조언 생성

### RAG (Retrieval-Augmented Generation)
- **LangChain**: RAG 프레임워크
- **FAISS**: 벡터 DB
- **Google Embeddings**: 문서 임베딩

### 프론트엔드
- **Streamlit**: 웹 인터페이스
- **Custom CSS**: 디자인 개선

---

## 📝 사용 예시

### 1. 가맹점 진단
```
입력: 가맹점 ID (예: 000F03E44A)
출력: 
  - 폐업 위험 점수: 1.9점 (🟢 안전)
  - 위험 요인: 운영기간 등급, 고객 수 등급
  - 강점: 매출금액 등급, 재방문율
```

### 2. 경영 지표 확인
```
💰 매출금액: ⭐⭐ 매우 좋음 (상위 10%)
🔄 재방문율: ⭐ 우수 (34.9%)
🏆 업종 내 순위: ⭐ 최상위 7.5% (상위권)
```

### 3. 맞춤형 전략
```
- 즉시 실행 방안
- 정부 지원 활용 (참고 문서 명시)
- 마케팅 개선 제안
- 유사 업종 성공 사례
```

---

## 🔒 보안 및 개인정보

- ✅ 모든 가맹점 데이터는 **마스킹 처리됨**
- ✅ API 키는 `.env` 파일로 관리 (GitHub에 업로드 안 됨)
- ✅ Streamlit Cloud는 Secrets로 안전하게 관리

---

## 📞 문의

공모전 관련 문의 또는 기술 문의가 있으시면 연락 주세요.

---

## 📜 라이센스

이 프로젝트는 공모전 제출용입니다.

**데이터 출처**: 신한카드 성동구 소상공인 관련 데이터
