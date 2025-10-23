# 🚀 배포 가이드 (5분 완성)

성동구 소상공인 AI 비밀상담사를 온라인으로 배포하는 방법입니다.

---

## 📦 1단계: 파일 준비 (1분)

### 필요한 파일들
```
📁 프로젝트 폴더/
├── 📄 app.py                    ← 메인 애플리케이션
├── 📄 requirements.txt          ← 패키지 목록
├── 📄 .gitignore               ← Git 제외 파일
├── 📄 README.md                ← 프로젝트 설명
└── 📁 data/                    ← 데이터 폴더
    ├── lgbm_closure_predictor.pkl
    ├── integrated_final_dataset.csv
    └── 📁 pdfs/
        ├── (참고1) 경영안정지원사업 비교표...pdf
        ├── [eBook]2024년 소상공인 역량강화사업...pdf
        ├── ★★2024 기업가형 소상공인 우수사례집...pdf
        ├── 2025년 중소기업육성자금 융자지원...pdf
        ├── 250527_소상공인_부담경감_크레딧...pdf
        └── 은행권 소상공인 지원방안...pdf
```

### ✅ 체크리스트
- [ ] `app.py` 파일
- [ ] `requirements.txt` 파일
- [ ] `.gitignore` 파일
- [ ] `README.md` 파일
- [ ] `data/` 폴더 안에 모델, 데이터, PDF 파일들

---

## 🔑 2단계: Gemini API 키 발급 (1분)

1. **Google AI Studio 접속**
   - 🔗 https://aistudio.google.com/app/apikey

2. **API 키 생성**
   - "Create API key" 버튼 클릭
   - 기존 프로젝트 선택 또는 새 프로젝트 생성

3. **API 키 복사**
   ```
   예시: AIzaSyD...여기에_키가_나옵니다...xYz
   ```
   - ⚠️ 이 키는 나중에 다시 볼 수 없으니 메모장에 복사!

---

## 📤 3단계: GitHub에 업로드 (2분)

### Option A: GitHub Desktop (쉬운 방법) 👍

1. **GitHub Desktop 다운로드**
   - 🔗 https://desktop.github.com

2. **New Repository 생성**
   - "File" → "New Repository"
   - Name: `seongdong-ai-consultant` (또는 원하는 이름)
   - Local Path: 프로젝트 폴더 선택
   - "Create Repository" 클릭

3. **Publish to GitHub**
   - "Publish repository" 클릭
   - ⚠️ "Keep this code private" **체크 해제** (public으로 만들기)
   - "Publish Repository" 클릭

### Option B: 명령어 (익숙한 분만)

```bash
cd 프로젝트_폴더

# Git 초기화
git init
git add .
git commit -m "Initial commit: AI 비밀상담사"

# GitHub에 업로드
git remote add origin https://github.com/사용자명/저장소명.git
git branch -M main
git push -u origin main
```

---

## ☁️ 4단계: Streamlit Cloud 배포 (1분)

1. **Streamlit 가입/로그인**
   - 🔗 https://share.streamlit.io
   - GitHub 계정으로 로그인

2. **New app 클릭**
   - 우측 상단 "New app" 버튼

3. **설정 입력**
   ```
   Repository: 여러분의_저장소명
   Branch: main
   Main file path: app.py
   ```

4. **⚠️ 중요: Secrets 설정**
   - "Advanced settings..." 클릭
   - "Secrets" 탭 클릭
   - 다음 내용 입력:
   ```toml
   GEMINI_API_KEY = "여기에_2단계에서_복사한_API_키_붙여넣기"
   ```

5. **Deploy! 클릭**
   - 1-2분 기다리면 배포 완료! 🎉

---

## ✅ 5단계: 확인 및 공유

### 배포 완료 확인
- 앱이 자동으로 실행됨
- URL 형식: `https://사용자명-저장소명.streamlit.app`

### 테스트
1. 가맹점 ID 입력 (예: 000F03E44A)
2. "진단 시작" 클릭
3. 결과가 잘 나오는지 확인

### 공유
- 이 URL을 공모전 제출 시 제공하면 끝! 🎊

---

## 🔧 자주 발생하는 문제

### 1. "Module not found" 에러
**원인**: requirements.txt 누락 또는 오타
**해결**: requirements.txt 파일이 있는지, 철자가 맞는지 확인

### 2. "File not found" 에러
**원인**: data 폴더가 GitHub에 업로드 안 됨
**해결**: 
- data 폴더와 그 안의 파일들이 모두 GitHub에 있는지 확인
- .gitignore에서 data/를 제외하지 않았는지 확인

### 3. "API quota exceeded" 에러
**원인**: Gemini API 무료 사용량 초과 (하루 50회)
**해결**: 
- 다음날까지 대기 (자정 UTC에 리셋)
- 또는 Google Cloud에서 결제 활성화

### 4. 앱이 느려요
**원인**: 44MB 데이터 로딩
**해결**: 정상입니다. 첫 로딩 후에는 캐시되어 빠름

---

## 📱 모바일 접속

- 생성된 URL은 모바일에서도 접속 가능
- 반응형 디자인으로 모바일 최적화됨

---

## 🎓 추가 팁

### 앱 업데이트 방법
1. GitHub에 새 코드 푸시
2. Streamlit Cloud가 자동으로 재배포
3. 1-2분 후 업데이트 완료

### 로그 확인
- Streamlit Cloud 대시보드에서 "Manage app" → "Logs"
- 에러 발생 시 여기서 확인 가능

### 앱 재시작
- "Manage app" → "Reboot app"
- 문제 발생 시 재시작으로 해결되는 경우 많음

---

## 💬 도움이 필요하면?

### Streamlit 공식 문서
- 🔗 https://docs.streamlit.io/streamlit-community-cloud

### Streamlit 커뮤니티
- 🔗 https://discuss.streamlit.io

---

## 🎉 축하합니다!

온라인 배포 완료! 이제 누구나 인터넷으로 접속 가능합니다.

**공모전 제출 시 포함 사항:**
✅ 배포된 앱 URL
✅ GitHub 저장소 URL
✅ README.md (프로젝트 설명)

**예시:**
```
앱 URL: https://myname-seongdong-ai.streamlit.app
GitHub: https://github.com/myname/seongdong-ai-consultant
```
