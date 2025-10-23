import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import shap

# ====================================================================
# 🚨 반드시 최상단에 위치해야 함!
# ====================================================================
st.set_page_config(
    page_title="AI 비밀상담사", 
    layout="wide", 
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# ====================================================================
# 커스텀 CSS 스타일
# ====================================================================
st.markdown("""
<style>
    /* 메인 헤더 스타일 */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* 메트릭 카드 */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
    }
    
    /* 위험도 표시 */
    .risk-high {
        color: #ef4444;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .risk-medium {
        color: #f59e0b;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .risk-low {
        color: #10b981;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    /* 섹션 헤더 */
    .section-header {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    /* 버튼 스타일 개선 */
    .stButton > button {
        width: 100%;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    /* 정보 박스 */
    .info-box {
        background: #f0f9ff;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
    }
    
    /* 경고 박스 */
    .warning-box {
        background: #fef3c7;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #f59e0b;
        margin: 1rem 0;
    }
    
    /* 성공 박스 */
    .success-box {
        background: #d1fae5;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #10b981;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ====================================================================
# A. 초기 설정 및 리소스 로드
# ====================================================================

# 환경 변수 로드
load_dotenv()

# Streamlit Cloud에서는 secrets 사용, 로컬에서는 .env 사용
if 'GEMINI_API_KEY' in st.secrets:
    API_KEY = st.secrets["GEMINI_API_KEY"]
else:
    API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("⚠️ GEMINI_API_KEY를 설정해주세요!")
    st.info("로컬: .env 파일에 설정 | Streamlit Cloud: secrets.toml에 설정")
    st.stop()

os.environ["GEMINI_API_KEY"] = API_KEY

# ===== 📂 경로 설정 (배포용 상대 경로) =====
# 현재 스크립트의 디렉토리 기준
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_PATH, 'data')
PDF_FOLDER = os.path.join(DATA_FOLDER, 'pdf')

MODEL_PATH = os.path.join(DATA_FOLDER, 'lgbm_closure_predictor.pkl')
DATA_PATH = os.path.join(DATA_FOLDER, 'integrated_final_dataset.csv')

# 컬럼명 한글 매핑 사전
COLUMN_KOREAN_NAMES = {
    'OPERATING_DAYS': '운영 기간',
    'MCT_OPE_MS_CN_RANK': '운영개월수 등급',
    'RC_M1_SAA_RANK': '매출금액 등급',
    'RC_M1_TO_UE_CT_RANK': '매출건수 등급',
    'RC_M1_UE_CUS_CN_RANK': '고객 수 등급',
    'RC_M1_AV_NP_AT_RANK': '객단가 등급',
    'APV_CE_RAT_RANK': '취소율 등급',
    'DLV_SAA_RAT': '배달매출 비율',
    'M1_SME_RY_SAA_RAT': '업종 평균 대비 매출금액',
    'M1_SME_RY_CNT_RAT': '업종 평균 대비 매출건수',
    'M12_SME_RY_SAA_PCE_RT': '업종 내 매출 순위',
    'M12_SME_BZN_SAA_PCE_RT': '상권 내 매출 순위',
    'M12_SME_RY_ME_MCT_RAT': '업종 내 폐업 가맹점 비중',
    'M12_SME_BZN_ME_MCT_RAT': '상권 내 폐업 가맹점 비중',
    'M12_MAL_1020_RAT': '남성 20대이하 고객 비중',
    'M12_MAL_30_RAT': '남성 30대 고객 비중',
    'M12_MAL_40_RAT': '남성 40대 고객 비중',
    'M12_MAL_50_RAT': '남성 50대 고객 비중',
    'M12_MAL_60_RAT': '남성 60대이상 고객 비중',
    'M12_FME_1020_RAT': '여성 20대이하 고객 비중',
    'M12_FME_30_RAT': '여성 30대 고객 비중',
    'M12_FME_40_RAT': '여성 40대 고객 비중',
    'M12_FME_50_RAT': '여성 50대 고객 비중',
    'M12_FME_60_RAT': '여성 60대이상 고객 비중',
    'MCT_UE_CLN_REU_RAT': '재방문 고객 비중',
    'MCT_UE_CLN_NEW_RAT': '신규 고객 비중',
    'RC_M1_SHC_RSD_UE_CLN_RAT': '거주 이용 고객 비율',
    'RC_M1_SHC_WP_UE_CLN_RAT': '직장 이용 고객 비율',
    'RC_M1_SHC_FLP_UE_CLN_RAT': '유동인구 이용 고객 비율',
    'HPSN_MCT_ZCD_NM': '업종',
    'MCT_SIGUNGU_NM': '가맹점 지역'
}

# PDF 파일 목록
PDF_FILES = [
    '(참고1)  경영안정지원사업 비교표(크레딧 vs 비즈플러스 vs 배달‧택배).pdf',
    '[eBook]2024년 소상공인 역량강화사업 우수사례집.pdf',
    '★★2024 기업가형 소상공인 우수사례집_저용량.pdf',
    '2025년 중소기업육성자금 융자지원 변경계획 공고문(251020).pdf',
    '250527_소상공인_부담경감_크레딧_지원사업_설명자료(지역홍보용,_2쪽).pdf',
    '은행권 소상공인 지원방안(금융위자료).pdf',
    '소상공인 부담경감 크레딧 지원사업.pdf'  # ← 새로 추가!
]
PDF_PATHS = [os.path.join(PDF_FOLDER, f) for f in PDF_FILES]

# LLM 초기화
@st.cache_resource
def get_llm():
    """LLM 초기화 (캐싱)"""
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=API_KEY,
            temperature=0.2
        )
    except Exception as e:
        st.error(f"LLM 초기화 실패: {e}")
        return None

llm = get_llm()

# --- 리소스 로드 함수 ---
@st.cache_resource(show_spinner=False)
def load_resources():
    """모델, 데이터, Vector DB를 한 번만 로드"""
    
    loading_placeholder = st.empty()
    
    try:
        # 1. 모델 및 데이터 로드
        loading_placeholder.info("🔄 리소스 로드 중...")
        
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
            return None, None, None, None, None
        if not os.path.exists(DATA_PATH):
            st.error(f"❌ 데이터 파일을 찾을 수 없습니다: {DATA_PATH}")
            return None, None, None, None, None
        
        lgbm_model = joblib.load(MODEL_PATH)
        df = pd.read_csv(DATA_PATH, encoding='utf-8')
        
        # 2. SHAP Explainer 생성
        df_latest = df.loc[df.groupby('ENCODED_MCT')['TA_YM'].idxmax()].copy()
        
        # 구간 데이터 → 숫자로 변환
        rank_cols = ['MCT_OPE_MS_CN', 'RC_M1_SAA', 'RC_M1_TO_UE_CT', 
                    'RC_M1_UE_CUS_CN', 'RC_M1_AV_NP_AT', 'APV_CE_RAT']
        
        for col in rank_cols:
            if col in df_latest.columns:
                df_latest[f'{col}_RANK'] = df_latest[col].astype(str).str.split('_').str[0].apply(
                    lambda x: int(x) if x.isdigit() else np.nan
                )
                df_latest = df_latest.drop(columns=[col])
        
        # 피처 컬럼 추출
        exclude_cols = ['ENCODED_MCT', 'TA_YM', 'MCT_ME_D', 'ARE_D', 'TA_YM_DT',
                       'HPSN_MCT_BZN_CD_NM', 'MCT_BSE_AR', 'MCT_NM', 'MCT_BRD_NUM']
        features = [col for col in df_latest.columns if col not in exclude_cols and col != 'IS_CLOSED']
        
        # SHAP용 샘플 데이터
        X_sample = df_latest[features].copy()
        
        # 범주형 변수를 category 타입으로 변환
        categorical_features = ['HPSN_MCT_ZCD_NM', 'MCT_SIGUNGU_NM']
        for col in categorical_features:
            if col in X_sample.columns:
                X_sample[col] = X_sample[col].astype('category')
        
        # 숫자형 결측치 처리
        numerical_cols = X_sample.select_dtypes(include=[np.number]).columns
        X_sample[numerical_cols] = X_sample[numerical_cols].fillna(0)
        
        X_sample = X_sample.head(100)
        explainer = shap.TreeExplainer(lgbm_model)
        
        # 3. RAG Vector DB 구축
        documents = []
        loaded_pdfs = []
        
        for pdf_path in PDF_PATHS:
            if os.path.exists(pdf_path):
                try:
                    loader = PyPDFLoader(pdf_path)
                    docs = loader.load()
                    documents.extend(docs)
                    loaded_pdfs.append(os.path.basename(pdf_path))
                except Exception as e:
                    pass
        
        if not documents:
            st.error("❌ PDF 파일을 찾을 수 없습니다. 경로와 파일명을 확인하세요.")
            return None, None, None, None, None
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=API_KEY
        )
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        loading_placeholder.success(f"✅ 초기화 완료 (데이터: {len(df):,}건, PDF: {len(loaded_pdfs)}개)")
        
        return lgbm_model, df, explainer, vectorstore, features
    
    except Exception as e:
        loading_placeholder.error(f"❌ 리소스 로드 실패: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None, None, None

# 리소스 로드
lgbm_model, df_final, explainer, vectorstore, feature_names = load_resources()


# ====================================================================
# B. 핵심 함수: 예측 및 RAG 전략 생성
# ====================================================================

def interpret_rank(rank_value):
    """등급 데이터를 사람이 이해하기 쉬운 형태로 해석 (6단계)"""
    try:
        rank_str = str(rank_value)
        
        # "1_10%이하" 또는 "6_상위6구간(하위1구간)" 형태 파싱
        if '_' in rank_str:
            rank_num = rank_str.split('_')[0]
            rank_num = int(rank_num)
            
            # 6단계 평가
            if rank_num == 1:
                return "⭐⭐ 매우 좋음 (상위 10%)", "success"
            elif rank_num == 2:
                return "⭐ 좋음 (상위 10-30%)", "success"
            elif rank_num == 3:
                return "🟢 다소 좋음 (상위 30-50%)", "info"
            elif rank_num == 4:
                return "🟡 다소 나쁨 (하위 25-50%)", "warning"
            elif rank_num == 5:
                return "🟠 나쁨 (하위 10-25%)", "warning"
            elif rank_num == 6:
                return "🔴 매우 나쁨 (하위 10%)", "danger"
        
        return rank_value, "info"
    except:
        return rank_value, "info"


def interpret_percentile(value):
    """백분위를 해석하여 순위로 표현"""
    try:
        pct = float(value)
        if pct <= 10:
            return f"⭐ 최상위 {pct:.1f}% (상위권)", "success"
        elif pct <= 25:
            return f"🟢 상위 {pct:.1f}%", "success"
        elif pct <= 50:
            return f"🟡 중상위 {pct:.1f}%", "warning"
        elif pct <= 75:
            return f"🟠 중하위 {pct:.1f}%", "warning"
        else:
            return f"🔴 하위 {pct:.1f}%", "danger"
    except:
        return value, "info"


def interpret_ratio(value, label):
    """비율 데이터 해석"""
    try:
        ratio = float(value)
        
        if '업종평균대비' in label:
            if ratio >= 100:
                return f"⭐ 우수 (업종평균의 {ratio:.0f}%)", "success"
            elif ratio >= 80:
                return f"🟢 양호 (업종평균의 {ratio:.0f}%)", "success"
            elif ratio >= 60:
                return f"🟡 보통 (업종평균의 {ratio:.0f}%)", "warning"
            else:
                return f"🔴 부진 (업종평균의 {ratio:.0f}%)", "danger"
        
        elif '재방문' in label:
            if ratio >= 50:
                return f"⭐ 우수 ({ratio:.1f}%)", "success"
            elif ratio >= 35:
                return f"🟢 양호 ({ratio:.1f}%)", "success"
            elif ratio >= 25:
                return f"🟡 보통 ({ratio:.1f}%)", "warning"
            else:
                return f"🔴 낮음 ({ratio:.1f}%)", "danger"
        
        elif '배달매출' in label:
            if ratio >= 40:
                return f"📱 높음 ({ratio:.1f}%)", "info"
            elif ratio >= 20:
                return f"📱 보통 ({ratio:.1f}%)", "info"
            else:
                return f"📱 낮음 ({ratio:.1f}%)", "info"
        
        elif '취소' in label:
            if ratio <= 3:
                return f"⭐ 우수 ({ratio:.1f}%)", "success"
            elif ratio <= 7:
                return f"🟢 양호 ({ratio:.1f}%)", "success"
            elif ratio <= 12:
                return f"🟡 주의 ({ratio:.1f}%)", "warning"
            else:
                return f"🔴 높음 ({ratio:.1f}%)", "danger"
        
        return f"{ratio:.1f}%", "info"
    except:
        return value, "info"


def extract_merchant_data(mct_id, df, features):
    """가맹점의 실제 데이터 값 추출 및 해석 (근거 자료용)"""
    try:
        mct_data = df[df['ENCODED_MCT'] == mct_id]
        if mct_data.empty:
            return None
        
        latest_data = mct_data.loc[mct_data['TA_YM'].idxmax()].copy()
        
        # 주요 지표 추출 및 해석
        data_dict = {}
        
        # 등급 지표들 (해석 필요)
        if 'RC_M1_SAA' in latest_data.index:
            interpreted, status = interpret_rank(latest_data['RC_M1_SAA'])
            data_dict['💰 매출금액'] = (interpreted, status)
        
        if 'RC_M1_TO_UE_CT' in latest_data.index:
            interpreted, status = interpret_rank(latest_data['RC_M1_TO_UE_CT'])
            data_dict['🧾 매출건수'] = (interpreted, status)
        
        if 'RC_M1_UE_CUS_CN' in latest_data.index:
            interpreted, status = interpret_rank(latest_data['RC_M1_UE_CUS_CN'])
            data_dict['👥 고객수'] = (interpreted, status)
        
        if 'RC_M1_AV_NP_AT' in latest_data.index:
            interpreted, status = interpret_rank(latest_data['RC_M1_AV_NP_AT'])
            data_dict['💵 객단가'] = (interpreted, status)
        
        if 'APV_CE_RAT' in latest_data.index:
            interpreted, status = interpret_rank(latest_data['APV_CE_RAT'])
            data_dict['❌ 취소율'] = (interpreted, status)
        
        # 비율 지표
        if 'MCT_UE_CLN_REU_RAT' in latest_data.index:
            interpreted, status = interpret_ratio(latest_data['MCT_UE_CLN_REU_RAT'], '재방문')
            data_dict['🔄 재방문율'] = (interpreted, status)
        
        if 'DLV_SAA_RAT' in latest_data.index:
            interpreted, status = interpret_ratio(latest_data['DLV_SAA_RAT'], '배달매출')
            data_dict['🛵 배달매출비율'] = (interpreted, status)
        
        # 업종/상권 비교
        if 'M1_SME_RY_SAA_RAT' in latest_data.index:
            interpreted, status = interpret_ratio(latest_data['M1_SME_RY_SAA_RAT'], '업종평균대비')
            data_dict['📊 업종 대비 매출'] = (interpreted, status)
        
        if 'M12_SME_RY_SAA_PCE_RT' in latest_data.index:
            interpreted, status = interpret_percentile(latest_data['M12_SME_RY_SAA_PCE_RT'])
            data_dict['🏆 업종 내 순위'] = (interpreted, status)
        
        if 'M12_SME_BZN_SAA_PCE_RT' in latest_data.index:
            interpreted, status = interpret_percentile(latest_data['M12_SME_BZN_SAA_PCE_RT'])
            data_dict['📍 상권 내 순위'] = (interpreted, status)
        
        # 운영 기간
        if 'OPERATING_DAYS' in latest_data.index:
            days = int(latest_data['OPERATING_DAYS'])
            months = days // 30
            years = months // 12
            remaining_months = months % 12
            
            if years > 0:
                period_str = f"⏱️ {years}년 {remaining_months}개월 운영"
            else:
                period_str = f"⏱️ {months}개월 운영"
            
            data_dict['🗓️ 운영기간'] = (period_str, "info")
        
        return data_dict
    
    except Exception as e:
        return None


def predict_risk_with_shap(mct_id, df, model, explainer, features):
    """실제 LightGBM 모델로 폐업 위험 예측 + SHAP 분석"""
    try:
        # 1. 해당 가맹점의 최신 데이터 추출
        mct_data = df[df['ENCODED_MCT'] == mct_id]
        if mct_data.empty:
            return None, None, None, None, None
        
        latest_data = mct_data.loc[mct_data['TA_YM'].idxmax()].copy()
        
        # 2. 구간 데이터 → 숫자로 변환
        rank_cols = ['MCT_OPE_MS_CN', 'RC_M1_SAA', 'RC_M1_TO_UE_CT', 
                    'RC_M1_UE_CUS_CN', 'RC_M1_AV_NP_AT', 'APV_CE_RAT']
        
        for col in rank_cols:
            if col in latest_data.index:
                val = str(latest_data[col]).split('_')[0]
                try:
                    latest_data[f'{col}_RANK'] = int(val)
                except:
                    latest_data[f'{col}_RANK'] = 0
                latest_data = latest_data.drop(col)
        
        # 3. 범주형 변수를 category 타입으로 변환
        categorical_features = ['HPSN_MCT_ZCD_NM', 'MCT_SIGUNGU_NM']
        
        # 4. 피처 준비
        X_dict = {}
        for feat in features:
            if feat in latest_data.index:
                val = latest_data[feat]
                X_dict[feat] = val if pd.notna(val) else 0
            else:
                X_dict[feat] = 0
        
        X = pd.DataFrame([X_dict])
        
        # 5. 범주형 컬럼을 category 타입으로 변환
        for col in categorical_features:
            if col in X.columns:
                X[col] = X[col].astype('category')
        
        # 6. 숫자형 결측치 처리
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        X[numerical_cols] = X[numerical_cols].fillna(0)
        
        # 7. 예측 (확률 → 점수로 변환)
        risk_prob = model.predict_proba(X)[0][1]
        risk_score = risk_prob * 100
        
        # 8. SHAP 값 계산
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1][0]
        else:
            shap_values = shap_values[0]
        
        # 9. 위험/안전 요인 추출 (한글명으로 변환)
        shap_df = pd.DataFrame({
            'feature': features,
            'shap_value': shap_values
        }).sort_values('shap_value', ascending=False)
        
        risk_factors_raw = shap_df[shap_df['shap_value'] > 0].head(3)['feature'].tolist()
        safe_factors_raw = shap_df[shap_df['shap_value'] < 0].tail(3)['feature'].tolist()
        
        # 한글명으로 변환
        risk_factors = [COLUMN_KOREAN_NAMES.get(f, f) for f in risk_factors_raw]
        safe_factors = [COLUMN_KOREAN_NAMES.get(f, f) for f in safe_factors_raw]
        
        # 10. 메타 정보 추출
        업종 = latest_data.get('HPSN_MCT_ZCD_NM', '알 수 없음')
        지역 = latest_data.get('MCT_SIGUNGU_NM', '성동구')
        
        return risk_score, risk_factors, safe_factors, 업종, 지역
    
    except Exception as e:
        st.error(f"예측 오류: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None, None, None


def generate_diagnosis_comment(risk_score, risk_factors, safe_factors, 업종, 지역, llm, vectorstore, merchant_data):
    """진단 결과에 대한 AI 코멘트 생성 (RAG 활용) - 데이터 근거 포함"""
    if not llm:
        return "⚠️ AI 코멘트를 생성할 수 없습니다.", []
    
    try:
        # 위험도에 따른 상태 판단
        if risk_score >= 70:
            status = "고위험"
            query = f"{업종} 소상공인 폐업 방지 지원 정책 융자"
        elif risk_score >= 40:
            status = "주의 필요"
            query = f"{업종} 소상공인 매출 증대 경영안정 지원"
        else:
            status = "안정"
            query = f"{업종} 소상공인 성장 지원 정책 사례"
        
        # RAG로 관련 정책 검색
        relevant_docs = vectorstore.similarity_search(query, k=2)
        policy_context = "\n".join([doc.page_content[:300] for doc in relevant_docs])
        
        # 출처 문서 추출
        source_docs = []
        for doc in relevant_docs:
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                source_path = doc.metadata['source']
                source_filename = os.path.basename(source_path)
                if source_filename not in source_docs:
                    source_docs.append(source_filename)
        
        # 실제 데이터 정보 추가 (해석된 형태)
        data_info = ""
        if merchant_data:
            data_info = "\n[가맹점 경영 지표 (성동구 비교)]\n"
            for key, value_tuple in merchant_data.items():
                interpreted_value, _ = value_tuple
                data_info += f"- {key}: {interpreted_value}\n"
        
        prompt = f"""
당신은 소상공인 경영 전문 컨설턴트입니다. 다음 진단 결과를 분석하여 사업주가 이해하기 쉽게 설명해주세요.

🚨 필수 준수 사항:
1. 아래 [관련 정부 지원 정책]에 명시된 내용만 사용하세요
2. 정책명, 지원금액, 신청기간은 문서에 정확히 나온 내용만 언급
3. 문서에 없는 정책은 절대 언급하지 마세요
4. 불확실한 내용은 "관련 기관 확인 필요"라고 명시
5. 추측이나 상상으로 답변하지 마세요

**⚠️ 중요: 모든 제안과 진단에는 구체적인 수치 근거를 포함하되, 일반 사장님이 이해하기 쉬운 표현을 사용하세요.**

[가맹점 정보]
- 업종: {업종}
- 지역: {지역}
- 폐업 위험도: {risk_score:.1f}점 ({status})

[주요 위험 요인]
{chr(10).join(['- ' + f for f in risk_factors]) if risk_factors else '- 없음'}

[주요 강점]
{chr(10).join(['- ' + f for f in safe_factors]) if safe_factors else '- 없음'}

{data_info}

[관련 정부 지원 정책]
{policy_context}

다음 형식으로 **일반 사장님이 이해하기 쉽게** 답변하세요:

**🏪 현재 상태**
(경영 상태를 위 지표들을 인용하여 쉽게 설명. 예: "매출금액이 하위권이지만, 재방문율은 우수한 편입니다")

**⚠️ 주의할 점**
(위험 요인을 위 지표로 구체적으로 설명. 예: "업종 대비 매출이 부진하고, 상권 내에서도 하위권에 속합니다")

**💪 강점**
(강점을 위 지표로 설명. 예: "재방문율이 우수하여 단골 고객 확보는 잘 되고 있습니다")

**🎁 추천 지원 정책**
(현재 상태에 맞는 정부 지원 프로그램 1-2개 추천)

**💡 한 줄 조언**
(즉시 실행 가능한 구체적인 조언)
"""
        
        response = llm.invoke(prompt)
        return response.content, source_docs
    
    except Exception as e:
        return f"❌ 코멘트 생성 실패: {e}", []


def generate_chatbot_response(user_question, diagnosis_info, chat_history, vectorstore, llm):
    """챗봇 응답 생성 (대화 히스토리 + RAG 활용)"""
    if not llm or not vectorstore:
        return "⚠️ 챗봇을 사용할 수 없습니다."
    
    try:
        # 진단 정보
        risk_score = diagnosis_info['risk_score']
        risk_factors = diagnosis_info['risk_factors']
        safe_factors = diagnosis_info['safe_factors']
        업종 = diagnosis_info['업종']
        지역 = diagnosis_info['지역']
        
        # RAG 검색
        relevant_docs = vectorstore.similarity_search(user_question, k=2)
        rag_context = "\n".join([doc.page_content[:400] for doc in relevant_docs])
        
        # 대화 히스토리 (최근 4개만)
        recent_history = chat_history[-4:] if len(chat_history) > 4 else chat_history
        history_text = "\n".join([
            f"{'사용자' if msg['role'] == 'user' else 'AI'}: {msg['content']}"
            for msg in recent_history
        ])
        
        prompt = f"""
당신은 성동구 소상공인 전문 상담사입니다. 사업주와의 대화를 이어가며 실질적인 도움을 주세요.

🚨 필수 준수 사항:
1. 아래 [관련 정책 및 사례]에 있는 내용만 참고하세요
2. 지원 프로그램명, 금액, 조건은 문서 내용 그대로만 작성
3. 문서에 없는 프로그램이나 사례는 언급 금지
4. 성공 사례도 문서에 나온 것만 사용
5. 확실하지 않으면 "추가 확인 필요"라고 명시

[가맹점 진단 정보]
- 업종: {업종}
- 지역: {지역}
- 폐업 위험 점수: {risk_score:.1f}점
- 주요 위험 요인: {', '.join(risk_factors) if risk_factors else '없음'}
- 주요 강점: {', '.join(safe_factors) if safe_factors else '없음'}

[이전 대화]
{history_text if history_text else '(첫 질문입니다)'}

[관련 정책/사례 정보]
{rag_context}

[사용자 질문]
{user_question}

다음 원칙에 따라 답변하세요:
1. **친근하고 공감하는 톤**: "~하시면 좋을 것 같아요", "걱정되시죠?" 등
2. **구체적인 답변**: 추상적인 조언보다 실행 가능한 구체적 방법 제시
3. **진단 정보 활용**: 위 진단 결과를 참고하여 맞춤형 답변
4. **정책 정보 연결**: 관련 정부 지원이 있다면 자연스럽게 언급
5. **3-5문장 내외**: 너무 길지 않게, 핵심만 전달
6. **대화 맥락 유지**: 이전 대화 내용을 고려하여 자연스럽게 연결

답변:
"""
        
        response = llm.invoke(prompt)
        return response.content
    
    except Exception as e:
        return f"❌ 응답 생성 실패: {e}"


def generate_rag_strategy(risk_score, risk_factors, safe_factors, 업종, 지역, vectorstore, llm, merchant_data):
    """RAG 기반 맞춤형 전략 생성 - 데이터 근거 포함"""
    if not llm:
        return "⚠️ LLM이 초기화되지 않았습니다.", []
    
    if not vectorstore:
        return "⚠️ Vector DB가 초기화되지 않았습니다.", []
    
    try:
        # 1. RAG 검색 쿼리 생성
        if risk_score >= 70:
            query = f"{업종} 소상공인 고위험 폐업 방지 융자 지원 정책 사례"
        elif risk_score >= 40:
            query = f"{업종} 소상공인 매출 증대 마케팅 우수 사례"
        else:
            query = f"{업종} 소상공인 경영 안정화 성공 사례"
        
        # 2. Vector DB에서 관련 문서 검색
        relevant_docs = vectorstore.similarity_search(query, k=3)
        
        if not relevant_docs:
            context = "관련 정책 정보를 찾을 수 없습니다."
        else:
            context = "\n\n".join([doc.page_content[:500] for doc in relevant_docs])
        
        # 출처 문서 추출
        source_docs = []
        for doc in relevant_docs:
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                source_path = doc.metadata['source']
                source_filename = os.path.basename(source_path)
                if source_filename not in source_docs:
                    source_docs.append(source_filename)
        
        # 실제 데이터 정보 추가 (해석된 형태)
        data_info = ""
        if merchant_data:
            data_info = "\n[가맹점 경영 지표 (성동구 비교)]\n"
            for key, value_tuple in merchant_data.items():
                interpreted_value, _ = value_tuple
                data_info += f"- {key}: {interpreted_value}\n"
        
        # 3. Gemini에 전달할 프롬프트
        prompt = f"""
당신은 성동구 소상공인 전문 경영 컨설턴트입니다.

**⚠️ 중요: 모든 전략과 제안에는 위 경영 지표를 근거로 포함하되, 일반 사장님이 이해하기 쉬운 표현을 사용하세요.**

[가맹점 정보]
- 업종: {업종}
- 지역: {지역}
- 폐업 위험 점수: {risk_score:.1f}점 (100점 만점)

[AI 진단 결과]
- 주요 위험 요인: {', '.join(risk_factors) if risk_factors else '없음'}
- 주요 강점: {', '.join(safe_factors) if safe_factors else '없음'}

{data_info}

[관련 정책 및 사례 (RAG 검색 결과)]
{context}

위 정보를 바탕으로 다음 형식으로 **일반 사장님이 이해하기 쉽게** 답변하세요:

**📊 종합 진단**
- 위 경영 지표들을 종합하여 현재 상태를 쉽게 설명
- 예: "매출은 하위권이지만 재방문율이 우수하여, 기존 고객은 잘 관리되고 있습니다"

**💡 맞춤형 전략**

**1. 즉시 실행 방안 💪**
- **제안**: (구체적인 실행 방법)
- **이유**: (위 지표 중 어떤 부분을 개선하기 위한 것인지 설명)
- **예상 효과**: (개선 목표를 쉽게 설명)

**2. 정부 지원 활용 방법 🎁**
- **지원 프로그램**: (구체적인 프로그램명)
- **신청 자격**: (위 지표를 보았을 때 해당되는지)
- **지원 내용**: (금액, 기간 등)

**3. 마케팅 개선 제안 📱**
- **전략**: (구체적인 마케팅 방법)
- **근거**: (위 지표 중 어떤 부분에서 이 전략이 필요한지)
- **실행 방법**: (단계별 실행 방법)

**📌 성공 사례**
- 유사한 상황에서 성공한 가게 사례
- 구체적인 성과 수치 포함

**⚠️ 위 경영 지표들을 반드시 언급하며, 사장님이 "아, 우리 가게 상황에 딱 맞는 조언이네!"라고 느낄 수 있도록 작성하세요.**
"""
        
        # 4. Gemini 호출
        response = llm.invoke(prompt)
        
        if not response or not response.content:
            return "⚠️ AI 응답을 받지 못했습니다.", source_docs
        
        return response.content, source_docs
    
    except Exception as e:
        import traceback
        return f"❌ 전략 생성 실패: {e}\n\n```\n{traceback.format_exc()}\n```", []


# ====================================================================
# C. Streamlit UI
# ====================================================================

def main():
    # 헤더
    st.markdown("""
    <div class="main-header" style="padding: 1rem 2rem;">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <h1 style="margin: 0; font-size: 1.8rem; font-weight: 700; line-height: 1.2;">
                    🤖 성동 SAM <span style="font-size: 0.7rem; opacity: 0.7; font-weight: 400;">| Seongdong AI-based Management</span>
                </h1>
                <p style="font-size: 0.9rem; opacity: 0.85; margin: 0.3rem 0 0 0;">
                    ✨ 성공을 위한 동반자, 성동구 소상공인 AI 비밀상담사 '성동SAM'과 함께해요!
                </p>
            </div>
        </div>
        <div style="display: flex; gap: 1rem; align-items: center; margin-top: 0.8rem; padding-top: 0.8rem; border-top: 1px solid rgba(255,255,255,0.2);">
            <span style="font-size: 0.8rem; opacity: 0.8;">📈 폐업 위험 진단 (LightGBM+SHAP)</span>
            <span style="opacity: 0.5;">|</span>
            <span style="font-size: 0.8rem; opacity: 0.8;">💡 맞춤형 전략 (RAG+Gemini 2.5)</span>
            <span style="opacity: 0.5;">|</span>
            <span style="font-size: 0.75rem; opacity: 0.6;">📊 신한카드 데이터 86,590건</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 사이드바
    with st.sidebar:
        st.header("💡 사용 방법")
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; 
                    border-radius: 10px; 
                    color: white;
                    margin-bottom: 1rem;">
            <h3 style="color: white; margin-top: 0;">🚀 Quick Guide</h3>
            <div style="font-size: 1.1rem; line-height: 2rem;">
                <b>1️⃣</b> 가맹점ID 입력<br>
                <b>2️⃣</b> '진단 시작' 클릭<br>
                <b>3️⃣</b> AI 코멘트 확인<br>
                <b>4️⃣</b> 맞춤형 전략 생성<br>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 리소스 상태
        st.subheader("📊 시스템 상태")
        if lgbm_model is not None and df_final is not None:
            st.success("✅ 모든 리소스 로드 완료")
            
            # metric 글씨 크기 조절
            st.markdown("""
            <style>
            [data-testid="stMetricValue"] {
                font-size: 1.3rem;
            }
            </style>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("데이터", f"{len(df_final):,}건")
            with col2:
                st.metric("PDF", "6개")
        else:
            st.error("❌ 리소스 로드 실패")
        
        st.markdown("---")
        
        # 시스템 정보 (하단으로 이동)
        with st.expander("🔧 시스템 정보"):
            st.caption("**데이터 폴더**")
            st.code("./data/", language=None)
            
            st.caption("**모델 파일**")
            st.code(os.path.basename(MODEL_PATH), language=None)
            
            st.caption("**데이터 파일**")
            st.code(os.path.basename(DATA_PATH), language=None)
            
            st.caption("**PDF 파일**")
            st.code("./data/pdf/ (6개)", language=None)
    
    if lgbm_model is None or df_final is None:
        st.error("❌ 리소스 로드에 실패했습니다. 사이드바의 시스템 정보를 확인하세요.")
        return
    
    # 가맹점 ID 입력 영역
    st.markdown('<div class="section-header"><h3>🔍 가맹점 진단</h3></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        mct_id = st.text_input(
               "🔍 가맹점 ID 입력(Sample ID : AD57E72BC9, 869C372EFC, E8829764C8)", value="", key="mct_input", placeholder="예: AD57E72BC9")
    st.caption("💡 우리 가게의 가맹점구분번호를 넣고 Enter 입력 후 '진단 시작' 버튼을 클릭하세요")
        
    with col2:
         diagnose_btn = st.button("🏥 진단 시작", type="primary", use_container_width=True)
    
    # 진단 실행
    if diagnose_btn and mct_id:
        # 세션 상태 초기화
        st.session_state.diagnosis_done = False
        st.session_state.show_strategy = False
        st.session_state.chat_history = []
        # 이전 전략 삭제
        if 'generated_strategy' in st.session_state:
            del st.session_state.generated_strategy
        if 'strategy_sources' in st.session_state:
            del st.session_state.strategy_sources
        
        with st.spinner("⏳ AI가 경영 상태를 분석하고 있습니다..."):
            risk_score, risk_factors, safe_factors, 업종, 지역 = predict_risk_with_shap(
                mct_id, df_final, lgbm_model, explainer, feature_names
            )
            
            # 실제 데이터 추출
            merchant_data = extract_merchant_data(mct_id, df_final, feature_names)
        
        if risk_score is None:
            st.error(f"❌ '{mct_id}' 가맹점을 찾을 수 없습니다.")
            sample_ids = df_final['ENCODED_MCT'].head(5).tolist()
            st.info(f"💡 샘플 가맹점 ID: {', '.join(sample_ids)}")
            return
        
        # 진단 결과를 세션에 저장
        st.session_state.diagnosis_result = {
            'mct_id': mct_id,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'safe_factors': safe_factors,
            '업종': 업종,
            '지역': 지역,
            'merchant_data': merchant_data
        }
        st.session_state.diagnosis_done = True
    
    # 진단 결과 표시
    if st.session_state.get('diagnosis_done', False) and 'diagnosis_result' in st.session_state:
        result = st.session_state.diagnosis_result
        risk_score = result['risk_score']
        risk_factors = result['risk_factors']
        safe_factors = result['safe_factors']
        업종 = result['업종']
        지역 = result['지역']
        mct_id = result['mct_id']
        merchant_data = result.get('merchant_data', None)
        
        # 구분선
        st.markdown("---")
        
        # 결과 표시 - 탭으로 구성
        st.markdown('<div class="section-header"><h3>📋 진단 결과</h3></div>', unsafe_allow_html=True)
        
        # 위험도 표시
        col_a, col_b, col_c = st.columns([1, 1, 2])
        
        with col_a:
            if risk_score >= 70:
                status = "🔴 고위험"
                risk_class = "risk-high"
            elif risk_score >= 40:
                status = "🟡 주의"
                risk_class = "risk-medium"
            else:
                status = "🟢 안정"
                risk_class = "risk-low"
            
            st.metric("폐업 위험 점수", f"{risk_score:.1f}점")
            st.markdown(f'<p class="{risk_class}">{status}</p>', unsafe_allow_html=True)
        
        with col_b:
            st.metric("업종", 업종)
            st.metric("지역", 지역)
        
        with col_c:
            st.markdown("**🚨 주요 위험 요인**")
            if risk_factors:
                for factor in risk_factors:
                    st.markdown(f"• {factor}")
            else:
                st.markdown("• 특정 위험 요인 없음")
            
            st.markdown("**✅ 주요 강점**")
            if safe_factors:
                for factor in safe_factors:
                    st.markdown(f"• {factor}")
            else:
                st.markdown("• 특정 강점 없음")
        
        # 진행 바
        st.markdown("**위험도 시각화**")
        
        # 색상 결정
        if risk_score >= 70:
            bar_color = "#ef4444"  # 빨강
        elif risk_score >= 40:
            bar_color = "#f59e0b"  # 주황
        else:
            bar_color = "#10b981"  # 초록
        
        # 진행 바와 레이블
        st.markdown(f"""
<div style="margin: 1rem 0;">
    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
        <span style="color: #10b981; font-weight: 500;">← 안전</span>
        <span style="font-weight: 600; color: {bar_color}; font-size: 1.1rem;">
            현재: {risk_score:.1f}점
        </span>
        <span style="color: #ef4444; font-weight: 500;">위험 →</span>
    </div>
    <div style="position: relative; margin-bottom: 3rem;">
        <div style="width: 100%; height: 30px; background: linear-gradient(to right, #10b981, #fbbf24, #ef4444); border-radius: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"></div>
        <div style="position: absolute; left: {risk_score}%; top: 15px; transform: translate(-50%, -50%); width: 24px; height: 24px; background: white; border: 4px solid {bar_color}; border-radius: 50%; box-shadow: 0 3px 8px rgba(0,0,0,0.3); z-index: 2;"></div>
        <div style="position: absolute; left: {risk_score}%; top: -25px; transform: translateX(-50%); background: {bar_color}; color: white; padding: 4px 12px; border-radius: 12px; font-weight: bold; font-size: 0.9rem; box-shadow: 0 2px 4px rgba(0,0,0,0.2); white-space: nowrap; z-index: 3;">{risk_score:.1f}점</div>
        <div style="position: absolute; left: {risk_score}%; top: 35px; transform: translateX(-50%); width: 0; height: 0; border-left: 8px solid transparent; border-right: 8px solid transparent; border-top: 10px solid {bar_color}; z-index: 1;"></div>
    </div>
    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #666;">
        <span>0 (낮음)</span>
        <span>50 (보통)</span>
        <span>100 (높음)</span>
    </div>
</div>
""", unsafe_allow_html=True)


        
        # 📊 실제 데이터 근거 표시
        if merchant_data:
            st.markdown("---")
            st.markdown('<div class="section-header"><h3>📊 우리 가게 경영 지표 (성동구 비교)</h3></div>', unsafe_allow_html=True)
            st.caption("💡 성동구 소상공인들과 비교한 우리 가게의 위치를 확인하세요!")
            st.caption("📌 출처: 신한카드 요식업종 데이터 86,590건 분석자료")
            
            # 데이터를 3개 컬럼으로 나눠서 표시
            data_items = list(merchant_data.items())
            col_count = 3
            
            # 3개씩 묶어서 행으로 표시
            for i in range(0, len(data_items), col_count):
                cols = st.columns(col_count)
                
                for idx, (key, value_tuple) in enumerate(data_items[i:i+col_count]):
                    interpreted_value, status = value_tuple
                    
                    with cols[idx]:
                        # 상태에 따른 색상 스타일
                        if status == "success":
                            bg_color = "#d1fae5"
                            border_color = "#10b981"
                        elif status == "warning":
                            bg_color = "#fef3c7"
                            border_color = "#f59e0b"
                        elif status == "danger":
                            bg_color = "#fee2e2"
                            border_color = "#ef4444"
                        else:
                            bg_color = "#f0f9ff"
                            border_color = "#3b82f6"
                        
                        st.markdown(f"""
                        <div style="background: {bg_color}; 
                                    padding: 1rem; 
                                    border-radius: 8px; 
                                    border-left: 4px solid {border_color};
                                    margin-bottom: 0.5rem;">
                            <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.3rem;">{key}</div>
                            <div style="font-size: 1.1rem; font-weight: bold; color: #1a1a1a;">{interpreted_value}</div>
                        </div>
                        """, unsafe_allow_html=True)
        
        # AI 분석 코멘트
        st.markdown("---")
        st.markdown('<div class="section-header"><h3>💬 AI 진단 코멘트 (근거 기반)</h3></div>', unsafe_allow_html=True)
        
        with st.spinner("🤖 AI가 진단 결과를 분석하고 정책을 검색하고 있습니다..."):
            analysis_comment, diagnosis_sources = generate_diagnosis_comment(
                risk_score, risk_factors, safe_factors, 업종, 지역, llm, vectorstore, merchant_data
            )
        
        st.markdown(f'<div class="info-box">{analysis_comment}</div>', unsafe_allow_html=True)
        
        # 참고 자료 출처 표시
        if diagnosis_sources:
            st.markdown("---")
            st.markdown('<div class="section-header"><h3>📚 참고 자료 (정책 문서)</h3></div>', unsafe_allow_html=True)
            
            for idx, source in enumerate(diagnosis_sources, 1):
                # PDF 파일명 정리
                display_name = source.replace('.pdf', '').replace('_', ' ')
                
                # GitHub raw URL 생성
                github_url = f"https://raw.githubusercontent.com/ICeT-smk/seongdong-ai-consultant/main/data/pdf/{source}"
                
                # 파일 아이콘과 다운로드 버튼
                st.markdown(f"""
                <div style="background: #f8fafc; 
                            padding: 0.8rem; 
                            border-radius: 5px; 
                            border-left: 3px solid #667eea;
                            margin-bottom: 0.5rem;">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div style="display: flex; align-items: center;">
                            <span style="font-size: 1.5rem; margin-right: 0.5rem;">📄</span>
                            <span style="font-weight: 500;">{display_name}</span>
                        </div>
                        <a href="{github_url}" download="{source}" 
                           style="text-decoration: none; 
                                  background: #667eea; 
                                  color: white; 
                                  padding: 0.4rem 0.8rem; 
                                  border-radius: 5px; 
                                  font-size: 0.85rem;
                                  white-space: nowrap;">
                            📥 다운로드
                        </a>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.caption("💡 위 정부 지원 정책 문서를 기반으로 맞춤형 조언을 제공했습니다.")

        
 # 챗봇 기능
        st.markdown("---")
        st.markdown('<div class="section-header"><h3>💬성동SAM과 자유상담</h3></div>', unsafe_allow_html=True)
        st.caption("진단 결과에 대해 자유롭게 질문하세요!")
        
        # 세션 상태에 채팅 히스토리 저장
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if 'current_diagnosis' not in st.session_state:
            st.session_state.current_diagnosis = None
        
        # 현재 진단 정보 저장
        st.session_state.current_diagnosis = {
            'mct_id': mct_id,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'safe_factors': safe_factors,
            '업종': 업종,
            '지역': 지역
        }
        
        # 채팅 히스토리 제한 (최근 10개 메시지만 유지)
        if len(st.session_state.chat_history) > 10:
            st.session_state.chat_history = st.session_state.chat_history[-10:]
        
        # 채팅 히스토리 표시
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # 사용자 입력
        if user_question := st.chat_input("예: 재방문율을 높이려면 어떻게 해야 하나요?"):
            # 사용자 메시지 추가
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)
            
            # AI 응답 생성 (스트리밍)
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                # 스트리밍 응답
                try:
                    # 프롬프트 생성 (generate_chatbot_response 함수 내용을 여기로)
                    diagnosis_info = st.session_state.current_diagnosis
                    
                    # 관련 문서 검색 (k=1로 줄여서 빠르게)
                    relevant_docs = vectorstore.similarity_search(user_question, k=1)
                    context = "\n\n".join([doc.page_content for doc in relevant_docs])
                    
                    # 채팅 히스토리 (최근 4개만)
                    recent_history = st.session_state.chat_history[-5:] if len(st.session_state.chat_history) > 5 else st.session_state.chat_history
                    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history[:-1]])
                    
                    # 간결한 프롬프트
                    prompt = f"""당신은 성동구 소상공인 AI 컨설턴트 'SAM'입니다.

[진단 정보]
- 폐업 위험도: {diagnosis_info['risk_score']:.1f}점
- 업종: {diagnosis_info['업종']}
- 지역: {diagnosis_info['지역']}

[관련 정책 정보]
{context[:500]}

[대화 히스토리]
{history_text}

[질문]
{user_question}

⚠️ 중요:
1. 3-4문장으로 간결하게 답변하세요
2. 구체적이고 실행 가능한 조언을 제공하세요

답변:"""
                    
                    # 스트리밍으로 응답 생성
                    for chunk in llm.stream(prompt):
                        full_response += chunk.content
                        message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                
                except Exception as e:
                    full_response = "죄송합니다. 답변 생성 중 오류가 발생했습니다. 다시 시도해주세요."
                    message_placeholder.markdown(full_response)
            
            # AI 응답 저장
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})

if __name__ == '__main__':
    main()