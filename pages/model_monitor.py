from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_recall_curve


# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(
    page_title="자동 분석 리포트",
    page_icon="📘",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "insurance_policyholder_churn_synthetic.csv"
MODEL_PATH = BASE_DIR / "model" / "churn_model_new.pkl"
THRESHOLD_PATH = BASE_DIR / "model" / "threshold_new.pkl"

TARGET_COL = "churn_flag"
DROP_COLS = ["customer_id", "as_of_date", "churn_type", "churn_probability_true"]


# =========================================================
# 스타일
# =========================================================
st.markdown("""
<style>
.main {
    background-color: #f6f8fb;
}
.block-container {
    padding-top: 1.4rem;
    padding-bottom: 3rem;
    max-width: 1380px;
}
.report-title {
    font-size: 2.5rem;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 0.25rem;
    letter-spacing: -0.02em;
}
.report-subtitle {
    color: #475569;
    font-size: 1.04rem;
    margin-bottom: 1.2rem;
}
.section-title {
    font-size: 1.75rem;
    font-weight: 800;
    color: #0f172a;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
.card {
    background: white;
    border-radius: 22px;
    padding: 1.35rem 1.45rem;
    box-shadow: 0 6px 22px rgba(15, 23, 42, 0.06);
    border: 1px solid #e2e8f0;
}
.kpi-card {
    border-radius: 24px;
    padding: 1.55rem 1.55rem 1.2rem 1.55rem;
    min-height: 210px;
    box-shadow: 0 8px 28px rgba(15, 23, 42, 0.06);
}
.kpi-neutral {
    background: #ffffff;
    border: 1px solid #e2e8f0;
}
.kpi-high {
    background: #fff1f2;
    border: 1px solid #fecdd3;
}
.kpi-medium {
    background: #fff7ed;
    border: 1px solid #fed7aa;
}
.kpi-low {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
}
.kpi-label {
    font-size: 1.1rem;
    color: #334155;
    margin-bottom: 0.8rem;
    font-weight: 700;
}
.kpi-value {
    font-size: 2.85rem;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.1;
}
.kpi-sub {
    font-size: 1rem;
    color: #475569;
    margin-top: 0.75rem;
}
.badge {
    display: inline-block;
    padding: 0.42rem 0.85rem;
    border-radius: 999px;
    font-size: 0.88rem;
    font-weight: 700;
}
.badge-done {
    background: #dcfce7;
    color: #15803d;
}
.feature-card {
    background: white;
    border-radius: 18px;
    padding: 1.25rem 1.35rem;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
    margin-bottom: 1rem;
}
.feature-title {
    font-size: 1.28rem;
    font-weight: 800;
    color: #0f172a;
}
.feature-impact {
    font-size: 2rem;
    font-weight: 800;
    color: #0f172a;
    text-align: right;
}
.feature-desc {
    color: #334155;
    font-size: 1.04rem;
    margin-top: 1rem;
    line-height: 1.75;
}
.progress-wrap {
    width: 100%;
    height: 12px;
    background: #e5e7eb;
    border-radius: 999px;
    overflow: hidden;
    margin-top: 1rem;
    margin-bottom: 1rem;
}
.progress-bar-red {
    height: 100%;
    background: #ef4444;
    border-radius: 999px;
}
.progress-bar-orange {
    height: 100%;
    background: #f97316;
    border-radius: 999px;
}
.progress-bar-green {
    height: 100%;
    background: #10b981;
    border-radius: 999px;
}
.pattern-high {
    background: #fff1f2;
    border: 1px solid #fca5a5;
    border-radius: 18px;
    padding: 1.2rem;
}
.pattern-medium {
    background: #fff7ed;
    border: 1px solid #fdba74;
    border-radius: 18px;
    padding: 1.2rem;
}
.pattern-low {
    background: #f0fdf4;
    border: 1px solid #86efac;
    border-radius: 18px;
    padding: 1.2rem;
}
.insight-card {
    border-radius: 18px;
    padding: 1.2rem;
    min-height: 170px;
}
.insight-blue {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
}
.insight-purple {
    background: #faf5ff;
    border: 1px solid #e9d5ff;
}
.insight-yellow {
    background: #fffbeb;
    border: 1px solid #fde68a;
}
.action-wrap {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    border-radius: 24px;
    padding: 1.7rem;
    color: white;
}
.action-box {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 18px;
    padding: 1.2rem;
    min-height: 220px;
}
.small-metric {
    background: rgba(255,255,255,0.78);
    border-radius: 14px;
    padding: 0.9rem 1rem;
    margin-top: 0.8rem;
}
.small-metric-label {
    color: #475569;
    font-size: 0.95rem;
}
.small-metric-value {
    font-size: 1.08rem;
    font-weight: 800;
    margin-top: 0.35rem;
}
.hr-space {
    height: 16px;
}
.debug-box {
    background:#0f172a;
    color:#e2e8f0;
    border-radius:16px;
    padding:1rem 1.1rem;
    font-size:0.92rem;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 전처리 / 파생변수 / threshold 함수
# =========================================================
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    cat_pipe = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]
    )
    num_pipe = Pipeline(
        steps=[('imputer', SimpleImputer(strategy='median'))]
    )

    return ColumnTransformer(
        transformers=[
            ('cat', cat_pipe, cat_cols),
            ('num', num_pipe, num_cols),
        ],
        remainder='drop',
    )


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    result = X.copy()

    # 안전하게 없는 컬럼이 있으면 기본값 생성
    defaults = {
        "premium_change_pct": 0.0,
        "num_price_increases_last_3y": 0,
        "late_payment_count_12m": 0,
        "missed_payment_flag": 0,
        "complaint_flag": 0,
        "quote_requested_flag": 0,
        "coverage_downgrade_flag": 0,
        "customer_tenure_months": 0,
        "multi_policy_flag": 0,
        "payment_frequency": "Unknown",
        "policy_type": "Unknown",
    }
    for col, default in defaults.items():
        if col not in result.columns:
            result[col] = default

    result['premium_increase_shock'] = np.clip(result['premium_change_pct'], 0, None) * (1 + result['num_price_increases_last_3y'])
    result['premium_jump_flag'] = (result['premium_change_pct'] >= 0.12).astype(int)
    result['payment_risk_score'] = result['late_payment_count_12m'] + result['missed_payment_flag'] * 2
    result['service_risk_score'] = result['complaint_flag'] + result['quote_requested_flag'] + result['coverage_downgrade_flag']
    result['engagement_risk_score'] = result['payment_risk_score'] + result['service_risk_score']
    result['tenure_inverse'] = 1 / (result['customer_tenure_months'] + 1)
    result['single_policy_short_tenure'] = ((result['multi_policy_flag'] == 0) & (result['customer_tenure_months'] <= 24)).astype(int)
    result['premium_x_complaint'] = result['premium_jump_flag'] * result['complaint_flag']
    result['premium_x_quote'] = result['premium_jump_flag'] * result['quote_requested_flag']
    result['late_x_quote'] = (result['late_payment_count_12m'] >= 2).astype(int) * result['quote_requested_flag']
    result['downgrade_x_quote'] = result['coverage_downgrade_flag'] * result['quote_requested_flag']
    result['monthly_payment_flag'] = (result['payment_frequency'] == 'Monthly').astype(int)
    result['monthly_x_premium_jump'] = result['monthly_payment_flag'] * result['premium_jump_flag']
    result['auto_or_health'] = result['policy_type'].isin(['Auto', 'Health']).astype(int)
    return result


def fbeta_score_from_pr(precision: np.ndarray, recall: np.ndarray, beta: float) -> np.ndarray:
    beta2 = beta * beta
    return (1 + beta2) * (precision * recall) / (beta2 * precision + recall + 1e-12)


def select_threshold(thresholds, precision, recall, opt_metric='f1', min_precision=0.0, min_recall=0.0):
    if opt_metric == 'f2':
        scores = fbeta_score_from_pr(precision, recall, beta=2.0)
    else:
        scores = fbeta_score_from_pr(precision, recall, beta=1.0)

    mask = (precision >= min_precision) & (recall >= min_recall)
    if mask.any():
        idx = int(np.nanargmax(scores * mask))
        return float(thresholds[idx])

    idx = int(np.nanargmax(scores))
    return float(thresholds[idx])


def select_threshold_by_accuracy(y_true, y_prob, thresholds):
    candidate_thresholds = np.unique(np.clip(np.concatenate(([0.0], thresholds, [1.0])), 0.0, 1.0))
    best_threshold = 0.5
    best_accuracy = -1.0

    for threshold in candidate_thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        accuracy = accuracy_score(y_true, y_pred)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)

    return best_threshold, best_accuracy


def save_threshold_plot(y_true, y_prob, selected_threshold, out_path):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    thresholds = thresholds.astype(float)
    precision_t = precision[1:]
    recall_t = recall[1:]
    f1 = fbeta_score_from_pr(precision_t, recall_t, beta=1.0)
    f2 = fbeta_score_from_pr(precision_t, recall_t, beta=2.0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), dpi=140)
    axes[0].plot(recall, precision, color='#4C78A8')
    axes[0].set_title('PR Curve')
    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precision')

    axes[1].plot(thresholds, recall_t, label='Recall', color='#1F77B4')
    axes[1].plot(thresholds, precision_t, label='Precision', color='#FF7F0E')
    axes[1].plot(thresholds, f1, label='F1', color='#2CA02C', linestyle='--')
    axes[1].plot(thresholds, f2, label='F2', color='#9467BD', linestyle='--')
    axes[1].axvline(selected_threshold, color='red', linestyle=':', label='Selected')
    axes[1].set_title('Threshold vs Metrics')
    axes[1].set_xlabel('Threshold')
    axes[1].set_ylabel('Score')
    axes[1].legend(fontsize=8)

    axes[2].scatter(thresholds, f2, s=12, color='#7F7F7F')
    axes[2].axvline(selected_threshold, color='red', linestyle=':')
    axes[2].set_title('F2 by Threshold')
    axes[2].set_xlabel('Threshold')
    axes[2].set_ylabel('F2')

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


# =========================================================
# 유틸
# =========================================================
def fmt_int(x):
    return f"{int(x):,}"


def fmt_pct(x):
    return f"{x:.1f}%"


def safe_mean(series):
    if series is None or len(series) == 0:
        return 0.0
    return float(np.nanmean(series))


def safe_rate(df: pd.DataFrame, col: str):
    if col not in df.columns or len(df) == 0:
        return 0.0
    return safe_mean(df[col]) * 100


def safe_numeric_mean(df: pd.DataFrame, col: str):
    if col not in df.columns or len(df) == 0:
        return 0.0
    return safe_mean(df[col])


def risk_bucket(prob, low_cut=0.35, high_cut=0.65):
    if prob >= high_cut:
        return "고위험"
    elif prob >= low_cut:
        return "중위험"
    return "저위험"


def bar_class(color: str):
    if color == "red":
        return "progress-bar-red"
    if color == "orange":
        return "progress-bar-orange"
    return "progress-bar-green"


# =========================================================
# 데이터/모델 로드
# =========================================================
@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"데이터 파일이 없습니다: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def align_features_to_model(model, X: pd.DataFrame):
    """
    모델이 기억하는 학습 컬럼(feature_names_in_) 기준으로
    예측 입력을 정렬한다.
    """
    aligned = X.copy()

    if hasattr(model, "feature_names_in_"):
        trained_cols = list(model.feature_names_in_)

        for col in trained_cols:
            if col not in aligned.columns:
                aligned[col] = 0

        aligned = aligned[trained_cols]
        return aligned, trained_cols

    # feature_names_in_가 없으면 그대로 반환
    return aligned, list(aligned.columns)


def prepare_scored_dataframe():
    df = load_data().copy()

    if TARGET_COL not in df.columns:
        raise ValueError(f"'{TARGET_COL}' 컬럼이 데이터에 없습니다.")

    y = df[TARGET_COL].astype(int)

    drop_cols = [c for c in DROP_COLS if c in df.columns] + [TARGET_COL]
    X = df.drop(columns=drop_cols, errors="ignore")

    # 학습 때와 동일하게 파생변수 생성
    X = add_engineered_features(X)

    model = load_model()

    if not hasattr(model, "predict_proba"):
        raise ValueError("현재 모델은 predict_proba를 지원하지 않습니다.")

    X_aligned, trained_cols = align_features_to_model(model, X)
    y_prob = model.predict_proba(X_aligned)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y, y_prob)

    if len(thresholds) == 0:
        selected_threshold = 0.5
    else:
        selected_threshold = select_threshold(
            thresholds=thresholds,
            precision=precision[1:],
            recall=recall[1:],
            opt_metric='f2',
            min_precision=0.0,
            min_recall=0.0
        )

    low_cut = max(0.0, selected_threshold * 0.55)
    high_cut = selected_threshold

    scored = df.copy()
    scored["churn_probability"] = y_prob
    scored["risk_group"] = scored["churn_probability"].apply(
        lambda x: risk_bucket(x, low_cut=low_cut, high_cut=high_cut)
    )

    debug_info = {
        "aligned_feature_count": len(X_aligned.columns),
        "first_20_features": list(X_aligned.columns[:20]),
    }

    return scored, selected_threshold, debug_info


# =========================================================
# 텍스트 인사이트 생성
# =========================================================
def get_feature_insights(scored_df: pd.DataFrame):
    high_df = scored_df[scored_df["risk_group"] == "고위험"]

    items = []

    # 1. 보험료 인상률
    if "premium_change_pct" in scored_df.columns:
        overall = safe_numeric_mean(scored_df, "premium_change_pct") * 100
        high_val = safe_numeric_mean(high_df, "premium_change_pct") * 100
        impact = min(95, max(55, high_val * 4))
        items.append({
            "title": "보험료 인상률 (premium_change_pct)",
            "impact": impact,
            "level": "높은 영향" if impact >= 80 else "중간 영향",
            "color": "red" if impact >= 80 else "orange",
            "desc": f"전체 평균 보험료 인상률은 {overall:.1f}%이며, 고위험 고객군에서는 평균 {high_val:.1f}%로 더 높게 나타납니다. 보험료 인상은 고객의 가격 민감도를 자극해 이탈 가능성을 높이는 핵심 신호입니다."
        })

    # 2. 민원 발생 여부
    if "complaint_flag" in scored_df.columns:
        overall = safe_rate(scored_df, "complaint_flag")
        high_val = safe_rate(high_df, "complaint_flag")
        impact = min(92, max(50, high_val * 1.25))
        items.append({
            "title": "민원 발생 여부 (complaint_flag)",
            "impact": impact,
            "level": "높은 영향" if impact >= 80 else "중간 영향",
            "color": "red" if impact >= 80 else "orange",
            "desc": f"전체 고객 중 민원 경험 비율은 {overall:.1f}%이고, 고위험 고객군에서는 {high_val:.1f}%입니다. 민원은 서비스 불만과 관계 약화가 동시에 드러나는 대표적인 위험 신호입니다."
        })

    # 3. 연체 횟수
    if "late_payment_count_12m" in scored_df.columns:
        overall = safe_numeric_mean(scored_df, "late_payment_count_12m")
        high_val = safe_numeric_mean(high_df, "late_payment_count_12m")
        impact = min(88, max(48, high_val * 28))
        items.append({
            "title": "납부 연체 횟수 (late_payment_count_12m)",
            "impact": impact,
            "level": "높은 영향" if impact >= 80 else "중간 영향",
            "color": "red" if impact >= 80 else "orange",
            "desc": f"최근 12개월 평균 연체 횟수는 전체 {overall:.2f}회, 고위험군 {high_val:.2f}회입니다. 연체 증가는 재정 부담과 계약 유지 의지 약화의 초기 신호로 해석할 수 있습니다."
        })

    # 4. 타사 견적 요청
    if "quote_requested_flag" in scored_df.columns:
        overall = safe_rate(scored_df, "quote_requested_flag")
        high_val = safe_rate(high_df, "quote_requested_flag")
        impact = min(85, max(45, high_val * 1.15))
        items.append({
            "title": "타사 견적 요청 (quote_requested_flag)",
            "impact": impact,
            "level": "높은 영향" if impact >= 80 else "중간 영향",
            "color": "red" if impact >= 80 else "orange",
            "desc": f"전체 고객 대비 타사 견적 요청 비율은 {overall:.1f}%이며, 고위험 고객군에서는 {high_val:.1f}%입니다. 가격 비교 행동은 실제 이탈 직전의 탐색 단계일 가능성이 큽니다."
        })

    # 5. 보장 축소 요청
    if "coverage_downgrade_flag" in scored_df.columns:
        overall = safe_rate(scored_df, "coverage_downgrade_flag")
        high_val = safe_rate(high_df, "coverage_downgrade_flag")
        impact = min(80, max(42, high_val * 1.2))
        items.append({
            "title": "보장범위 축소 요청 (coverage_downgrade_flag)",
            "impact": impact,
            "level": "중간 영향",
            "color": "orange",
            "desc": f"보장 축소 요청 비율은 전체 {overall:.1f}%, 고위험군 {high_val:.1f}%입니다. 비용 절감 시도가 커질수록 장기 유지 의향이 낮아졌을 가능성이 있습니다."
        })

    # 6. 자동이체 여부
    if "autopay_enabled" in scored_df.columns:
        overall = safe_rate(scored_df, "autopay_enabled")
        high_val = safe_rate(high_df, "autopay_enabled")
        impact = min(74, max(40, 80 - (high_val * 0.4)))
        items.append({
            "title": "자동이체 활성화 여부 (autopay_enabled)",
            "impact": impact,
            "level": "중간 영향",
            "color": "orange",
            "desc": f"전체 자동이체 활성화 비율은 {overall:.1f}%이며, 고위험군에서는 {high_val:.1f}%입니다. 자동이체 비중이 낮을수록 납부 이탈과 관리 부담이 커질 수 있습니다."
        })

    items = sorted(items, key=lambda x: x["impact"], reverse=True)
    return items[:5]


def get_pattern_summary(scored_df: pd.DataFrame):
    high_df = scored_df[scored_df["risk_group"] == "고위험"]
    med_df = scored_df[scored_df["risk_group"] == "중위험"]
    low_df = scored_df[scored_df["risk_group"] == "저위험"]

    high_premium = safe_numeric_mean(high_df, "premium_change_pct") * 100
    high_complaint = safe_rate(high_df, "complaint_flag")
    high_quote = safe_rate(high_df, "quote_requested_flag")
    high_resolution = safe_numeric_mean(high_df, "complaint_resolution_days")

    med_late = safe_numeric_mean(med_df, "late_payment_count_12m")
    med_down = safe_rate(med_df, "coverage_downgrade_flag")
    med_contact = safe_numeric_mean(med_df, "num_contacts_12m")

    low_tenure = safe_numeric_mean(low_df, "customer_tenure_months") / 12
    low_ontime = 100 - safe_rate(low_df, "missed_payment_flag")
    low_multi = safe_rate(low_df, "multi_policy_flag")

    return {
        "high": {
            "text": (
                f"고위험 고객은 보험료 인상과 민원, 타사 견적 탐색이 함께 나타나는 경향이 있습니다. "
                f"평균 보험료 인상률은 {high_premium:.1f}%이며, 민원 경험 비율은 {high_complaint:.1f}%입니다. "
                f"민원 해결 소요일도 평균 {high_resolution:.1f}일로 나타나 고객 불만이 오래 지속될 가능성이 있습니다."
            ),
            "m1_label": "평균 보험료 인상률",
            "m1_value": f"+{high_premium:.1f}%",
            "m2_label": "민원 제기 비율",
            "m2_value": f"{high_complaint:.1f}%",
            "m3_label": "타사 견적 요청 비율",
            "m3_value": f"{high_quote:.1f}%",
        },
        "medium": {
            "text": (
                f"중위험 고객은 납부 연체, 보장 축소 요청, 잦은 문의가 관찰되는 집단입니다. "
                f"즉시 이탈보다는 이탈 전조가 누적되는 단계로, 적절한 개입 시 저위험군으로 전환될 가능성이 있습니다."
            ),
            "m1_label": "평균 연체 횟수",
            "m1_value": f"{med_late:.1f}회",
            "m2_label": "보장 축소 요청 비율",
            "m2_value": f"{med_down:.1f}%",
            "m3_label": "평균 고객 접촉 수",
            "m3_value": f"{med_contact:.1f}회",
        },
        "low": {
            "text": (
                f"저위험 고객은 장기 유지 성향이 높고 납부 안정성이 좋은 집단입니다. "
                f"복수 상품 보유 비율이 높고 전반적인 서비스 마찰이 적어 유지 가능성이 높습니다."
            ),
            "m1_label": "평균 고객 기간",
            "m1_value": f"{low_tenure:.1f}년",
            "m2_label": "정시 납부율",
            "m2_value": f"{low_ontime:.1f}%",
            "m3_label": "복수 계약 비율",
            "m3_value": f"{low_multi:.1f}%",
        }
    }


def get_ai_insights(scored_df: pd.DataFrame):
    high_df = scored_df[scored_df["risk_group"] == "고위험"]

    quote_rate = safe_rate(high_df, "quote_requested_flag")
    complaint_rate = safe_rate(high_df, "complaint_flag")
    premium_rate = safe_numeric_mean(high_df, "premium_change_pct") * 100

    return [
        {
            "cls": "insight-blue",
            "title": "보험료 급등 위험",
            "text": f"고위험 고객군의 평균 보험료 인상률은 {premium_rate:.1f}%입니다. 가격 충격이 커질수록 고객의 이탈 가능성이 함께 높아집니다."
        },
        {
            "cls": "insight-purple",
            "title": "경쟁사 비교 신호",
            "text": f"고위험 고객 중 타사 견적 요청 비율은 {quote_rate:.1f}%입니다. 가격 비교 행동은 실제 이탈 직전의 강한 탐색 신호일 수 있습니다."
        },
        {
            "cls": "insight-yellow",
            "title": "민원 해결 중요성",
            "text": f"고위험 고객 중 민원 경험 비율은 {complaint_rate:.1f}%입니다. 민원 대응 속도와 해결 품질이 고객 유지에 직접적인 영향을 줄 수 있습니다."
        }
    ]


def get_action_text(high_cnt, med_cnt, low_cnt):
    return {
        "high": {
            "title": f"고위험 고객 ({fmt_int(high_cnt)}명)",
            "body": """
• 즉시 리텐션 대상자 분류  
• 보험료 인상 고객 우선 상담  
• 민원 고객 1:1 후속 조치  
• 타사 견적 요청 고객 전용 혜택 제공
"""
        },
        "medium": {
            "title": f"중위험 고객 ({fmt_int(med_cnt)}명)",
            "body": """
• 정기 모니터링 대상 운영  
• 납부 안정화 캠페인 진행  
• 보장 축소 요청 고객 재제안  
• 만족도 설문 및 피드백 수집
"""
        },
        "low": {
            "title": f"저위험 고객 ({fmt_int(low_cnt)}명)",
            "body": """
• 장기 유지 혜택 제공  
• 교차 판매 대상 확대  
• 우수고객 프로그램 연계  
• 추천 캠페인 운영
"""
        }
    }


# =========================================================
# 화면
# =========================================================
try:
    scored_df, selected_threshold, debug_info = prepare_scored_dataframe()

    total_cnt = len(scored_df)
    high_cnt = int((scored_df["risk_group"] == "고위험").sum())
    med_cnt = int((scored_df["risk_group"] == "중위험").sum())
    low_cnt = int((scored_df["risk_group"] == "저위험").sum())

    high_pct = (high_cnt / total_cnt * 100) if total_cnt else 0
    med_pct = (med_cnt / total_cnt * 100) if total_cnt else 0
    low_pct = (low_cnt / total_cnt * 100) if total_cnt else 0

    feature_items = get_feature_insights(scored_df)
    patterns = get_pattern_summary(scored_df)
    ai_insights = get_ai_insights(scored_df)
    actions = get_action_text(high_cnt, med_cnt, low_cnt)

    # 상단 헤더
    left, right = st.columns([6, 1.2])
    with left:
        st.markdown('<div class="report-title">자동 분석 리포트</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="report-subtitle">AI가 분석한 고객 이탈 위험 현황 및 주요 인사이트 · 업데이트: {datetime.now().strftime("%Y년 %m월 %d일")} · 선택 threshold: {selected_threshold:.3f}</div>',
            unsafe_allow_html=True
        )
    with right:
        st.markdown('<div style="height: 12px;"></div>', unsafe_allow_html=True)
        st.markdown('<span class="badge badge-done">분석 완료</span>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # KPI 카드
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="kpi-card kpi-neutral">
            <div class="kpi-label">전체 고객 수</div>
            <div class="kpi-value">{fmt_int(total_cnt)}</div>
            <div class="kpi-sub">100% 전체 고객 기준</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="kpi-card kpi-high">
            <div class="kpi-label">고위험 고객</div>
            <div class="kpi-value">{fmt_int(high_cnt)}</div>
            <div class="kpi-sub">{fmt_pct(high_pct)} 즉시 조치 필요</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="kpi-card kpi-medium">
            <div class="kpi-label">중위험 고객</div>
            <div class="kpi-value">{fmt_int(med_cnt)}</div>
            <div class="kpi-sub">{fmt_pct(med_pct)} 모니터링 필요</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="kpi-card kpi-low">
            <div class="kpi-label">저위험 고객</div>
            <div class="kpi-value">{fmt_int(low_cnt)}</div>
            <div class="kpi-sub">{fmt_pct(low_pct)} 안정적 유지</div>
        </div>
        """, unsafe_allow_html=True)

    # 위험군 분포 분석
    st.markdown('<div class="section-title">위험군 분포 분석</div>', unsafe_allow_html=True)
    dist_left, dist_right = st.columns([1.1, 1.2])

    with dist_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=130)
        sizes = [high_cnt, med_cnt, low_cnt]
        labels = [f"고위험 {high_pct:.1f}%", f"중위험 {med_pct:.1f}%", f"저위험 {low_pct:.1f}%"]
        colors = ["#ef4444", "#f59e0b", "#10b981"]

        ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            startangle=0,
            wedgeprops={"linewidth": 1.2, "edgecolor": "white"},
            textprops={"fontsize": 12}
        )
        ax.axis("equal")
        ax.set_title("현재 위험군 분포", fontsize=15, pad=20)
        st.pyplot(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with dist_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 위험군별 상세 정보")

        groups = [
            ("고위험 고객", high_cnt, high_pct, "red"),
            ("중위험 고객", med_cnt, med_pct, "orange"),
            ("저위험 고객", low_cnt, low_pct, "green"),
        ]

        for title, count, pct, color in groups:
            st.markdown(f"""
            <div style="border:1px solid #e5e7eb; border-radius:16px; padding:1rem 1.1rem; margin-bottom:1rem;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="font-size:1.2rem; font-weight:800; color:#0f172a;">{title}</div>
                    <div style="font-size:1rem; color:#0f172a; border:1px solid #e5e7eb; padding:0.25rem 0.6rem; border-radius:999px;">{fmt_int(count)}명</div>
                </div>
                <div class="progress-wrap">
                    <div class="{bar_class(color)}" style="width:{pct}%;"></div>
                </div>
                <div style="text-align:right; font-size:1rem; color:#334155;">{pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # 주요 이탈 영향 요인
    st.markdown('<div class="section-title">주요 이탈 영향 요인</div>', unsafe_allow_html=True)

    if not feature_items:
        st.info("영향 요인을 생성할 수 있는 컬럼이 부족합니다.")
    else:
        for item in feature_items:
            badge_bg = "#fee2e2" if item["color"] == "red" else "#ffedd5"
            badge_color = "#dc2626" if item["color"] == "red" else "#ea580c"

            st.markdown(f"""
            <div class="feature-card">
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div>
                        <div class="feature-title">{item["title"]}</div>
                        <div style="margin-top:0.8rem;">
                            <span style="display:inline-block; background:{badge_bg}; color:{badge_color}; padding:0.35rem 0.7rem; border-radius:999px; font-weight:700; font-size:0.9rem;">
                                {item["level"]}
                            </span>
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <div style="color:#64748b;">영향도</div>
                        <div class="feature-impact">{int(item["impact"])}%</div>
                    </div>
                </div>
                <div class="progress-wrap">
                    <div class="{bar_class(item["color"])}" style="width:{item["impact"]}%;"></div>
                </div>
                <div class="feature-desc">{item["desc"]}</div>
            </div>
            """, unsafe_allow_html=True)

    # 고객 행동 패턴 분석
    st.markdown('<div class="section-title">고객 행동 패턴 분석</div>', unsafe_allow_html=True)

    p1, p2, p3 = patterns["high"], patterns["medium"], patterns["low"]

    st.markdown(f"""
    <div class="pattern-high">
        <div style="font-size:1.55rem; font-weight:800; color:#991b1b;">고위험 고객의 전형적 패턴</div>
        <div style="font-size:1.08rem; color:#b91c1c; margin-top:0.8rem; line-height:1.8;">{p1["text"]}</div>
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:14px; margin-top:1rem;">
            <div class="small-metric">
                <div class="small-metric-label">{p1["m1_label"]}</div>
                <div class="small-metric-value" style="color:#991b1b;">{p1["m1_value"]}</div>
            </div>
            <div class="small-metric">
                <div class="small-metric-label">{p1["m2_label"]}</div>
                <div class="small-metric-value" style="color:#991b1b;">{p1["m2_value"]}</div>
            </div>
            <div class="small-metric">
                <div class="small-metric-label">{p1["m3_label"]}</div>
                <div class="small-metric-value" style="color:#991b1b;">{p1["m3_value"]}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="hr-space"></div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="pattern-medium">
        <div style="font-size:1.55rem; font-weight:800; color:#9a3412;">중위험 고객의 전형적 패턴</div>
        <div style="font-size:1.08rem; color:#c2410c; margin-top:0.8rem; line-height:1.8;">{p2["text"]}</div>
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:14px; margin-top:1rem;">
            <div class="small-metric">
                <div class="small-metric-label">{p2["m1_label"]}</div>
                <div class="small-metric-value" style="color:#9a3412;">{p2["m1_value"]}</div>
            </div>
            <div class="small-metric">
                <div class="small-metric-label">{p2["m2_label"]}</div>
                <div class="small-metric-value" style="color:#9a3412;">{p2["m2_value"]}</div>
            </div>
            <div class="small-metric">
                <div class="small-metric-label">{p2["m3_label"]}</div>
                <div class="small-metric-value" style="color:#9a3412;">{p2["m3_value"]}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="hr-space"></div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="pattern-low">
        <div style="font-size:1.55rem; font-weight:800; color:#166534;">저위험 고객의 전형적 패턴</div>
        <div style="font-size:1.08rem; color:#15803d; margin-top:0.8rem; line-height:1.8;">{p3["text"]}</div>
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:14px; margin-top:1rem;">
            <div class="small-metric">
                <div class="small-metric-label">{p3["m1_label"]}</div>
                <div class="small-metric-value" style="color:#166534;">{p3["m1_value"]}</div>
            </div>
            <div class="small-metric">
                <div class="small-metric-label">{p3["m2_label"]}</div>
                <div class="small-metric-value" style="color:#166534;">{p3["m2_value"]}</div>
            </div>
            <div class="small-metric">
                <div class="small-metric-label">{p3["m3_label"]}</div>
                <div class="small-metric-value" style="color:#166534;">{p3["m3_value"]}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # AI 핵심 인사이트
    st.markdown('<div class="section-title">AI 핵심 인사이트</div>', unsafe_allow_html=True)
    i1, i2, i3 = st.columns(3)

    for col, insight in zip([i1, i2, i3], ai_insights):
        with col:
            st.markdown(f"""
            <div class="insight-card {insight["cls"]}">
                <div style="font-size:1.42rem; font-weight:800; color:#0f172a;">{insight["title"]}</div>
                <div style="font-size:1.05rem; color:#334155; margin-top:0.9rem; line-height:1.8;">
                    {insight["text"]}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 권장 조치 사항
    st.markdown('<div class="section-title">권장 조치 사항</div>', unsafe_allow_html=True)
    st.markdown('<div class="action-wrap">', unsafe_allow_html=True)

    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown(f"""
        <div class="action-box">
            <div style="font-size:1.45rem; font-weight:800;">{actions["high"]["title"]}</div>
            <div style="margin-top:1rem; font-size:1.03rem; line-height:1.9; white-space:pre-line;">
                {actions["high"]["body"]}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with a2:
        st.markdown(f"""
        <div class="action-box">
            <div style="font-size:1.45rem; font-weight:800;">{actions["medium"]["title"]}</div>
            <div style="margin-top:1rem; font-size:1.03rem; line-height:1.9; white-space:pre-line;">
                {actions["medium"]["body"]}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with a3:
        st.markdown(f"""
        <div class="action-box">
            <div style="font-size:1.45rem; font-weight:800;">{actions["low"]["title"]}</div>
            <div style="margin-top:1rem; font-size:1.03rem; line-height:1.9; white-space:pre-line;">
                {actions["low"]["body"]}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # 샘플 테이블
    st.markdown('<div class="section-title">분석 결과 샘플</div>', unsafe_allow_html=True)
    preview_cols = [
        c for c in [
            "customer_id",
            "churn_probability",
            "risk_group",
            "premium_change_pct",
            "complaint_flag",
            "late_payment_count_12m",
            "quote_requested_flag",
            "coverage_downgrade_flag",
            "autopay_enabled",
            "num_contacts_12m",
        ] if c in scored_df.columns
    ]

    show_df = scored_df[preview_cols].copy()
    if "churn_probability" in show_df.columns:
        show_df = show_df.sort_values("churn_probability", ascending=False)

    st.dataframe(
        show_df.head(30),
        use_container_width=True,
        hide_index=True
    )

    # 디버그 정보
    with st.expander("디버그 정보 보기"):
        st.markdown(f"""
<div class="debug-box">
aligned feature count: {debug_info["aligned_feature_count"]}

first 20 aligned features:
{debug_info["first_20_features"]}
</div>
        """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"자동 분석 리포트 생성 중 오류가 발생했습니다: {e}")