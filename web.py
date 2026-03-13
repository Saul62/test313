from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    import shap
except Exception:
    shap = None

st.set_page_config(
    page_title="ICU谵妄风险预测系统",
    page_icon="🏥",
    layout="wide",
)

APP_TITLE = "重症肺炎患者谵妄风险预测模型"
APP_SUBTITLE = ""

BASE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    BASE_DIR / "lightgbm_delirium_model.pkl",
    BASE_DIR / "results" / "lightgbm_delirium_model.pkl",
]

FEATURE_SPECS = [
    {
        "key": "mbp",
        "label": "平均动脉压",
        "unit": "mmHg",
        "min": 44.0,
        "max": 130.0,
        "default": 76.3,
        "step": 0.1,
        "help": "mbp：平均动脉压",
    },
    {
        "key": "temperature",
        "label": "体温",
        "unit": "°C",
        "min": 35.7,
        "max": 39.0,
        "default": 36.9,
        "step": 0.1,
        "help": "temperature：体温",
    },
    {
        "key": "po2_max",
        "label": "氧分压最大值",
        "unit": "mmHg",
        "min": 15.0,
        "max": 465.0,
        "default": 152.0,
        "step": 1.0,
        "help": "po2_max：氧分压最大值",
    },
    {
        "key": "sedatives_hosp_firstday_used",
        "label": "入ICU后24h内镇静药物使用",
        "type": "select",
        "options": {"否": 0.0, "是": 1.0},
        "default": "是",
        "help": "0=否，1=是",
    },
    {
        "key": "NLR",
        "label": "NLR",
        "unit": "",
        "min": 0.0,
        "max": 125.6,
        "default": 12.6,
        "step": 0.1,
        "help": "NLR",
    },
    {
        "key": "SII",
        "label": "SII",
        "unit": "",
        "min": 0.0,
        "max": 46452.0,
        "default": 1956.4,
        "step": 1.0,
        "help": "SII",
    },
    {
        "key": "SIRI",
        "label": "SIRI",
        "unit": "",
        "min": 0.0,
        "max": 86.9,
        "default": 8.9,
        "step": 0.1,
        "help": "SIRI",
    },
    {
        "key": "sodium_max",
        "label": "血钠最大值",
        "unit": "mEq/L",
        "min": 120.0,
        "max": 162.0,
        "default": 140.0,
        "step": 0.1,
        "help": "sodium_max：血钠最大值",
    },
    {
        "key": "ventilation_status",
        "label": "通气方式",
        "type": "select",
        "options": {
            "1 基础氧疗": 1.0,
            "2 高流量通气": 2.0,
            "3 无创通气": 3.0,
            "4 有创通气": 4.0,
        },
        "default": "2 高流量通气",
        "help": "1=基础氧疗，2=高流量通气，3=无创通气，4=有创通气",
    },
    {
        "key": "pao2fio2ratio_min",
        "label": "氧合指数最低值",
        "unit": "mmHg",
        "min": 20.0,
        "max": 603.4,
        "default": 122.0,
        "step": 0.1,
        "help": "pao2fio2ratio_min：氧合指数最低值",
    },
    {
        "key": "spo2",
        "label": "血氧饱和度",
        "unit": "%",
        "min": 81.0,
        "max": 100.0,
        "default": 97.0,
        "step": 0.1,
        "help": "spo2：血氧饱和度",
    },
    {
        "key": "apsiii",
        "label": "APSIII评分",
        "unit": "",
        "min": 7.0,
        "max": 142.3,
        "default": 50.0,
        "step": 0.1,
        "help": "apsiii：APSIII评分",
    },
    {
        "key": "pt_min",
        "label": "凝血酶原时间最小值",
        "unit": "s",
        "min": 8.0,
        "max": 32.5,
        "default": 13.6,
        "step": 0.1,
        "help": "pt_min：凝血酶原时间最小值",
    },
    {
        "key": "abs_lymphocytes_max",
        "label": "绝对淋巴细胞计数最大值",
        "unit": "",
        "min": 0.0,
        "max": 8.4,
        "default": 0.85,
        "step": 0.01,
        "help": "abs_lymphocytes_max：绝对淋巴细胞计数最大值",
    },
]

FEATURE_LABELS = {spec["key"]: spec["label"] for spec in FEATURE_SPECS}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(17, 112, 255, 0.12), transparent 28%),
                radial-gradient(circle at top right, rgba(0, 153, 129, 0.10), transparent 22%),
                linear-gradient(180deg, #f6f9fc 0%, #eef4f7 100%);
        }
        .hero-card {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(13, 61, 89, 0.08);
            border-radius: 20px;
            padding: 20px 24px;
            box-shadow: 0 10px 30px rgba(15, 45, 70, 0.08);
            backdrop-filter: blur(8px);
            margin-bottom: 18px;
        }
        .hero-title {
            font-size: 2rem;
            font-weight: 800;
            color: #0f3550;
            margin-bottom: 0.2rem;
        }
        .hero-subtitle {
            color: #47657c;
            font-size: 1rem;
            margin-bottom: 0;
        }
        .risk-banner {
            border-radius: 18px;
            padding: 18px 20px;
            color: white;
            font-weight: 700;
            margin-bottom: 12px;
        }
        .risk-low { background: linear-gradient(135deg, #0b9e6f, #28c78c); }
        .risk-mid { background: linear-gradient(135deg, #ff9f1c, #ffbf69); color: #372100; }
        .risk-high { background: linear-gradient(135deg, #df3f5c, #ff6b6b); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def find_model_file() -> Path:
    for path in MODEL_CANDIDATES:
        if path.exists():
            return path
    checked = "\n".join(str(path) for path in MODEL_CANDIDATES)
    raise FileNotFoundError(f"未找到模型文件，请检查以下路径：\n{checked}")


@st.cache_resource
def load_artifact() -> dict[str, Any]:
    model_path = find_model_file()
    artifact = joblib.load(model_path)
    if isinstance(artifact, dict) and "model" in artifact:
        artifact["model_path"] = str(model_path)
        return artifact

    return {
        "model": artifact,
        "feature_columns": list(getattr(artifact, "feature_names_in_", [spec["key"] for spec in FEATURE_SPECS])),
        "label_column": "delirium",
        "model_path": str(model_path),
    }


def render_sidebar_info() -> None:
    st.sidebar.markdown("## 模型说明")
    st.sidebar.markdown(
        "本模型基于 LightGBM 算法构建，使用多种系统炎症标志物及临床指标预测重症肺炎患者谵妄风险。"
        "输出为谵妄发生概率（0~1），建议以 0.5 为临床决策阈值。"
    )


def get_risk_level(probability: float) -> tuple[str, str, str]:
    if probability < 0.50:
        return "低风险", "risk-low", "当前预测概率低于0.5，建议继续观察并定期评估谵妄风险。"
    return "高风险", "risk-high", "当前预测概率已达到或超过 0.5，建议重点关注谵妄风险并尽早干预。"


def build_input_form(feature_columns: list[str]) -> tuple[bool, dict[str, float]]:
    specs_by_key = {spec["key"]: spec for spec in FEATURE_SPECS}
    inputs: dict[str, float] = {}

    with st.form("delirium_predict_form", clear_on_submit=False):
        cols = st.columns(2)
        for idx, feature in enumerate(feature_columns):
            spec = specs_by_key[feature]
            with cols[idx % 2]:
                if spec.get("type") == "select":
                    options = list(spec["options"].keys())
                    default_index = options.index(spec["default"])
                    selected = st.selectbox(
                        spec["label"],
                        options=options,
                        index=default_index,
                        help=spec["help"],
                        key=f"field_{feature}",
                    )
                    inputs[feature] = float(spec["options"][selected])
                else:
                    label = spec["label"] if not spec.get("unit") else f"{spec['label']} ({spec['unit']})"
                    inputs[feature] = float(
                        st.number_input(
                            label,
                            min_value=float(spec["min"]),
                            max_value=float(spec["max"]),
                            value=float(spec["default"]),
                            step=float(spec["step"]),
                            help=spec["help"],
                            key=f"field_{feature}",
                        )
                    )

        submitted = st.form_submit_button("开始预测", use_container_width=True)

    return submitted, inputs


def render_shap_force_plot(model: Any, input_df: pd.DataFrame) -> None:
    if shap is None:
        st.info("当前环境未安装 `shap`，已跳过 SHAP 力图。")
        return

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    if isinstance(shap_values, list):
        shap_row = np.asarray(shap_values[-1])[0]
    else:
        shap_array = np.asarray(shap_values)
        if shap_array.ndim == 3:
            shap_row = shap_array[0, :, -1]
        else:
            shap_row = shap_array[0]

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = float(np.asarray(expected_value).reshape(-1)[-1])
    else:
        expected_value = float(expected_value)

    display_features = input_df.rename(columns=FEATURE_LABELS)
    force_plot = shap.force_plot(
        expected_value,
        shap_row,
        display_features.iloc[0],
        matplotlib=False,
        show=False,
    )
    html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
    components.html(html, height=320, scrolling=True)


def main() -> None:
    inject_styles()

    try:
        artifact = load_artifact()
    except Exception as exc:
        st.error(f"模型加载失败：{exc}")
        return

    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-title">{APP_TITLE}</div>
            <p class="hero-subtitle">{APP_SUBTITLE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_sidebar_info()

    st.markdown("### 患者指标录入")
    submitted, user_inputs = build_input_form(feature_columns)

    if not submitted:
        return

    input_df = pd.DataFrame([[user_inputs[col] for col in feature_columns]], columns=feature_columns)

    try:
        probability = float(model.predict_proba(input_df)[0, 1])
    except Exception as exc:
        st.error(f"预测失败：{exc}")
        return

    risk_name, risk_class, risk_text = get_risk_level(probability)
    st.markdown("### 预测结果")
    st.markdown(
        f"""
        <div class="risk-banner {risk_class}">
            预测结论：{risk_name} | 谵妄发生概率 {probability:.1%}
            <div style="font-size:0.95rem;font-weight:500;margin-top:6px;">{risk_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### SHAP力图")
    try:
        render_shap_force_plot(model, input_df)
    except Exception as exc:
        st.warning(f"SHAP 力图生成失败：{exc}")


if __name__ == "__main__":
    main()
