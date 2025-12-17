import streamlit as st
import pandas as pd
import numpy as np
import time
import json
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
# 載入環境變數
load_dotenv()
# ==========================================
# 1. 配置與設置
# ==========================================
st.set_page_config(page_title="Enerflex Asset Guardian", layout="wide", page_icon="🛡️")

# 自定義 CSS: 優化 Metric 顯示與區塊間距
st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #ff4b4b;
    }
    div.block-container {padding-top: 2rem;}
    </style>
    """, unsafe_allow_html=True)
# ==========================================
# 1.5 Azure OpenAI 初始化
# ==========================================
@st.cache_resource
def init_azure_openai():
    """初始化 Azure OpenAI 客戶端"""
    try:
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        if not azure_endpoint or not api_key or not api_version:
            st.error("Azure OpenAI 環境變數未正確設置，請檢查 .env 檔案。")
            return None
        client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version
        )
        return client
    except Exception as e:
        st.error(f"Azure OpenAI 初始化失敗: {str(e)}")
        return None
    
# 全局閾值
ANOMALY_THRESHOLD = 0.15

# ==========================================
# 2. 核心邏輯 (保持不變)
# ==========================================

def load_real_data(file_path="nasa_sample.csv"):
    try:
        df = pd.read_csv(file_path)
        target_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        df = df.rename(columns={target_col: "Vibration (IPS)"})
        if len(df) > 500:
            df = df.tail(500).reset_index(drop=True)
        df["Timestamp"] = df.index
        return df
    except FileNotFoundError:
        st.error(f"找不到檔案: {file_path}")
        return None
    
def get_manual_content():
    return """
    [Ariel JGT/4 Maintenance Manual, Section 5-2]
    Symptom: High frequency vibration on cylinder head.
    Probable Cause: Suction Valve Spring Fatigue.
    Action: Inspect valve seat and replace spring kit (Part# B-1234-VLV).
    
    [Section 5-3]
    Warning Signs:
    - Vibration exceeding 0.15 IPS
    - Frequency spike in 2-4 kHz range
    - Temperature increase near valve assembly
    
    [Section 5-4]
    Recommended Actions:
    1. Immediate shutdown if vibration > 0.20 IPS
    2. Schedule valve inspection within 24 hours
    3. Order replacement parts (Lead time: 2-3 days)
    """

def call_mock_sap_api(part_id):
    time.sleep(0.5)
    response = {
        "status": "success",
        "system": "SAP-S4HANA-PROD",
        "data": {
            "material_id": part_id,
            "description": "KIT, VALVE, SUCTION, JGT/4",
            "plant": "OM01 (Oman Maradi)",
            "qty": 2,
            "loc": "WH-A"
        }
    }
    return response

def diagnose_with_azure_openai(client, vibration_data, manual_context):
    """使用 Azure OpenAI 進行智能診斷"""
    
    # 準備振動數據摘要
    recent_readings = vibration_data.tail(10)['Vibration (IPS)'].tolist()
    max_vibration = vibration_data['Vibration (IPS)'].max()
    avg_vibration = vibration_data['Vibration (IPS)'].mean()
    trend = "increasing" if recent_readings[-1] > recent_readings[0] else "stable/decreasing"
    
    # 構建 Prompt
    prompt = f"""You are an expert maintenance engineer for Enerflex compressor systems.

**Current Situation:**
- Maximum Vibration: {max_vibration:.4f} IPS
- Average Vibration: {avg_vibration:.4f} IPS
- Recent Trend: {trend}
- Threshold: {ANOMALY_THRESHOLD} IPS
- Recent 10 Readings: {[f'{x:.4f}' for x in recent_readings]}

**Reference Manual:**
{manual_context}

**Task:**
Provide a concise diagnostic report including:
1. Root Cause Analysis (2-3 sentences)
2. Severity Level (Low/Medium/High/Critical)
3. Recommended Actions (numbered list, max 3 items)
4. Estimated Downtime if not addressed

Response must be valid JSON only with this exact structure:
{{
    "root_cause": "your analysis here",
    "severity": "High",
    "actions": ["action 1", "action 2", "action 3"],
    "downtime_risk": "estimated timeframe"
}}
"""

    try:
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=[
                {"role": "system", "content": "You are a specialized AI assistant for industrial equipment diagnostics. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        # 解析回應
        result_text = response.choices[0].message.content.strip()
        
        # 移除可能的 markdown code block 標記
        if result_text.startswith("```json"):
            result_text = result_text.replace("```json", "").replace("```", "").strip()
        elif result_text.startswith("```"):
            result_text = result_text.replace("```", "").strip()
        
        # 嘗試解析 JSON
        try:
            diagnosis = json.loads(result_text)
            
            # 驗證必要欄位
            if not all(key in diagnosis for key in ['root_cause', 'severity', 'actions', 'downtime_risk']):
                raise ValueError("Missing required fields")
                
            return diagnosis
            
        except (json.JSONDecodeError, ValueError) as e:
            # JSON 解析失敗，返回預設結構
            st.warning(f"AI response parsing issue, using fallback format")
            return {
                "root_cause": "Suction Valve Spring Fatigue based on vibration pattern analysis",
                "severity": "High",
                "actions": [
                    "Immediate shutdown if vibration exceeds 0.20 IPS",
                    "Schedule valve inspection within 24 hours", 
                    "Order replacement parts (Part# B-1234-VLV)"
                ],
                "downtime_risk": "3-5 days if not addressed promptly"
            }
        
    except Exception as e:
        st.error(f"AI 診斷失敗: {str(e)}")
        return {
            "root_cause": "System diagnostic error - manual inspection required",
            "severity": "High",
            "actions": ["Contact maintenance team immediately"],
            "downtime_risk": "Unknown"
        }
# ==========================================
# 3. Streamlit UI (優化版佈局)
# ==========================================

st.title("🛡️ Enerflex Asset Guardian | Cognitive Maintenance")
azure_client = init_azure_openai()
# --- 上層：監控面板 (Top Monitor) ---
# 比例 3:1，讓圖表寬一點，指標在旁邊
top_col1, top_col2 = st.columns([3, 1])

with top_col1:
    st.subheader("📡 Zone 1: Real-time Monitor")
    chart_placeholder = st.empty()

with top_col2:
    st.subheader("📊 Status")
    metric_placeholder = st.empty()
    status_placeholder = st.empty() # 用來顯示 "Running" 或 "Alert"
    run_btn = st.button("▶️ Start Simulation", type="primary", use_container_width=True)

# 變數初始化
if 'simulation_df' not in st.session_state:
    st.session_state['simulation_df'] = None # 用來存圖表數據

if 'data_finished' not in st.session_state:
    st.session_state['data_finished'] = False
if 'final_val' not in st.session_state:
    st.session_state['final_val'] = 0.0
if 'ai_diagnosis' not in st.session_state:
    st.session_state['ai_diagnosis'] = None

# --- 執行模擬邏輯 ---
if run_btn:
    # 重置狀態
    st.session_state['sap_checked'] = False
    st.session_state['data_finished'] = False
    st.session_state['ai_diagnosis'] = None
    
    # 生成數據
    dummy_df = pd.DataFrame({
        "Timestamp": range(100),
        "bearing_1": np.concatenate([
            np.random.normal(0.06, 0.002, 70), 
            np.linspace(0.06, 0.2, 30) + np.random.normal(0, 0.01, 30) 
        ])
    })
    dummy_df.to_csv("nasa_sample.csv", index=False)
    data = load_real_data("nasa_sample.csv")

    if data is not None:
        status_placeholder.info("System Running...")
        for i in range(1, len(data)):
            current_df = data.iloc[:i]
            # 更新圖表
            chart_placeholder.line_chart(current_df.set_index("Timestamp"), height=300)
            
            val = current_df.iloc[-1]["Vibration (IPS)"]
            
            # 更新指標
            delta_color = "normal" if val < ANOMALY_THRESHOLD else "inverse"
            metric_placeholder.metric(
                "Vibration (IPS)", 
                f"{val:.3f}", 
                delta=f"{val-0.06:.3f}", 
                delta_color=delta_color
            )
            time.sleep(0.06) # 加快一點速度
        
        st.session_state['data_finished'] = True
        st.session_state['final_val'] = val
        st.session_state['simulation_df'] = data


# --- 下層：決策戰情室 (Bottom Action Center) ---
# 只有在數據跑完且有異常時才顯示
if st.session_state['simulation_df'] is not None:
    # 畫最後一張靜態圖
    chart_placeholder.line_chart(st.session_state['simulation_df'].set_index("Timestamp"), height=300)
    
    # 顯示最後的 Metric
    val = st.session_state['final_val']
    delta_color = "normal" if val < ANOMALY_THRESHOLD else "inverse"
    metric_placeholder.metric("Vibration (IPS)", f"{val:.3f}", delta=f"{val-0.06:.3f}", delta_color=delta_color)
    if val > ANOMALY_THRESHOLD:
        status_placeholder.error("⛔ CRITICAL ALERT")
        
        st.divider() # 分隔線
        st.subheader("🧠 Zone 2 & 3: Incident Response Center")
        
        # 這裡將下面分為左右兩半：左邊是 AI 腦，右邊是 SAP 手
        action_col1, action_col2 = st.columns(2, gap="medium")
        
        # === 左下：AI 診斷 ===
        # with action_col1:
        #     st.info("🤖 **Step 1: AI Diagnosis (RAG Engine)**")
            
        #     # 使用 status 元件讓 loading 更好看
        #     with st.status("Analyzing vibration patterns...", expanded=True) as status:
        #         time.sleep(1)
        #         manual_text = get_manual_content()
        #         status.update(label="Diagnosis Complete", state="complete", expanded=False)
            
        #     st.success("**Root Cause:** Suction Valve Spring Fatigue")
            
        #     with st.expander("📄 View Retrieved Context (Evidence)", expanded=True):
        #         st.code(manual_text, language="text")
        
        with action_col1:
            st.info("🤖 **Step 1: AI Diagnosis (Azure OpenAI + RAG)**")
            
            # 只在首次運行 AI 診斷
            if st.session_state['ai_diagnosis'] is None and azure_client:
                with st.status("Analyzing with Azure OpenAI...", expanded=True) as status:
                    manual_text = get_manual_content()
                    diagnosis = diagnose_with_azure_openai(
                        azure_client, 
                        st.session_state['simulation_df'], 
                        manual_text
                    )
                    st.session_state['ai_diagnosis'] = diagnosis
                    status.update(label="AI Analysis Complete ✨", state="complete", expanded=False)
            
            # 顯示診斷結果
            if st.session_state['ai_diagnosis']:
                diag = st.session_state['ai_diagnosis']
                
                # 顯示嚴重程度
                severity_colors = {
                    "Low": "🟢",
                    "Medium": "🟡", 
                    "High": "🟠",
                    "Critical": "🔴"
                }
                severity_icon = severity_colors.get(diag.get('severity', 'High'), "🔴")
                st.warning(f"{severity_icon} **Severity:** {diag.get('severity', 'High')}")
                
                # 根因分析 - 修復這裡
                root_cause_text = diag.get('root_cause', 'Analysis in progress')
                st.success(f"**Root Cause:** {root_cause_text}")
                
                # 建議行動
                if 'actions' in diag and isinstance(diag['actions'], list):
                    st.markdown("**Recommended Actions:**")
                    for idx, action in enumerate(diag['actions'], 1):
                        st.markdown(f"{idx}. {action}")
                
                # 停機風險
                if 'downtime_risk' in diag:
                    st.error(f"⚠️ **Downtime Risk:** {diag['downtime_risk']}")
                
                # 顯示 RAG 檢索到的原始內容
                with st.expander("📄 Retrieved Manual Context", expanded=False):
                    st.code(get_manual_content(), language="text")
            
            elif not azure_client:
                st.error("Azure OpenAI 未配置，使用基礎診斷模式")
                st.success("**Root Cause:** Suction Valve Spring Fatigue (Basic Mode)")
                
                # 基礎模式也顯示手動內容
                with st.expander("📄 View Retrieved Context (Evidence)", expanded=True):
                    st.code(get_manual_content(), language="text")
        # === 右下：SAP 執行 ===
        with action_col2:
            st.warning("🏢 **Step 2: SAP Execution (ERP Bridge)**")
            
            # 初始化
            if 'sap_checked' not in st.session_state:
                st.session_state['sap_checked'] = False

            # 按鈕 1: 查庫存
            if st.button("🔍 Check SAP Inventory (MM Module)", use_container_width=True):
                st.session_state['sap_checked'] = True
            
            if st.session_state['sap_checked']:
                sap_data = call_mock_sap_api("B-1234-VLV")
                
                # 使用 col 讓 JSON 和結果並排顯示，節省空間
                res_c1, res_c2 = st.columns([1, 1])
                with res_c1:
                    with st.expander("View API JSON", expanded=False): # 預設收起 JSON
                        st.json(sap_data)
                with res_c2:
                    if sap_data['data']['qty'] > 0:
                        st.success(f"✅ Stock: {sap_data['data']['qty']} EA")
                    else:
                        st.error("Out of Stock")

                # Human-in-the-Loop 區域
                st.markdown("**👷 Engineer Approval**")
                engineer_notes = st.text_area("Field Notes", "Confirmed valve issue. Proceed.", height=80)
                
                # 按鈕 2: 開單
                if st.button("🚀 Approve & Create Work Order (PM Module)", type="primary", use_container_width=True):
                    st.toast("Connecting to SAP S/4HANA...", icon="⏳")
                    time.sleep(1)
                    st.balloons()
                    st.success(f"✅ PM Order Created! [Ref: {int(time.time())}]")
                    st.caption(f"Logged Notes: {engineer_notes}")

    else:
        status_placeholder.success("✅ Normal Operation")
        st.success("Equipment is running within optimal parameters.")