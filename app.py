import streamlit as st
import pandas as pd
import numpy as np
import time
import json

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

# ==========================================
# 3. Streamlit UI (優化版佈局)
# ==========================================

st.title("🛡️ Enerflex Asset Guardian | Cognitive Maintenance")

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

# --- 執行模擬邏輯 ---
if run_btn:
    # 重置狀態
    st.session_state['sap_checked'] = False
    st.session_state['data_finished'] = False
    
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
        with action_col1:
            st.info("🤖 **Step 1: AI Diagnosis (RAG Engine)**")
            
            # 使用 status 元件讓 loading 更好看
            with st.status("Analyzing vibration patterns...", expanded=True) as status:
                time.sleep(1)
                manual_text = get_manual_content()
                status.update(label="Diagnosis Complete", state="complete", expanded=False)
            
            st.success("**Root Cause:** Suction Valve Spring Fatigue")
            
            with st.expander("📄 View Retrieved Context (Evidence)", expanded=True):
                st.code(manual_text, language="text")

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