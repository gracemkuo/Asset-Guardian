import streamlit as st
import pandas as pd
import numpy as np
import time
from openai import AzureOpenAI

# ==========================================
# 1. 配置與設置 (Configuration)
# ==========================================
st.set_page_config(page_title="Enerflex Asset Guardian", layout="wide")

# 這裡填入你的 Azure OpenAI 資訊 (面試演示時若怕網路問題，可寫一個 Mock 函數切換)
# client = AzureOpenAI(
#     api_key="YOUR_KEY",
#     api_version="2024-02-15-preview",
#     azure_endpoint="YOUR_ENDPOINT"
# )

# ==========================================
# 2. 後端邏輯模擬 (The "Left Brain" & "The Bridge")
# ==========================================

def simulate_sensor_data(steps=50):
    """
    模擬 Ariel JGT/4 壓縮機震動數據。
    前段正常，後段出現線性漂移 (Drift)，模擬閥門疲勞。
    """
    normal_vibration = np.random.normal(loc=0.5, scale=0.05, size=steps-15)
    # 異常漂移：震動值逐漸升高，但尚未觸發高高報警 (High-High Alarm)
    drifting_vibration = np.linspace(0.5, 0.95, 15) + np.random.normal(loc=0.0, scale=0.05, size=15)
    
    data = np.concatenate([normal_vibration, drifting_vibration])
    df = pd.DataFrame({"Timestamp": range(steps), "Vibration (IPS)": data})
    return df

def check_sap_inventory(part_id):
    """
    [The Bridge] 模擬查詢 SAP MM 模組
    """
    # 模擬 SAP 數據庫返回
    sap_db = {
        "B-1234-VLV": {"name": "Suction Valve Kit, JGT/4", "stock": 2, "warehouse": "Oman-Maradi-WH1"},
        "S-9988-SEAL": {"name": "Rod Packing Seal", "stock": 0, "warehouse": "Oman-Maradi-WH1"}
    }
    return sap_db.get(part_id, None)

def generate_diagnosis(vibration_level):
    """
    [The Right Brain] 呼叫 Azure OpenAI 進行 RAG 診斷
    """
    # 這裡演示用的 Prompt，實際專案會包含 RAG 檢索到的 Context
    prompt = f"""
    Context: 
    - Equipment: Ariel JGT/4 Compressor at Oman Maradi Field.
    - Sensor: Vibration sensor on Cylinder #2.
    - Current Value: {vibration_level:.2f} IPS (Trending Up).
    - Historical Log: Last valve maintenance was 6 months ago.
    
    Task:
    Analyze the vibration drift. Identify the likely root cause based on Ariel manuals.
    Suggest the specific part number for suction valve repair.
    Keep it concise.
    """
    
    # 若無 API Key，使用預設回應確保 Demo 順暢
    return """
    **診斷分析 (Diagnosis):**
    根據震動趨勢 (Trend Analysis) 顯示，2號氣缸出現漸進式震動升高。這與 **吸氣閥彈簧疲勞 (Suction Valve Spring Fatigue)** 的特徵高度吻合 (Ariel Manual Sec 5.2)。這並非突發故障，而是效能衰退。
    
    **建議行動 (Prescription):**
    建議在計畫性停機期間更換吸氣閥組件。
    
    **所需備件 (Part Identification):**
    Part No: B-1234-VLV (Suction Valve Kit)
    """

# ==========================================
# 3. 前端介面 (Streamlit UI)
# ==========================================

st.title("🛡️ Enerflex Asset Guardian | Oman Pilot")
st.markdown("**Site:** Maradi Huraymah | **Unit:** C-201 (Ariel JGT/4) | **Status:** :orange[Warning]")

col1, col2 = st.columns([2, 1])

# --- 左側：即時數據監控 (The Analyst) ---
with col1:
    st.subheader("1. 實時震動監控 (Real-time SCADA Feed)")
    
    if st.button("啟動模擬數據流 (Simulate Stream)"):
        data = simulate_sensor_data()
        chart_placeholder = st.empty()
        
        # 模擬數據流動效果
        for i in range(1, len(data)):
            current_df = data.iloc[:i]
            chart_placeholder.line_chart(current_df.set_index("Timestamp"))
            time.sleep(0.05)
            
        current_val = data.iloc[-1]["Vibration (IPS)"]
        st.session_state['last_val'] = current_val
        
        if current_val > 0.8:
            st.error(f"⚠️ 偵測到異常漂移 (Drift Detected)! 當前數值: {current_val:.2f} IPS")
            st.session_state['anomaly'] = True
        else:
            st.success("系統運轉正常")

# --- 右側：AI 診斷與 SAP 整合 (The Expert & The Bridge) ---
with col2:
    st.subheader("2. Cognitive Maintenance Engine")
    
    if st.session_state.get('anomaly'):
        with st.spinner("AI 正在檢索 Ariel 手冊與歷史日誌 (RAG)..."):
            time.sleep(1.5) # 模擬運算時間
            diagnosis = generate_diagnosis(st.session_state['last_val'])
            st.markdown(diagnosis)
        
        # --- SAP 整合關鍵部分 ---
        st.divider()
        st.subheader("3. SAP 自動化流程 (The Bridge)")
        
        if st.button("執行 SAP 庫存檢查 (Check SAP MM)"):
            part_id = "B-1234-VLV"
            inventory = check_sap_inventory(part_id)
            
            if inventory:
                st.success(f"✅ SAP 庫存確認: {inventory['stock']} EA")
                st.info(f"倉庫位置: {inventory['warehouse']}")
                
                with st.expander("查看自動生成的工單 (Draft Work Order)", expanded=True):
                    st.markdown(f"""
                    **SAP PM Notification #20251024**
                    - **Type:** Predictive Maintenance
                    - **Asset:** C-201
                    - **Material:** {part_id} ({inventory['name']})
                    - **Priority:** High (Pre-emptive)
                    """)
                    if st.button("批准並發送至維修團隊 (Approve to SAP)"):
                        st.balloons()
                        st.toast("工單已同步至 SAP S/4HANA!", icon="🚀")
            else:
                st.error("SAP 庫存不足，已自動觸發採購申請 (PR)")