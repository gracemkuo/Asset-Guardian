import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
from datetime import datetime, timedelta

# ==========================================
# CONFIGURATION & MOCK DATA
# ==========================================
st.set_page_config(page_title="Enerflex Asset Guardian", layout="wide")

# DEMO_MODE: True means we use simulated LLM responses (Safest for interviews). 
# False would require actual Azure OpenAI API keys.
DEMO_MODE = True 

# 模擬的 RAG 知識庫 (Ariel JGT/4 Manual Snippets)
KNOWLEDGE_BASE = {
    "Vib_High_Cyl2": """
    [Source: Ariel JGT/4 Maintenance Manual, Section 5.2]
    Symptom: High frequency vibration on Cylinder 2 throw.
    Probable Cause: Valve plate fatigue or broken spring in Suction Valve.
    Action: Inspect suction valve assembly. Replace if debris found.
    Part Number: B-5732-K (Suction Valve Kit).
    """
}

# 模擬的 SAP ERP 庫存數據
SAP_DATABASE = {
    "B-5732-K": {"name": "Suction Valve Kit (JGT/4)", "stock": 4, "location": "Oman Warehouse A", "lead_time": "1 Day"},
    "A-1102-X": {"name": "Piston Ring Set", "stock": 0, "location": "Houston HQ", "lead_time": "14 Days"}
}

# ==========================================
# 1. LEFT BRAIN: The Analyst (Data Simulation)
# ==========================================
def generate_sensor_data(n_points=100, drift=False):
    """
    Simulates sensor data. 
    If drift=True, adds a gradual increase to simulate 'Data Drift' before failure.
    Purpose: Demonstrate LSTM's ability to catch trends before thresholds.
    """
    dates = [datetime.now() - timedelta(minutes=x) for x in range(n_points)]
    dates.reverse()
    
    # Base vibration signal (Normal operation around 0.3 IPS)
    noise = np.random.normal(0, 0.02, n_points)
    values = 0.3 + noise
    
    if drift:
        # Simulate a linear drift starting from the middle of the dataset
        drift_values = np.linspace(0, 0.4, n_points)
        values = values + drift_values
        
    return pd.DataFrame({"Timestamp": dates, "Vibration (IPS)": values})

# ==========================================
# 2. RIGHT BRAIN: The Expert (RAG Engine)
# ==========================================
def query_cognitive_engine(anomaly_context):
    """
    Simulates the RAG (Retrieval-Augmented Generation) process.
    Input: Anomaly data (Context).
    Output: Natural language diagnosis.
    """
    if DEMO_MODE:
        time.sleep(1.5) # Simulate processing time
        
        # 1. Retrieve
        retrieved_doc = KNOWLEDGE_BASE["Vib_High_Cyl2"]
        
        # 2. Generate (Simulated LLM Output)
        response = f"""
        **Diagnosis:** Based on the vibration drift pattern in Cylinder 2, the Cognitive Engine has detected a **Suction Valve Failure**.
        
        **Evidence:**
        * Model Confidence: 98.2% (LSTM Drift Detection)
        * Retrieved Context: "{retrieved_doc.strip()}"
        
        **Recommendation:**
        Isolate Unit 3 immediately. Inspect Cylinder 2 Suction Valve. Prepare Part #B-5732-K.
        """
        return response, "B-5732-K"
    else:
        # Here you would put actual Azure OpenAI / LangChain code
        return "LLM API Call Placeholder", None

# ==========================================
# 3. EXECUTION: SAP Integration
# ==========================================
def check_sap_inventory(part_id):
    """
    Simulates API call to SAP ERP system.
    """
    time.sleep(0.5) # Simulate API latency
    return SAP_DATABASE.get(part_id, None)

# ==========================================
# MAIN INTERFACE (Streamlit)
# ==========================================
def main():
    # --- Sidebar ---
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png", width=50) # Placeholder logo
    st.sidebar.title("Simulation Controls")
    simulation_state = st.sidebar.radio("Scenario:", ["Normal Operation", "Anomaly Detected (Drift)"])
    
    # --- Header ---
    st.title("🛡️ Enerflex Asset Guardian")
    st.markdown("### Cognitive Maintenance Pilot | Oman - Maradi Huraymah Field")
    st.markdown("---")

    # --- Dashboard Layout ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📡 Real-time Sensor Telemetry (Ariel JGT/4)")
        
        # Determine data state
        is_drift = (simulation_state == "Anomaly Detected (Drift)")
        df = generate_sensor_data(drift=is_drift)
        
        # Plotting
        fig = px.line(df, x='Timestamp', y='Vibration (IPS)', title="Vibration Sensor - Cylinder 2")
        
        # Add Threshold Line (Visualizing Reactive vs Proactive)
        fig.add_hline(y=0.6, line_dash="dash", line_color="red", annotation_text="SCADA Trip Threshold (0.6 IPS)")
        
        if is_drift:
            # Highlight the drift area
            fig.add_vrect(x0=df['Timestamp'].iloc[50], x1=df['Timestamp'].iloc[-1], 
                          fillcolor="yellow", opacity=0.1, annotation_text="AI Detected Drift")
            st.plotly_chart(fig, use_container_width=True)
            
            # Simulated LSTM Alert
            st.error("🚨 ALERT: LSTM Model detected anomaly drift (Confidence: 98.2%). Triggering Cognitive Diagnosis...")
        else:
            st.plotly_chart(fig, use_container_width=True)
            st.success("✅ System Status: Healthy. Model Drift: 0.02% (Normal).")

    with col2:
        st.subheader("🧠 Cognitive Engine (RAG)")
        
        if is_drift:
            with st.spinner('Analyzing sensor patterns & Retrieving Manuals...'):
                diagnosis_report, part_id = query_cognitive_engine("High Vibration Cyl 2")
            
            # Display RAG Result
            st.markdown("### 🤖 AI Diagnosis")
            st.info(diagnosis_report)
            
            st.markdown("---")
            st.subheader("🛠️ Human-in-the-Loop Action")
            
            if st.button("Approve & Check SAP Inventory"):
                st.write(f"Querying SAP for Part ID: **{part_id}**...")
                part_info = check_sap_inventory(part_id)
                
                if part_info:
                    st.success(f"📦 **Stock Available!**")
                    st.json(part_info)
                    st.button("🚀 Execute Work Order #OM-2024-998")
                else:
                    st.warning("Part not found.")
        else:
            st.markdown("*Waiting for alerts...*")
            st.markdown("The system is monitoring 24/7. No intervention needed.")

if __name__ == "__main__":
    main()