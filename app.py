import streamlit as st
import pandas as pd
import numpy as np
import time
import json
from openai import AzureOpenAI
import os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==========================================
# 1. Configuration and Settings
# ==========================================
st.set_page_config(page_title="Enerflex Asset Guardian", layout="wide", page_icon="🛡️")

# Custom CSS
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
# 1.5 Azure OpenAI Initialization
# ==========================================
@st.cache_resource
def init_azure_openai():
    """Initialize Azure OpenAI client"""
    try:
        azure_endpoint = st.secrets["AZURE_OPENAI_ENDPOINT"]
        api_key = st.secrets["AZURE_OPENAI_API_KEY"]
        api_version = st.secrets["AZURE_OPENAI_API_VERSION"]
        if not azure_endpoint or not api_key or not api_version:
            st.error("Azure OpenAI configuration not set")
            return None
        client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version
        )
        return client
    except Exception as e:
        st.error(f"Azure OpenAI initialization failed: {str(e)}")
        return None
    
# Global thresholds
ANOMALY_THRESHOLD = 0.15      # AI alert threshold (Drift)
SCADA_TRIP_THRESHOLD = 0.6    # SCADA trip threshold (Red line)

# ==========================================
# New: Function for creating professional charts
# ==========================================
def create_vibration_chart(df, show_thresholds=True):
    """
    Create professional vibration monitoring chart using Plotly
    """
    # Generate timestamps (counting back from current time)
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=2)
    
    # Ensure df has correct length
    num_points = len(df)
    timestamps = pd.date_range(start=start_time, end=end_time, periods=num_points)

    # Create chart
    fig = go.Figure()

    # Main data line
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=df['Vibration (IPS)'],
        mode='lines',
        name='Vibration',
        line=dict(color='#1f77b4', width=2.5),
        hovertemplate='<b>Time:</b> %{x|%H:%M}<br><b>Vibration:</b> %{y:.4f} IPS<extra></extra>',
        showlegend=False
    ))
    
    if show_thresholds:
        # AI alert threshold (orange dotted line) - placed below data line
        fig.add_hline(
            y=ANOMALY_THRESHOLD,
            line_dash="dot",
            line_color="orange",
            line_width=2,
            annotation_text=f"AI Detected Drift ({ANOMALY_THRESHOLD} IPS)",
            annotation_position="right",
            annotation=dict(font_size=11, font_color="orange")
        )
        
        # SCADA trip threshold (red dashed line)
        fig.add_hline(
            y=SCADA_TRIP_THRESHOLD,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text=f"SCADA Trip Threshold ({SCADA_TRIP_THRESHOLD} IPS)",
            annotation_position="right",
            annotation=dict(font_size=11, font_color="red")
        )
        
        # If exceeding AI threshold, mark yellow zone
        if df['Vibration (IPS)'].max() > ANOMALY_THRESHOLD:
            # Find the first point exceeding threshold
            exceed_mask = df['Vibration (IPS)'] > ANOMALY_THRESHOLD
            if exceed_mask.any():
                exceed_idx = exceed_mask.idxmax()
                fig.add_vrect(
                    x0=timestamps[exceed_idx],
                    x1=timestamps[-1],
                    fillcolor="yellow",
                    opacity=0.15,
                    line_width=0,
                    annotation_text="AI Drift Zone",
                    annotation_position="top left",
                    annotation=dict(font_size=10)
                )
    
    # Calculate Y axis range
    max_val = df['Vibration (IPS)'].max()
    y_max = max(SCADA_TRIP_THRESHOLD * 1.2, max_val * 1.15)

    # Update layout
    fig.update_layout(
        title={
            'text': 'Vibration Sensor - Cylinder 2',
            'font': {'size': 18, 'color': '#2c3e50', 'family': 'Arial, sans-serif'},
            'x': 0.02,
            'xanchor': 'left'
        },
        xaxis_title='Timestamp',
        yaxis_title='Vibration (IPS)',
        xaxis=dict(
            tickformat='%H:%M\n%b %d, %Y',
            showgrid=True,
            gridcolor='#e0e0e0',
            gridwidth=1,
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#e0e0e0',
            gridwidth=1,
            zeroline=False,
            range=[0, y_max]
        ),
        hovermode='x unified',
        height=450,  # Increased height
        margin=dict(l=80, r=120, t=60, b=80),  # Adjust margins, leave space on right for annotations
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=12, color="#2c3e50")
    )
    
    return fig
# ==========================================
# 2. Core Logic (unchanged parts)
# ==========================================

def load_real_data(file_path="vibration_data_sample.csv"):
    try:
        df = pd.read_csv(file_path)
        target_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        df = df.rename(columns={target_col: "Vibration (IPS)"})
        if len(df) > 500:
            df = df.tail(500).reset_index(drop=True)
        df["Timestamp"] = df.index
        return df
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return None

def get_real_manual_content_from_azure(user_query, azure_openai_client):
    """
    [Real RAG Retrieval]
    """
    try:
        search_client = SearchClient(
            endpoint=st.secrets["SEARCH_ENDPOINT"], 
            index_name=st.secrets["SEARCH_INDEX_NAME"], 
            credential=AzureKeyCredential(st.secrets["SEARCH_KEY"])
        )
        
        embedding_response = azure_openai_client.embeddings.create(
            input=user_query,
            model="text-embedding-ada-002"
        )
        query_vector = embedding_response.data[0].embedding
        
        vector_query = VectorizedQuery(
            vector=query_vector, 
            k_nearest_neighbors=3, 
            fields="text_vector" 
        )
        
        results = search_client.search(  
            search_text=None,  
            vector_queries=[vector_query],
            select=["chunk", "title"]
        )  
        
        retrieved_text = ""
        for result in results:
            source_info = result.get('title', 'Unknown Source')
            text_content = result.get('chunk', '')
            retrieved_text += f"[Source: {source_info}]\n{text_content}\n\n"
            
        return retrieved_text if retrieved_text else "No relevant info found in manuals."

    except Exception as e:
        return f"Search Error: {str(e)} (Using mock data instead)"

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

def run_edge_slm_triage(vibration_val):
    """
    [Edge AI] Use SLM (such as Phi-3 Mini) for edge-based quick triage
    """
    time.sleep(0.5)
    
    if vibration_val > 0.18:
        return {
            "status": "CRITICAL ESCALATION",
            "msg": "⚠️ High-frequency harmonics detected. Immediate cloud analysis required.",
            "should_escalate": True
        }
    elif vibration_val > 0.15:
        return {
            "status": "WARNING",
            "msg": "⚠️ Vibration drift detected. Recommend logging event.",
            "should_escalate": True
        }
    else:
        return {
            "status": "NORMAL",
            "msg": "✅ Minor fluctuation. No action needed.",
            "should_escalate": False
        }

def diagnose_with_azure_openai(client, vibration_data, manual_context):
    """Perform intelligent diagnosis using Azure OpenAI"""
    
    recent_readings = vibration_data.tail(10)['Vibration (IPS)'].tolist()
    max_vibration = vibration_data['Vibration (IPS)'].max()
    avg_vibration = vibration_data['Vibration (IPS)'].mean()
    trend = "increasing" if recent_readings[-1] > recent_readings[0] else "stable/decreasing"
    
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
            model=st.secrets["AZURE_OPENAI_DEPLOYMENT_NAME"],
            messages=[
                {"role": "system", "content": "You are a specialized AI assistant for industrial equipment diagnostics. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        result_text = response.choices[0].message.content.strip()
        
        if result_text.startswith("```json"):
            result_text = result_text.replace("```json", "").replace("```", "").strip()
        elif result_text.startswith("```"):
            result_text = result_text.replace("```", "").strip()
        
        try:
            diagnosis = json.loads(result_text)
            
            if not all(key in diagnosis for key in ['root_cause', 'severity', 'actions', 'downtime_risk']):
                raise ValueError("Missing required fields")
                
            return diagnosis
            
        except (json.JSONDecodeError, ValueError) as e:
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
        st.error(f"AI diagnosis failed: {str(e)}")
        return {
            "root_cause": "System diagnostic error - manual inspection required",
            "severity": "High",
            "actions": ["Contact maintenance team immediately"],
            "downtime_risk": "Unknown"
        }

# ==========================================
# 3. Streamlit UI (using new chart function)
# ==========================================

st.title("🛡️ Enerflex Asset Guardian | Oman - Maradi Huraymah Field")

azure_client = init_azure_openai()

# --- Top layer: Monitoring panel ---
top_col1, top_col2 = st.columns([3, 1])

with top_col1:
    st.subheader("📡 Zone 1: Real-time Monitor (Ariel JGT/4)")
    chart_placeholder = st.empty()

with top_col2:
    st.subheader("📊 Status")
    metric_placeholder = st.empty()
    status_placeholder = st.empty()
    run_btn = st.button("▶️ Start Simulation", type="primary", width='stretch')

# Variable initialization
if 'simulation_df' not in st.session_state:
    st.session_state['simulation_df'] = None

if 'data_finished' not in st.session_state:
    st.session_state['data_finished'] = False
if 'final_val' not in st.session_state:
    st.session_state['final_val'] = 0.0
if 'ai_diagnosis' not in st.session_state:
    st.session_state['ai_diagnosis'] = None

# --- Execute simulation logic ---
if run_btn:
    st.session_state['sap_checked'] = False
    st.session_state['data_finished'] = False
    st.session_state['ai_diagnosis'] = None

    # Generate data
    dummy_df = pd.DataFrame({
        "Timestamp": range(100),
        "bearing_1": np.concatenate([
            np.random.normal(0.06, 0.002, 70), 
            np.linspace(0.06, 0.2, 30) + np.random.normal(0, 0.01, 30) 
        ])
    })
    dummy_df.to_csv("vibration_data_sample.csv", index=False)
    data = load_real_data("vibration_data_sample.csv")

    if data is not None:
        status_placeholder.info("System Running...")
        for i in range(1, len(data)):
            current_df = data.iloc[:i]

            # Use new Plotly chart
            fig = create_vibration_chart(current_df, show_thresholds=True)
            chart_placeholder.plotly_chart(fig, width='stretch')
            
            val = current_df.iloc[-1]["Vibration (IPS)"]

            # Update metrics
            delta_color = "normal" if val < ANOMALY_THRESHOLD else "inverse"
            metric_placeholder.metric(
                "Vibration (IPS)", 
                f"{val:.3f}", 
                delta=f"{val-0.06:.3f}", 
                delta_color=delta_color
            )
            time.sleep(0.06)
        
        st.session_state['data_finished'] = True
        st.session_state['final_val'] = val
        st.session_state['simulation_df'] = data

# --- Bottom layer: Decision control room ---
if st.session_state['simulation_df'] is not None:
    # Draw final static chart
    final_fig = create_vibration_chart(st.session_state['simulation_df'], show_thresholds=True)
    chart_placeholder.plotly_chart(final_fig, width='stretch')

    # Display final metrics
    val = st.session_state['final_val']
    delta_color = "normal" if val < ANOMALY_THRESHOLD else "inverse"
    metric_placeholder.metric("Vibration (IPS)", f"{val:.3f}", delta=f"{val-0.06:.3f}", delta_color=delta_color)
    
    if val > ANOMALY_THRESHOLD:
        status_placeholder.error("⛔ CRITICAL ALERT")
        
        st.divider()
        st.subheader("🧠 Zone 2 & 3: Incident Response Center")

        action_col1, action_col2 = st.columns(2, gap="medium")

        # === Bottom left: AI diagnosis ===
        with action_col1:
            st.subheader("Zone 2: Hybrid AI Diagnosis")

            # --- Layer 1: Edge SLM (Phi-3) ---
            st.markdown("##### 1️⃣ Edge Triage (SLM: Phi-3 Mini)")
            
            last_val = st.session_state['final_val']
            slm_result = run_edge_slm_triage(last_val)
            
            if slm_result['status'] == "CRITICAL ESCALATION":
                st.error(f"**[{slm_result['status']}]** {slm_result['msg']}")
            else:
                st.warning(f"**[{slm_result['status']}]** {slm_result['msg']}")

            # --- Layer 2: Cloud LLM (GPT-4o) ---
            if slm_result['should_escalate']:
                st.markdown("##### 2️⃣ Cloud Expert Analysis (LLM: GPT-4o)")
                
                if st.session_state['ai_diagnosis'] is None and azure_client:
                    with st.status("🚀 SLM triggered Cloud Agent. Analyzing with Azure OpenAI...", expanded=True) as status:
                        search_query = "high vibration suction valve failure symptoms"
                        manual_text = get_real_manual_content_from_azure(search_query, azure_client)
                        
                        st.session_state['retrieved_context'] = manual_text
                        
                        status.write("Generating diagnosis...")
                        diagnosis = diagnose_with_azure_openai(
                            azure_client, 
                            st.session_state['simulation_df'], 
                            manual_text
                        )
                        st.session_state['ai_diagnosis'] = diagnosis
                        status.update(label="Deep Analysis Complete ✨", state="complete", expanded=False)
                
                if st.session_state['ai_diagnosis']:
                    diag = st.session_state['ai_diagnosis']
                    
                    severity_colors = {"Low": "🟢", "Medium": "🟡", "High": "🟠", "Critical": "🔴"}
                    severity_icon = severity_colors.get(diag.get('severity', 'High'), "🔴")
                    st.caption(f"{severity_icon} **Severity:** {diag.get('severity', 'High')}")
                    
                    st.success(f"**Root Cause:** {diag.get('root_cause', 'Analysis in progress')}")
                    
                    if 'actions' in diag and isinstance(diag['actions'], list):
                        st.markdown("**Recommended Actions:**")
                        for idx, action in enumerate(diag['actions'], 1):
                            st.markdown(f"{idx}. {action}")
                    
                    if 'downtime_risk' in diag:
                        st.error(f"⚠️ **Downtime Risk:** {diag['downtime_risk']}")

                    with st.expander("📄 RAG: Retrieved Manual Context (Azure AI Search)", expanded=False):
                        context_to_show = st.session_state.get('retrieved_context', "No content retrieved")
                        st.code(context_to_show, language="text")
            
            else:
                st.info("SLM determined no cloud analysis needed. Saving costs. 💰")

        # === Bottom right: SAP execution ===
        with action_col2:
            st.warning("🏢 **Step 2: SAP Execution (ERP Bridge)**")
            
            if 'sap_checked' not in st.session_state:
                st.session_state['sap_checked'] = False

            if st.button("🔍 Check SAP Inventory (MM Module)", width='stretch'):
                st.session_state['sap_checked'] = True
            
            if st.session_state['sap_checked']:
                sap_data = call_mock_sap_api("B-1234-VLV")
                
                res_c1, res_c2 = st.columns([1, 1])
                with res_c1:
                    with st.expander("View API JSON", expanded=False):
                        st.json(sap_data)
                with res_c2:
                    if sap_data['data']['qty'] > 0:
                        st.success(f"✅ Stock: {sap_data['data']['qty']} EA")
                    else:
                        st.error("Out of Stock")

                st.markdown("**👷 Engineer Approval Human-in-the-Loop Action**")
                engineer_notes = st.text_area("Field Notes", "Confirmed valve issue. Proceed.", height=80)
                
                if st.button("🚀 Approve & Create Work Order (PM Module)", type="primary", width='stretch'):
                    st.toast("Connecting to SAP S/4HANA...", icon="⏳")
                    time.sleep(1)
                    st.balloons()
                    st.success(f"✅ PM Order Created! [Ref: {int(time.time())}]")
                    st.caption(f"Logged Notes: {engineer_notes}")

    else:
        status_placeholder.success("✅ Normal Operation")
        st.success("Equipment is running within optimal parameters.")