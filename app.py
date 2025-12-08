import streamlit as st
import pandas as pd
import time
from engine import DataIngestor, StorageManager, SearchIndexer
from agent import TransformationAgent
import plotly.express as px
import os

# --- CONFIG ---
st.set_page_config(page_title="Agentic Data Foundry", layout="wide", page_icon="⚡")
os.environ["OPENAI_API_KEY"] = st.secrets.get("OPENAI_API_KEY", "") # Or input in UI

# --- STATE MANAGEMENT ---
if 'data_state' not in st.session_state:
    st.session_state['data_state'] = None
if 'current_step' not in st.session_state:
    st.session_state['current_step'] = 0
if 'logs' not in st.session_state:
    st.session_state['logs'] = []

def log(message):
    st.session_state['logs'].append(f"[{time.strftime('%H:%M:%S')}] {message}")

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚡ Neural Spark Agent")
    st.markdown("---")
    st.write("## ⚙️ Pipeline Status")
    
    steps = ["1. Ingestion", "2. Storage (Delta)", "3. Indexing", "4. Staging & Wrangles", "5. Transformation", "6. Serving"]
    current = st.session_state['current_step']
    
    for i, step in enumerate(steps):
        if i < current:
            st.success(f"✅ {step}")
        elif i == current:
            st.info(f"🔄 {step}")
        else:
            st.write(f"⬜ {step}")
            
    st.markdown("---")
    st.write("## 📝 System Logs")
    for msg in st.session_state['logs'][-10:]:
        st.text_area("Log", msg, height=2, label_visibility="collapsed", disabled=True)

# --- MAIN APP ---

# 1. INGESTION
if st.session_state['current_step'] == 0:
    st.header("Step 1: Data Ingestion & Capture")
    st.info("Upload PDF, CSV, Excel, or Connect to SQL/API")
    
    uploaded_file = st.file_uploader("Drop your data here", type=['csv', 'xlsx', 'pdf', 'json'])
    
    if uploaded_file:
        file_type = uploaded_file.name.split('.')[-1]
        if st.button("🚀 Ingest Data"):
            with st.spinner("Agent analyzing file structure..."):
                ingestor = DataIngestor()
                df = ingestor.read_file(uploaded_file, file_type)
                st.session_state['data_state'] = df
                st.session_state['dataset_name'] = uploaded_file.name.split('.')[0]
                log(f"Ingested {len(df)} rows from {uploaded_file.name}")
                st.session_state['current_step'] = 1
                st.rerun()

# 2. STORAGE
elif st.session_state['current_step'] == 1:
    st.header("Step 2: Storage (Parquet/Delta Layer)")
    df = st.session_state['data_state']
    
    st.dataframe(df.head(), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
        
    if st.button("💾 Commit to Data Lake"):
        with st.spinner("Writing to Parquet store..."):
            manager = StorageManager()
            path = manager.save_to_bronze(df, st.session_state['dataset_name'])
            log(f"Data persisted to {path}")
            st.session_state['current_step'] = 2
            st.rerun()

# 3. INDEXING
elif st.session_state['current_step'] == 2:
    st.header("Step 3: Indexing & Semantic Optimization")
    df = st.session_state['data_state']
    
    st.warning("Select a column to vectorize for advanced research queries.")
    index_col = st.selectbox("Key Column", df.columns)
    
    if st.button("🔍 Build Vector Index"):
        with st.spinner("Embedding vectors (ChromaDB)..."):
            indexer = SearchIndexer()
            # Ensure data is string for indexing
            indexer.index_data(df, "demo_collection", index_col)
            log(f"Vector index built on column: {index_col}")
            st.session_state['index_col'] = index_col
            st.session_state['current_step'] = 3
            st.rerun()

# 4. STAGING & SEARCH
elif st.session_state['current_step'] == 3:
    st.header("Step 4: Staging & Advanced Research")
    
    tab1, tab2 = st.tabs(["📊 Data Staging Hub", "🔎 Semantic Search"])
    
    with tab1:
        st.dataframe(st.session_state['data_state'], use_container_width=True)
        if st.button("Proceed to Transformation ➡️"):
            st.session_state['current_step'] = 4
            st.rerun()
            
    with tab2:
        query = st.text_input("Ask a question or search by concept (e.g., 'high value transactions')")
        if query:
            indexer = SearchIndexer()
            results = indexer.search(query, "demo_collection")
            
            st.write("### Research Results")
            # Display results in a readable way
            for i, doc in enumerate(results['documents'][0]):
                st.info(f"Result {i+1}: {doc}")

# 5. TRANSFORMATION
elif st.session_state['current_step'] == 4:
    st.header("Step 5: Agentic Transformation (Business Rules)")
    
    agent = TransformationAgent(api_key=os.environ.get("OPENAI_API_KEY"))
    rules_dict = agent.get_rule_dictionary()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Rule Dictionary")
        selected_rule = st.radio("Select Pre-defined Rule", ["Custom"] + list(rules_dict.keys()))
    
    with col2:
        st.subheader("Rule Definition")
        if selected_rule == "Custom":
            custom_rule = st.text_area("Describe logic (Natural Language)", "Filter rows where Sales > 5000 and Region is 'North'")
            rule_input = custom_rule
        else:
            st.write(f"**Logic:** {rules_dict[selected_rule]}")
            rule_input = rules_dict[selected_rule]
            
        if st.button("⚡ Execute Agent Job"):
            with st.spinner("Agent generating transformation code..."):
                # Pass data_state to agent
                df = st.session_state['data_state']
                # Mocking the AI agent application for visual stability if no key
                # In real prod, this calls OpenAI to generate Pandas/SQL
                new_df = agent.apply_business_rule(df, rule_input, selected_rule)
                
                st.session_state['data_state'] = new_df
                log(f"Applied rule: {selected_rule}")
                st.success("Transformation Complete!")
    
    st.write("### Preview Result")
    st.dataframe(st.session_state['data_state'].head(10), use_container_width=True)
    
    if st.button("Finalize & Load ➡️"):
        st.session_state['current_step'] = 5
        st.rerun()

# 6. SERVING / API
elif st.session_state['current_step'] == 5:
    st.header("Step 6: Data Serving & API Endpoint")
    
    st.write("The data is now optimized and ready for consumption via API.")
    
    tab1, tab2, tab3 = st.tabs(["JSON API View", "Visual Analytics", "Export"])
    
    with tab1:
        st.caption("GET /api/v1/data/latest")
        st.json(st.session_state['data_state'].head(5).to_dict(orient='records'))
        
    with tab2:
        df = st.session_state['data_state']
        numeric_cols = df.select_dtypes(include=['float', 'int']).columns
        if len(numeric_cols) > 0:
            x_axis = st.selectbox("X Axis", df.columns)
            y_axis = st.selectbox("Y Axis", numeric_cols)
            fig = px.bar(df, x=x_axis, y=y_axis, title="Data Visualization")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numeric data for visualization.")
            
    with tab3:
        csv = st.session_state['data_state'].to_csv(index=False).encode('utf-8')
        st.download_button("Download Processed CSV", csv, "processed_data.csv", "text/csv")
    
    if st.button("🔄 Start New Pipeline"):
        st.session_state['current_step'] = 0
        st.session_state['data_state'] = None
        st.rerun()
