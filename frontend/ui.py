import streamlit as st
import requests
import os
import pandas as pd
from datetime import datetime
import json

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="TikTok Geo-Compliance System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load external CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Try to load external CSS, fallback to inline if not found
try:
    load_css('frontend/static/styles.css')
except FileNotFoundError:
    st.warning("⚠️ External CSS file not found. Using inline styles.")
    # Add minimal inline CSS for basic functionality
    st.markdown("""
    <style>
    :root {
        --primary-color: #ff0050;
        --secondary-color: #25f4ee;
        --dark-bg: #161823;
    }
    .main-header {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(255, 0, 80, 0.3);
    }
    .main-header h1, .main-header p {
        color: white;
        margin: 0;
    }
    </style>
    """, unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8000/api/v1/compliance"

def check_compliance(feature_title: str, feature_description: str):
    payload = {
        "title": feature_title,
        "description": feature_description
    }
    try:
        response = requests.post(f"{API_URL}/check", json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def upload_regulation_file(uploaded_file):
    try:
        api_base_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        response = requests.post(f"{api_base_url}/upload_regulation", files=files, timeout=60)
        return response
    except Exception as e:
        return None

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# --- Enhanced Sidebar Navigation ---
with st.sidebar:
    # Sidebar header with logo
    st.markdown("""
        <div class="sidebar-logo">
            <h2>⚖️ TikTok</h2>
            <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0; font-size: 0.9rem;">Geo-Compliance System</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Navigation menu with enhanced styling
    st.markdown("### 🧭 Navigation")
    
    page = st.radio(
        "Select Page:",
        ["🔍 Compliance Checker", "📤 Upload Regulations", "📊 Analytics Dashboard"],
        label_visibility="collapsed"
    )
    
    # Quick stats in sidebar
    st.markdown("---")
    st.markdown("### 📈 Quick Stats")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            <div class="stats-card">
                <p class="stats-number">{}</p>
                <p class="stats-label">Analyses</p>
            </div>
        """.format(len(st.session_state.analysis_history)), unsafe_allow_html=True)
    
    with col2:
        compliant_count = sum(1 for analysis in st.session_state.analysis_history 
                            if analysis.get('flag', '').lower() == 'yes')
        st.markdown("""
            <div class="stats-card">
                <p class="stats-number">{}</p>
                <p class="stats-label">Flagged</p>
            </div>
        """.format(compliant_count), unsafe_allow_html=True)
    
    # Help section
    st.markdown("---")
    st.markdown("### ❓ Need Help?")
    with st.expander("📘 How to Use"):
        st.markdown("""
        **Compliance Checker:**
        1. Enter feature title and description
        2. Click 'Analyze Compliance'
        3. Review automated analysis results
        
        **Upload Regulations:**
        1. Select PDF regulation files
        2. Upload to knowledge base
        3. System will auto-index content
        
        **Analytics:**
        - View historical analysis trends
        - Export compliance reports
        - Monitor system performance
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: rgba(255,255,255,0.6); font-size: 0.8rem;">
            <p>🚀 TikTok TechJam 2025</p>
            <p>Built with ❤️ by Team 4</p>
        </div>
    """, unsafe_allow_html=True)

# ================================================
# Page 1: Enhanced Compliance Checker
# ================================================
if page == "🔍 Compliance Checker":
    # Main header
    st.markdown("""
        <div class="main-header">
            <h1>⚖️ Geo-Regulation Compliance System</h1>
            <p>Automated flagging of features requiring location-specific compliance logic with audit-ready reasoning and regulation mapping</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Input section
    st.markdown("### 📝 Feature Analysis Input")
    
    title = st.text_input(
        "Feature Name*", 
        placeholder="e.g. Curfew login blocker with ASL and GH for Utah minors",
        help="Enter a descriptive title for the feature"
    )
    
    description = st.text_area(
        "Feature Description*", 
        height=120,
        placeholder="e.g. To comply with the Utah Social Media Regulation Act, we are implementing a curfew-based login restriction for users under 18...",
        help="Provide detailed description including functionality, data usage, and geographic considerations"
    )
    
    # Analysis button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("🚀 Analyze Compliance", use_container_width=True, type="primary")
    
    # Results section
    if analyze_button:
        if not title or not description:
            st.markdown("""
                <div class="warning-badge">
                    ⚠️ Please fill in both Title and Description before analyzing
                </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("🔍 Analyzing feature compliance with LLM agents..."):
                try:
                    payload = {"title": title, "description": description}
                    api_base_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
                    
                    # Simulated response for demo purposes
                    import time
                    time.sleep(2)  # Simulate processing time
                    
                    # Mock response - replace with actual API call
                    data = {
                        "flag": "yes",
                        "reasoning": "This feature requires geo-specific compliance logic because it reads user location data to enforce region-specific copyright restrictions. The blocking of downloads based on geographic location directly relates to France's specific copyright enforcement requirements.",
                        "related_regulations": [
                            "EU Digital Service Act (DSA) - Article 14",
                            "France Copyright Law - Article L. 331-5",
                            "GDPR - Location data processing requirements"
                        ],
                        "confidence_score": 0.92,
                        "risk_level": "High"
                    }
                    
                    # Store in history
                    analysis_result = {
                        "timestamp": datetime.now().isoformat(),
                        "title": title,
                        "description": description[:100] + "..." if len(description) > 100 else description,
                        "flag": data.get("flag", "unknown"),
                        "confidence": data.get("confidence_score", 0.0),
                        "risk_level": data.get("risk_level", "Unknown")
                    }
                    st.session_state.analysis_history.append(analysis_result)
                    
                    # Results display
                    st.markdown("---")
                    st.markdown("## 📊 Analysis Results")
                    
                    # Agent processing status
                    st.markdown("### 🤖 Multi-Agent System Status")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown('<span class="agent-status agent-complete">🔍 Screening Agent: Complete</span>', unsafe_allow_html=True)
                    with col2:
                        st.markdown('<span class="agent-status agent-complete">📚 Research Agent: Complete</span>', unsafe_allow_html=True)
                    with col3:
                        st.markdown('<span class="agent-status agent-complete">✅ Validation Agent: Complete</span>', unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Main result
                    flag = data.get("flag", "Unknown")
                    if flag.lower() == "yes":
                        st.markdown("""
                            <div class="success-badge">
                                ✅ Feature REQUIRES geo-specific compliance logic
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <div class="info-badge">
                                ℹ️ Feature does NOT require geo-specific compliance logic
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Detailed results in columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Confidence and Risk
                        confidence = data.get("confidence_score", 0.0)
                        risk_level = data.get("risk_level", "Unknown")
                        
                        st.markdown("#### 🎯 Confidence Score")
                        st.progress(confidence)
                        st.markdown(f"**{confidence:.1%}** confidence in analysis")
                        
                        st.markdown("#### ⚠️ Risk Level")
                        risk_color = "🔴" if risk_level == "High" else "🟡" if risk_level == "Medium" else "🟢"
                        st.markdown(f"{risk_color} **{risk_level}** Risk")
                    
                    with col2:
                        # Reasoning
                        st.markdown("#### 📄 Analysis Reasoning")
                        st.info(data.get("reasoning", "No reasoning provided"))
                    
                    # Related Regulations
                    related_regs = data.get("related_regulations", [])
                    if related_regs:
                        st.markdown("#### 📚 Related Regulations")
                        for i, reg in enumerate(related_regs, 1):
                            st.markdown(f"**{i}.** {reg}")
                    
                    # Feedback Section - Key Addition for User Workflow
                    st.markdown("---")
                    st.markdown("### 🎯 Help Improve Our Multi-Agent System")
                    st.markdown("Your feedback trains our agents to provide better compliance analysis:")
                    
                    # Initialize feedback state
                    if 'feedback_submitted' not in st.session_state:
                        st.session_state.feedback_submitted = False
                        st.session_state.feedback_type = None
                    
                    if not st.session_state.feedback_submitted:
                        feedback_col1, feedback_col2, feedback_col3 = st.columns(3)
                        
                        with feedback_col1:
                            if st.button("✅ Accurate Analysis", key="feedback_accurate", use_container_width=True, type="primary"):
                                st.session_state.feedback_submitted = True
                                st.session_state.feedback_type = "accurate"
                                st.rerun()
                        
                        with feedback_col2:
                            if st.button("❌ Incorrect Result", key="feedback_incorrect", use_container_width=True):
                                st.session_state.feedback_submitted = True
                                st.session_state.feedback_type = "incorrect"
                                st.rerun()
                        
                        with feedback_col3:
                            if st.button("🤔 Needs More Context", key="feedback_context", use_container_width=True):
                                st.session_state.feedback_submitted = True
                                st.session_state.feedback_type = "context"
                                st.rerun()
                    
                    else:
                        # Show feedback result and collect additional input
                        if st.session_state.feedback_type == "accurate":
                            st.markdown("""
                                <div class="feedback-card">
                                    <h4>✅ Thanks for the positive feedback!</h4>
                                    <p>Your confirmation helps our agents learn successful analysis patterns.</p>
                                </div>
                            """, unsafe_allow_html=True)
                            
                        elif st.session_state.feedback_type == "incorrect":
                            st.markdown("""
                                <div class="feedback-card">
                                    <h4>❌ Thanks for catching that!</h4>
                                    <p>Help us understand what went wrong:</p>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            correction_feedback = st.text_area(
                                "What should the correct analysis be?",
                                placeholder="e.g., This feature should NOT require geo-compliance because...",
                                key="correction_input"
                            )
                            
                            if st.button("📝 Submit Correction", key="submit_correction"):
                                st.success("🚀 Correction submitted! Our agents will learn from this.")
                                
                        elif st.session_state.feedback_type == "context":
                            st.markdown("""
                                <div class="feedback-card">
                                    <h4>🤔 Tell us more!</h4>
                                    <p>What additional context would improve this analysis?</p>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            context_feedback = st.text_area(
                                "What context is missing?",
                                placeholder="e.g., Need to consider specific regional variations, industry context, etc.",
                                key="context_input"
                            )
                            
                            if st.button("📝 Submit Context", key="submit_context"):
                                st.success("🚀 Context submitted! This will help improve future analyses.")
                        
                        # Reset feedback option
                        if st.button("🔄 Provide Different Feedback", key="reset_feedback"):
                            st.session_state.feedback_submitted = False
                            st.session_state.feedback_type = None
                            st.rerun()
                    
                    with col1:
                        # CSV export
                        csv_data = f"Title,Description,Flag,Confidence,Risk Level,Reasoning\n"
                        csv_data += f'"{title}","{description}","{flag}",{confidence},"{risk_level}","{data.get("reasoning", "")}"'
                        
                        st.download_button(
                            label="📊 Download CSV Report",
                            data=csv_data,
                            file_name=f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        # JSON export
                        json_data = json.dumps(data, indent=2)
                        st.download_button(
                            label="📋 Download JSON Report",
                            data=json_data,
                            file_name=f"compliance_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    
                    with col3:
                        if st.button("🔄 Analyze Another Feature", use_container_width=True):
                            st.rerun()
                    
                except Exception as e:
                    st.markdown(f"""
                        <div class="warning-badge">
                            ❌ Error connecting to backend: {str(e)}
                        </div>
                    """, unsafe_allow_html=True)

# ================================================
# Page 2: Enhanced Upload Regulations
# ================================================
elif page == "📤 Upload Regulations":
    st.markdown("""
        <div class="main-header">
            <h1>📤 Upload Regulations</h1>
            <p>Upload PDF regulation documents to enhance the compliance knowledge base</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Upload interface
    st.markdown("### 📁 Upload Regulation Documents")
    
    # File uploader with enhanced styling
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload regulation PDF documents to be indexed and added to the compliance knowledge base"
    )
    
    if uploaded_files:
        st.markdown("### 📋 Upload Status")
        
        for i, uploaded_file in enumerate(uploaded_files):
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"📄 **{uploaded_file.name}** ({uploaded_file.size:,} bytes)")
            
            with col2:
                if st.button(f"Upload", key=f"upload_{i}", use_container_width=True):
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        response = upload_regulation_file(uploaded_file)
                        
                        if response and response.status_code == 200:
                            st.success("✅ Uploaded!")
                        else:
                            st.error("❌ Failed")
            
            with col3:
                st.write("⏳ Pending")
    
    # Supported regulations info
    st.markdown("---")
    st.markdown("### 📚 Currently Supported Regulations")
    
    regulations = [
        {"name": "EU Digital Service Act (DSA)", "status": "✅ Active", "coverage": "EU"},
        {"name": "California - Protecting Our Kids from Social Media Addiction Act", "status": "✅ Active", "coverage": "CA, US"},
        {"name": "Florida - Online Protections for Minors", "status": "✅ Active", "coverage": "FL, US"},
        {"name": "Utah Social Media Regulation Act", "status": "✅ Active", "coverage": "UT, US"},
        {"name": "US NCMEC Reporting Requirements", "status": "✅ Active", "coverage": "US"},
    ]
    
    for reg in regulations:
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.write(f"📋 **{reg['name']}**")
        with col2:
            st.write(reg['status'])
        with col3:
            st.write(reg['coverage'])

# ================================================
# Page 3: Analytics Dashboard
# ================================================
elif page == "📊 Analytics Dashboard":
    st.markdown("""
        <div class="main-header">
            <h1>📊 Analytics Dashboard</h1>
            <p>Monitor compliance analysis trends and system performance</p>
        </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.analysis_history:
        st.markdown("""
            <div class="feature-card" style="text-align: center;">
                <h3>📈 No Analysis Data Yet</h3>
                <p>Start analyzing features to see analytics and trends here.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Summary stats
        total_analyses = len(st.session_state.analysis_history)
        flagged_count = sum(1 for analysis in st.session_state.analysis_history 
                           if analysis.get('flag', '').lower() == 'yes')
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class="stats-card">
                    <p class="stats-number">{total_analyses}</p>
                    <p class="stats-label">Total Analyses</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="stats-card">
                    <p class="stats-number">{flagged_count}</p>
                    <p class="stats-label">Flagged Features</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            flagged_rate = (flagged_count / total_analyses * 100) if total_analyses > 0 else 0
            st.markdown(f"""
                <div class="stats-card">
                    <p class="stats-number">{flagged_rate:.1f}%</p>
                    <p class="stats-label">Flagged Rate</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_confidence = sum(analysis.get('confidence', 0) for analysis in st.session_state.analysis_history) / total_analyses
            st.markdown(f"""
                <div class="stats-card">
                    <p class="stats-number">{avg_confidence:.1%}</p>
                    <p class="stats-label">Avg Confidence</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Analysis history table
        st.markdown("### 📋 Recent Analysis History")
        
        if st.session_state.analysis_history:
            df = pd.DataFrame(st.session_state.analysis_history)
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            
            # Display as formatted table
            st.dataframe(
                df[['timestamp', 'title', 'description', 'flag', 'confidence', 'risk_level']],
                use_container_width=True
            )
            
            # Export all history
            if st.button("📊 Export All Analysis Data", use_container_width=False):
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="💾 Download Complete History (CSV)",
                    data=csv_data,
                    file_name=f"compliance_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

# --- Footer ---
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: rgba(255,255,255,0.6); padding: 2rem;">
        <p><strong>TikTok TechJam 2025</strong> | Geo-Regulation Compliance System</p>
        <p>🚀 Built with Streamlit • 🤖 Powered by LLM Agents • ⚖️ Ensuring Global Compliance</p>
        <p style="font-size: 0.8rem;">Mobile-Optimized ✅ | iOS & Android Compatible 📱</p>
    </div>
""", unsafe_allow_html=True)

# uvicorn 
# streamlit run frontend/ui.py