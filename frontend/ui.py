import streamlit as st
import requests
import os
import pandas as pd
from datetime import datetime
import json

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="TikTok Geo-Compliance System",
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
    st.warning("External CSS file not found. Using inline styles.")
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
    .feedback-section {
        background: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 1rem 0;
    }
    .required-feedback {
        background: rgba(255, 0, 80, 0.1);
        border: 1px solid rgba(255, 0, 80, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
API_URL = f"{API_BASE_URL}/api/v1/compliance"

def check_compliance(feature_title: str, feature_description: str, feature_document=None):
    """Enhanced compliance check with optional document upload"""
    try:
        # Prepare the multipart form data
        files = {}
        data = {
            "title": feature_title,
            "description": feature_description
        }
        
        # Add document if provided
        if feature_document is not None:
            files["document"] = (feature_document.name, feature_document.getvalue(), feature_document.type)
        
        # Make request with files and form data
        response = requests.post(
            f"{API_URL}/check", 
            data=data, 
            files=files if files else None,
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def submit_feedback(analysis_id: str, feedback_type: str, feedback_text: str = None, correction_data: dict = None):
    """Submit feedback for a compliance analysis"""
    try:
        payload = {
            "analysis_id": analysis_id,
            "feedback_type": feedback_type,  # 'positive', 'negative', 'needs_context'
            "feedback_text": feedback_text,
            "correction_data": correction_data,
            "timestamp": datetime.now().isoformat()
        }
        
        response = requests.post(f"{API_URL}/feedback", json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def upload_regulation_files_batch(files_with_metadata):
    """Batch upload function using the enhanced PDF processing pipeline with metadata."""
    try:
        files = []
        metadata = {}
        
        for i, item in enumerate(files_with_metadata):
            uploaded_file = item['file']
            files.append(("files", (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")))
            
            # Store metadata for each file using filename as key
            metadata[uploaded_file.name] = {
                'regulation_name': item['regulation_name'],
                'geo_jurisdiction': item['geo_jurisdiction']
            }
        
        # Send files and metadata separately
        data = {'metadata': json.dumps(metadata)}
        response = requests.post(
            f"{API_BASE_URL}/api/v1/upload-pdfs", 
            files=files, 
            data=data,
            timeout=120
        )
        return response
    except Exception as e:
        return None

def get_upload_stats():
    """Get statistics about the PDF upload pipeline."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/upload-stats", timeout=60)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        return None

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'current_analysis_id' not in st.session_state:
    st.session_state.current_analysis_id = None
if 'feedback_required' not in st.session_state:
    st.session_state.feedback_required = False

# --- Enhanced Sidebar Navigation ---
with st.sidebar:
    # Sidebar header with logo
    st.markdown("""
        <div class="sidebar-logo">
            <h2>TikTok</h2>
            <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0; font-size: 0.9rem;">Geo-Compliance System</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Navigation menu with enhanced styling
    st.markdown("### Navigation")
    
    page = st.radio(
        "Select Page:",
        ["Compliance Checker", "Upload Regulations", "Analytics Dashboard"],
        label_visibility="collapsed"
    )
    
    # Quick stats in sidebar
    st.markdown("---")
    st.markdown("### Quick Stats")
    
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
    st.markdown("### Need Help?")
    with st.expander("How to Use"):
        st.markdown("""
        **Compliance Checker:**
        1. Enter feature name and description
        2. Optionally upload feature documentation
        3. Click 'Analyze Compliance'
        4. Provide feedback to improve system
        
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
            <p>Built with ❤️ by Zero Shot at Winning</p>
        </div>
    """, unsafe_allow_html=True)

# ================================================
# Page 1: Enhanced Compliance Checker
# ================================================
if page == "Compliance Checker":
    # Main header
    st.markdown("""
        <div class="main-header">
            <h1>⚖️ Geo-Regulation Compliance System</h1>
            <p>Automated flagging of features requiring location-specific compliance logic with audit-ready reasoning and regulation mapping</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Input section
    st.markdown("### Feature Analysis Input")
    
    # Enhanced input form with document upload
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
    st.markdown("#### Optional Feature Documentation")
    feature_document = st.file_uploader(
        "Upload Feature Document",
        type=["pdf", "txt", "docx", "md"],
        help="Upload any technical specifications, design documents, or additional context about this feature"
    )
    
    if feature_document:
        st.success(f"{feature_document.name} uploaded ({feature_document.size:,} bytes)")
    
    # Analysis button
    analyze_button = st.button("🚀 Analyze Compliance", use_container_width=True, type="primary")
    
    # Results section
    if analyze_button:
        if not title or not description:
            st.markdown("""
                <div class="warning-badge">
                    ⚠️ Please fill in both Feature Name and Description before analyzing
                </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("🔍 Analyzing feature compliance with LLM agents..."):
                try:
                    # Call the enhanced compliance check with optional document
                    result = check_compliance(title, description, feature_document)
                    
                    if "error" in result:
                        st.error(f"Error: {result}")
                    else:
                        # Generate analysis ID for feedback tracking
                        analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(st.session_state.analysis_history)}"
                        st.session_state.current_analysis_id = analysis_id
                        
                        # Store in history with analysis ID
                        analysis_result = {
                            "analysis_id": analysis_id,
                            "timestamp": datetime.now().isoformat(),
                            "title": title,
                            "description": description[:100] + "..." if len(description) > 100 else description,
                            "has_document": feature_document is not None,
                            "document_name": feature_document.name if feature_document else None,
                            "flag": result.get("flag", "unknown"),
                            "confidence": result.get("confidence_score", 0.0),
                            "risk_level": result.get("risk_level", "Unknown"),
                            "full_result": result
                        }
                        st.session_state.analysis_history.append(analysis_result)
                        
                        # Results display
                        st.markdown("---")
                        st.markdown("## Analysis Results")
                        
                        # Agent processing status
                        st.markdown("### Multi-Agent System Status")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown('<span class="agent-status agent-complete">Screening Agent: Complete</span>', unsafe_allow_html=True)
                        with col2:
                            st.markdown('<span class="agent-status agent-complete">Research Agent: Complete</span>', unsafe_allow_html=True)
                        with col3:
                            st.markdown('<span class="agent-status agent-complete">Validation Agent: Complete</span>', unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        # Main result
                        flag = result.get("flag", "Unknown")
                        if flag.lower() == "yes":
                            st.markdown("""
                                <div class="success-badge">
                                    Feature REQUIRES geo-specific compliance logic
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                                <div class="info-badge">
                                    Feature does NOT require geo-specific compliance logic
                                </div>
                            """, unsafe_allow_html=True)
                        
                        # Document processing status
                        if feature_document:
                            st.markdown(f"**Document Processed:** {feature_document.name} included in analysis")
                        
                        # Detailed results in columns
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Confidence and Risk
                            confidence = result.get("confidence_score", 0.0)
                            risk_level = result.get("risk_level", "Unknown")
                            
                            st.markdown("#### Confidence Score")
                            st.progress(confidence)
                            st.markdown(f"**{confidence:.1%}** confidence in analysis")
                            
                            st.markdown("#### Risk Level")
                            risk_color = "🔴" if risk_level == "High" else "🟡" if risk_level == "Medium" else "🟢"
                            st.markdown(f"{risk_color} **{risk_level}** Risk")
                        
                        with col2:
                            # Reasoning
                            st.markdown("#### Analysis Reasoning")
                            st.info(result.get("reasoning", "No reasoning provided"))
                        
                        # Related Regulations
                        related_regs = result.get("related_regulations", [])
                        if related_regs:
                            st.markdown("#### Related Regulations")
                            for i, reg in enumerate(related_regs, 1):
                                st.markdown(f"**{i}.** {reg}")
                        
                        # Enhanced Feedback Section
                        st.markdown("---")
                        st.markdown("### Help Improve Our Multi-Agent System")
                        
                        # Initialize feedback state for this analysis
                        feedback_key = f"feedback_{analysis_id}"
                        if feedback_key not in st.session_state:
                            st.session_state[feedback_key] = {
                                'submitted': False,
                                'type': None,
                                'requires_text': False
                            }
                        
                        feedback_state = st.session_state[feedback_key]
                        
                        if not feedback_state['submitted']:
                            st.markdown("Your feedback trains our agents to provide better compliance analysis:")
                            
                            # Feedback buttons
                            feedback_col1, feedback_col2, feedback_col3 = st.columns(3)
                            
                            with feedback_col1:
                                if st.button("✅ Accurate Analysis", key=f"feedback_positive_{analysis_id}", use_container_width=True, type="primary"):
                                    # Positive feedback - just submit
                                    feedback_result = submit_feedback(analysis_id, "positive")
                                    if "error" not in feedback_result:
                                        feedback_state['submitted'] = True
                                        feedback_state['type'] = 'positive'
                                        st.rerun()
                                    else:
                                        st.error(f"Failed to submit feedback: {feedback_result['error']}")
                            
                            with feedback_col2:
                                if st.button("❌ Incorrect Result", key=f"feedback_negative_{analysis_id}", use_container_width=True):
                                    feedback_state['requires_text'] = True
                                    feedback_state['type'] = 'negative'
                                    st.rerun()
                            
                            with feedback_col3:
                                if st.button("🤔 Needs More Context", key=f"feedback_context_{analysis_id}", use_container_width=True):
                                    feedback_state['requires_text'] = True
                                    feedback_state['type'] = 'needs_context'
                                    st.rerun()
                            
                            # Handle negative feedback or context feedback (requires text input)
                            if feedback_state['requires_text'] and feedback_state['type']:
                                st.markdown("---")
                                
                                if feedback_state['type'] == 'negative':
                                    st.markdown("""
                                        <div class="feedback-section required-feedback">
                                            <h4>Help Us Understand What Went Wrong</h4>
                                            <p><strong>This feedback is required</strong> to help our agents learn from mistakes.</p>
                                        </div>
                                    """, unsafe_allow_html=True)
                                    
                                    correction_text = st.text_area(
                                        "What should the correct analysis be?*",
                                        placeholder="e.g., This feature should NOT require geo-compliance because it doesn't process location data or implement region-specific restrictions...",
                                        help="Please provide detailed feedback about what the correct analysis should be",
                                        key=f"correction_{analysis_id}",
                                        height=100
                                    )
                                    
                                    # Correction details
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        correct_flag = st.selectbox(
                                            "Correct Flag Should Be:",
                                            ["Yes - Requires geo-compliance", "No - Does not require geo-compliance"],
                                            key=f"correct_flag_{analysis_id}"
                                        )
                                    
                                    with col2:
                                        correct_risk = st.selectbox(
                                            "Correct Risk Level:",
                                            ["Low", "Medium", "High"],
                                            key=f"correct_risk_{analysis_id}"
                                        )
                                    
                                    if st.button(f"Submit Correction (Required)", key=f"submit_correction_{analysis_id}", 
                                               disabled=not correction_text.strip(), type="primary"):
                                        correction_data = {
                                            "correct_flag": "yes" if "Yes" in correct_flag else "no",
                                            "correct_risk_level": correct_risk,
                                            "original_result": result
                                        }
                                        
                                        feedback_result = submit_feedback(
                                            analysis_id, 
                                            "negative", 
                                            correction_text, 
                                            correction_data
                                        )
                                        
                                        if "error" not in feedback_result:
                                            feedback_state['submitted'] = True
                                            st.success("🚀 Correction submitted! Our agents will learn from this.")
                                            st.rerun()
                                        else:
                                            st.error(f"Failed to submit feedback: {feedback_result['error']}")
                                
                                elif feedback_state['type'] == 'needs_context':
                                    st.markdown("""
                                        <div class="feedback-section required-feedback">
                                            <h4>Help Us Improve Context Understanding</h4>
                                            <p><strong>This feedback is required</strong> to help our agents provide better analysis.</p>
                                        </div>
                                    """, unsafe_allow_html=True)
                                    
                                    context_text = st.text_area(
                                        "What additional context would improve this analysis?*",
                                        placeholder="e.g., Need to consider specific industry regulations, regional variations, technical implementation details, etc.",
                                        help="Please provide specific context that would help improve future analyses",
                                        key=f"context_{analysis_id}",
                                        height=100
                                    )
                                    
                                    if st.button(f"📝 Submit Context (Required)", key=f"submit_context_{analysis_id}", 
                                               disabled=not context_text.strip(), type="primary"):
                                        feedback_result = submit_feedback(analysis_id, "needs_context", context_text)
                                        
                                        if "error" not in feedback_result:
                                            feedback_state['submitted'] = True
                                            st.success("Context submitted! This will help improve future analyses.")
                                            st.rerun()
                                        else:
                                            st.error(f"Failed to submit feedback: {feedback_result['error']}")
                        
                        else:
                            # Show feedback confirmation
                            if feedback_state['type'] == 'positive':
                                st.markdown("""
                                    <div class="feedback-section">
                                        <h4>Thanks for the positive feedback!</h4>
                                        <p>Your confirmation helps our agents learn successful analysis patterns.</p>
                                    </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                    <div class="feedback-section">
                                        <h4>Feedback Submitted Successfully!</h4>
                                        <p>Thank you for helping improve our multi-agent system.</p>
                                    </div>
                                """, unsafe_allow_html=True)
                        
                        # Export options
                        st.markdown("---")
                        st.markdown("### Export Analysis")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            # CSV export
                            csv_data = f"Analysis ID,Title,Description,Has Document,Flag,Confidence,Risk Level,Reasoning\n"
                            csv_data += f'"{analysis_id}","{title}","{description}","{feature_document is not None}","{flag}",{confidence},"{risk_level}","{result.get("reasoning", "")}"'
                            
                            st.download_button(
                                label="Download CSV Report",
                                data=csv_data,
                                file_name=f"compliance_report_{analysis_id}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        
                        with col2:
                            # JSON export
                            export_data = {
                                "analysis_id": analysis_id,
                                "feature": {"title": title, "description": description, "has_document": feature_document is not None},
                                "result": result,
                                "timestamp": datetime.now().isoformat()
                            }
                            json_data = json.dumps(export_data, indent=2)
                            st.download_button(
                                label="Download JSON Report",
                                data=json_data,
                                file_name=f"compliance_analysis_{analysis_id}.json",
                                mime="application/json",
                                use_container_width=True
                            )
                        
                        with col3:
                            if st.button("Analyze Another Feature", use_container_width=True):
                                # Clear current analysis state
                                st.session_state.current_analysis_id = None
                                st.rerun()
                
                except Exception as e:
                    st.markdown(f"""
                        <div class="warning-badge">
                            Error during analysis: {str(e)}
                        </div>
                    """, unsafe_allow_html=True)

# ================================================
# Page 2: Upload Regulations (unchanged)
# ================================================
elif page == "Upload Regulations":
    st.markdown("""
        <div class="main-header">
            <h1>Upload Regulations</h1>
            <p>Upload PDF regulation documents to enhance the compliance knowledge base</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Upload interface
    st.markdown("### Upload Regulation Documents")
    
    # File uploader with enhanced styling
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload regulation PDF documents to be indexed and added to the compliance knowledge base"
    )
    
    # Metadata input fields for regulation documents
    if uploaded_files:
        st.markdown("### Document Metadata")
        st.markdown("Please provide metadata for each uploaded regulation document:")
        
        # Initialize metadata storage in session state
        if 'pdf_metadata' not in st.session_state:
            st.session_state.pdf_metadata = {}
        
        # Create metadata inputs for each uploaded file
        metadata_forms = []
        for i, uploaded_file in enumerate(uploaded_files):
            file_key = f"{uploaded_file.name}_{uploaded_file.size}"
            
            # Initialize metadata for this file if not exists
            if file_key not in st.session_state.pdf_metadata:
                st.session_state.pdf_metadata[file_key] = {
                    'regulation_name': '',
                    'geo_jurisdiction': ''
                }
            
            with st.expander(f"{uploaded_file.name}", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    regulation_name = st.text_input(
                        "Regulation Name/Title*",
                        value=st.session_state.pdf_metadata[file_key]['regulation_name'],
                        placeholder="e.g., Utah Social Media Regulation Act",
                        help="Enter the official name or title of the regulation",
                        key=f"reg_name_{i}_{file_key}"
                    )
                    st.session_state.pdf_metadata[file_key]['regulation_name'] = regulation_name
                
                with col2:
                    geo_jurisdiction = st.text_input(
                        "Geographic Jurisdiction*",
                        value=st.session_state.pdf_metadata[file_key]['geo_jurisdiction'],
                        placeholder="e.g., Utah, USA or European Union",
                        help="Enter the geographic jurisdiction where this regulation applies",
                        key=f"geo_jurisdiction_{i}_{file_key}"
                    )
                    st.session_state.pdf_metadata[file_key]['geo_jurisdiction'] = geo_jurisdiction
                
                # Validation indicators
                if regulation_name and geo_jurisdiction:
                    st.success("Metadata complete")
                else:
                    st.warning("⚠️ Please fill in both regulation name and geographic jurisdiction")
        
        # Check if all metadata is complete
        all_metadata_complete = all(
            st.session_state.pdf_metadata.get(f"{f.name}_{f.size}", {}).get('regulation_name') and
            st.session_state.pdf_metadata.get(f"{f.name}_{f.size}", {}).get('geo_jurisdiction')
            for f in uploaded_files
        )
    
    if uploaded_files:
        st.markdown("### Upload Status")
        
        # Display files to be uploaded
        for uploaded_file in uploaded_files:
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.write(f"**{uploaded_file.name}** ({uploaded_file.size:,} bytes)")
            
            with col2:
                st.write("Ready")
        
        # Batch upload controls
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            upload_disabled = not all_metadata_complete
            button_text = "🚀 Upload All Files" if all_metadata_complete else "⚠️ Complete Metadata First"
            
            if st.button(button_text, use_container_width=True, type="primary", disabled=upload_disabled):
                if all_metadata_complete:
                    with st.spinner(f"Processing {len(uploaded_files)} PDF files through the ingestion pipeline..."):
                        try:
                            # Prepare metadata for each file
                            files_with_metadata = []
                            for uploaded_file in uploaded_files:
                                file_key = f"{uploaded_file.name}_{uploaded_file.size}"
                                metadata = st.session_state.pdf_metadata[file_key]
                                files_with_metadata.append({
                                    'file': uploaded_file,
                                    'regulation_name': metadata['regulation_name'],
                                    'geo_jurisdiction': metadata['geo_jurisdiction']
                                })
                            
                            response = upload_regulation_files_batch(files_with_metadata)
                            
                            if response and response.status_code == 200:
                                result = response.json()
                                st.success(f"{result.get('message', 'Files uploaded successfully!')}")
                                
                                # Clear metadata after successful upload
                                st.session_state.pdf_metadata = {}
                                st.rerun()
                            else:
                                st.error(f"Upload failed: {str(response)}")
                        except Exception as e:
                            st.error(f"❌Upload error: {str(e)}")
                else:
                    st.error("❌ Please complete metadata for all files before uploading")

    # Supported regulations info
    st.markdown("---")
    st.markdown("### Currently Supported Regulations")
    
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
            st.write(f"**{reg['name']}**")
        with col2:
            st.write(reg['status'])
        with col3:
            st.write(reg['coverage'])

# ================================================
# Page 3: Analytics Dashboard (unchanged)
# ================================================
elif page == "Analytics Dashboard":
    st.markdown("""
        <div class="main-header">
            <h1>Analytics Dashboard</h1>
            <p>Monitor compliance analysis trends and system performance</p>
        </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.analysis_history:
        st.markdown("""
            <div class="feature-card" style="text-align: center;">
                <h3>No Analysis Data Yet</h3>
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
            st.metric("Total Analyses", total_analyses)
        with col2:
            st.metric("Flagged Features", flagged_count)
        with col3:
            flagged_rate = (flagged_count / total_analyses * 100) if total_analyses > 0 else 0
            st.metric("Flagged Rate", f"{flagged_rate:.1f}%")
        with col4:
            avg_confidence = sum(analysis.get('confidence', 0) for analysis in st.session_state.analysis_history) / total_analyses
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        
        # Analysis history table
        st.markdown("### Recent Analysis History")
        
        if st.session_state.analysis_history:
            df = pd.DataFrame(st.session_state.analysis_history)
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            
            # Display enhanced table
            display_cols = ['timestamp', 'title', 'description', 'has_document', 'flag', 'confidence', 'risk_level']
            st.dataframe(
                df[display_cols],
                use_container_width=True
            )

# --- Footer ---
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: rgba(255,255,255,0.6); padding: 2rem;">
        <p><strong>TikTok TechJam 2025</strong> | Geo-Regulation Compliance System</p>
        <p>Built with Streamlit • Powered by LLM Agents • Ensuring Global Compliance</p>
        <p style="font-size: 0.8rem;">Mobile-Optimized | iOS & Android Compatible</p>
    </div>
""", unsafe_allow_html=True)