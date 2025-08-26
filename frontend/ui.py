# streamlit_app.py
import streamlit as st
import requests

st.set_page_config(
    page_title="Geo-Regulation Compliance",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar ---
st.sidebar.title("Feature Compliance Checker")
st.sidebar.write("Enter your feature details and check geo-specific compliance logic.")

# --- Main Page ---
st.title("⚖️ Geo-Regulation Compliance System")
st.write("Automated flagging of features requiring location-specific compliance logic.\n"
         "Audit-ready reasoning and regulations are highlighted for clarity.")

# --- Input Section ---
st.header("Feature Input")
title = st.text_input("Feature Title")
description = st.text_area("Feature Description")

check_button = st.button("Check Compliance")

# --- Output Section ---
if check_button:
    if not title or not description:
        st.warning("Please fill in both Title and Description before checking.")
    else:
        with st.spinner("Analyzing feature compliance..."):
            # Call FastAPI backend
            payload = {
                "title": title,
                "description": description
            }
            try:
                response = requests.post("http://127.0.0.1:8000/check_compliance", json=payload)
                data = response.json()
                
                # Display compliance flag
                flag = data.get("flag", "Unknown")
                if flag.lower() == "yes":
                    st.success(f"✅ Feature requires geo-specific compliance logic")
                else:
                    st.info(f"ℹ️ Feature does NOT require geo-specific compliance logic")
                
                # Reasoning - collapsible for mobile
                with st.expander("📄 Reasoning"):
                    st.write(data.get("reasoning", "No reasoning provided"))
                
                # Related Regulations - collapsible for mobile
                related_regs = data.get("related_regulations", [])
                if related_regs:
                    with st.expander("📚 Related Regulations"):
                        for reg in related_regs:
                            st.write(f"- {reg}")
                
                # Optional: Download CSV
                csv_data = data.get("csv_output")
                if csv_data:
                    st.download_button(
                        label="⬇️ Download Compliance Report (CSV)",
                        data=csv_data,
                        file_name="compliance_report.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Error connecting to backend: {e}")

# --- Footer ---
st.markdown("---")
st.markdown("Developed for TikTok TechJam 2025 | Mobile-friendly MVP ✅")

# streamlit run frontend/ui.py