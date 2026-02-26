#!/usr/bin/env python3
"""Launch AEROTICA monitoring dashboard."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import streamlit as st
except ImportError:
    print("❌ Streamlit not installed. Run: pip install streamlit")
    sys.exit(1)


def main():
    """Launch dashboard."""
    st.set_page_config(
        page_title="AEROTICA Dashboard",
        page_icon="🌪️",
        layout="wide"
    )
    
    st.title("🌪️ AEROTICA")
    st.markdown("### Atmospheric Kinetic Energy Mapping Framework")
    
    # Sidebar
    with st.sidebar:
        st.header("Controls")
        site = st.selectbox(
            "Select Site",
            ["Tokyo", "Brest", "Edinburgh", "North Sea"]
        )
        refresh = st.button("Refresh Data")
    
    # Main content
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("AKE Score")
        st.metric("Current AKE", "0.84", "+0.03")
        st.metric("Classification", "VIABLE")
    
    with col2:
        st.subheader("Gust Risk")
        st.metric("Risk Level", "MODERATE")
        st.metric("Lead Time", "4.8 min")
    
    with col3:
        st.subheader("Parameters")
        st.metric("KED (22%)", "0.83")
        st.metric("THD (10%)", "0.72")
    
    # Charts
    st.subheader("Parameter Contributions")
    
    import pandas as pd
    import numpy as np
    
    params = pd.DataFrame({
        'Parameter': ['KED', 'TII', 'VSR', 'AOD', 'THD', 'PGF', 'HCI', 'ASI', 'LRC'],
        'Score': [0.83, 0.76, 0.89, 0.34, 0.72, 0.65, 0.59, 0.71, 0.44],
        'Weight': [22, 16, 14, 12, 10, 8, 7, 6, 5]
    })
    
    st.bar_chart(params.set_index('Parameter')['Score'])
    
    # Info
    with st.expander("About AEROTICA"):
        st.write("""
        AEROTICA integrates nine parameters into a single Atmospheric Kinetic Efficiency (AKE) index:
        - **KED**: Kinetic Energy Density (22%)
        - **TII**: Turbulence Intensity Index (16%)
        - **VSR**: Vertical Shear Ratio (14%)
        - **AOD**: Aerosol Optical Depth (12%)
        - **THD**: Thermal Helicity Dynamics (10%)
        - **PGF**: Pressure Gradient Force (8%)
        - **HCI**: Humidity-Convection Interaction (7%)
        - **ASI**: Atmospheric Stability Integration (6%)
        - **LRC**: Local Roughness Coefficient (5%)
        """)


if __name__ == "__main__":
    main()
