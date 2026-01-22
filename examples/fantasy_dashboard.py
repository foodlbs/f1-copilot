"""
F1 Fantasy Dashboard - Streamlit App

Simple dashboard for F1 fantasy recommendations using the vector database.

Run with: streamlit run examples/fantasy_dashboard.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import streamlit as st
from vector_db import F1VectorDB
from dotenv import load_dotenv

load_dotenv()

# Page config
st.set_page_config(
    page_title="F1 Fantasy Assistant",
    page_icon="🏎️",
    layout="wide"
)

# Initialize
@st.cache_resource
def load_vector_db():
    return F1VectorDB()

vdb = load_vector_db()

# Title
st.title("🏎️ F1 Fantasy Lineup Assistant")
st.markdown("AI-powered driver recommendations using historical data and performance trends")

# Sidebar
st.sidebar.header("Settings")

# Get circuits
circuits = [
    "Bahrain International Circuit",
    "Jeddah Corniche Circuit",
    "Albert Park Grand Prix Circuit",
    "Baku City Circuit",
    "Miami International Autodrome",
    "Autodromo Enzo e Dino Ferrari",
    "Circuit de Monaco",
    "Circuit de Barcelona-Catalunya",
    "Circuit Gilles Villeneuve",
    "Red Bull Ring",
    "Silverstone Circuit",
    "Hungaroring",
    "Circuit de Spa-Francorchamps",
    "Circuit Zandvoort",
    "Autodromo Nazionale di Monza",
    "Marina Bay Street Circuit",
    "Suzuka Circuit",
    "Losail International Circuit",
    "Circuit of the Americas",
    "Autódromo Hermanos Rodríguez",
    "Autódromo José Carlos Pace",
    "Las Vegas Street Circuit",
    "Yas Marina Circuit"
]

selected_circuit = st.sidebar.selectbox("Select Circuit", circuits)

strategy_type = st.sidebar.radio(
    "Strategy Focus",
    ["Balanced", "Qualifiers", "Overtakers"]
)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["🏁 Recommendations", "🏟️ Circuit Analysis", "👥 Head-to-Head", "❓ Ask AI"])

# Tab 1: Recommendations
with tab1:
    st.header(f"Driver Recommendations for {selected_circuit}")

    if st.button("Get Recommendations", type="primary", use_container_width=True):
        with st.spinner("Analyzing drivers..."):
            # Build query based on strategy
            if strategy_type == "Qualifiers":
                query = f"{selected_circuit} strong qualifier good grid position"
            elif strategy_type == "Overtakers":
                query = f"{selected_circuit} gains positions overtaking race pace"
            else:
                query = f"{selected_circuit} consistent good form best performers"

            # Search
            results = vdb.search_similar_races(
                query=query,
                top_k=10,
                filter_dict={'type': 'fantasy_driver_circuit', 'circuit': selected_circuit}
            )

            if not results:
                st.warning(f"No data found for {selected_circuit}")
            else:
                st.success(f"Found {len(results)} recommendations!")

                # Display results
                for i, result in enumerate(results, 1):
                    with st.expander(
                        f"#{i}. {result['metadata']['driver_name']} ({result['metadata']['team']}) - Score: {result['score']:.3f}",
                        expanded=(i <= 3)
                    ):
                        st.markdown(f"**Relevance Score:** {result['score']:.3f}")
                        st.markdown(f"**Team:** {result['metadata']['team']}")
                        st.markdown("---")
                        st.markdown(f"**Analysis:**")
                        st.info(result['text'])

# Tab 2: Circuit Analysis
with tab2:
    st.header(f"Circuit Analysis: {selected_circuit}")

    if st.button("Analyze Circuit", type="primary", use_container_width=True):
        with st.spinner("Analyzing circuit characteristics..."):
            results = vdb.search_similar_races(
                query=f"{selected_circuit} characteristics overtaking qualifying",
                top_k=1,
                filter_dict={'type': 'fantasy_circuit_analysis', 'circuit': selected_circuit}
            )

            if not results:
                st.warning(f"No analysis available for {selected_circuit}")
            else:
                analysis = results[0]['text']
                st.success("Analysis complete!")

                # Parse and display
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("🏁 Circuit Characteristics")
                    lines = analysis.split('|')
                    for line in lines[:4]:
                        st.write(line.strip())

                with col2:
                    st.subheader("📊 Strategy Implications")

                    # Generate recommendations
                    if "very high" in analysis and "qualifying" in analysis:
                        st.info("🎯 **Focus on Qualifiers**\n\nQualifying is critical here. Pick drivers who excel in qualifying.")
                    elif "high overtaking" in analysis:
                        st.info("🏁 **Pick Overtakers**\n\nOvertaking is possible. Race pace matters more than grid position.")
                    elif "low overtaking" in analysis:
                        st.warning("⚠️ **Qualifying Matters**\n\nDifficult to overtake. Starting position is very important.")

                    if "high risk" in analysis:
                        st.error("⚡ **High DNF Risk**\n\nConsider driver/team reliability when making picks.")

                st.markdown("---")
                st.markdown("**Full Analysis:**")
                st.code(analysis, language=None)

# Tab 3: Head-to-Head
with tab3:
    st.header("👥 Driver Comparison")

    col1, col2 = st.columns(2)

    with col1:
        driver1 = st.text_input("Driver 1 Code", "VER", help="e.g., VER, HAM, LEC")

    with col2:
        driver2 = st.text_input("Driver 2 Code", "PER", help="e.g., VER, HAM, LEC")

    if st.button("Compare Drivers", type="primary", use_container_width=True):
        with st.spinner("Comparing drivers..."):
            # Try both orders
            results = vdb.search_similar_races(
                query=f"{driver1} vs {driver2} head to head",
                top_k=1,
                filter_dict={'type': 'fantasy_head_to_head'}
            )

            if not results:
                results = vdb.search_similar_races(
                    query=f"{driver2} vs {driver1} head to head",
                    top_k=1,
                    filter_dict={'type': 'fantasy_head_to_head'}
                )

            if not results:
                st.warning(f"No head-to-head data for {driver1} vs {driver2}")
            else:
                st.success("Comparison found!")
                st.info(results[0]['text'])

                # Also show individual profiles
                st.markdown("---")
                st.subheader("Individual Profiles")

                col1, col2 = st.columns(2)

                with col1:
                    profile1 = vdb.search_similar_races(
                        query=f"{driver1} driver profile",
                        top_k=1,
                        filter_dict={'type': 'fantasy_driver_profile', 'driver_code': driver1}
                    )
                    if profile1:
                        st.markdown(f"**{profile1[0]['metadata']['driver_name']}**")
                        st.text(profile1[0]['text'])

                with col2:
                    profile2 = vdb.search_similar_races(
                        query=f"{driver2} driver profile",
                        top_k=1,
                        filter_dict={'type': 'fantasy_driver_profile', 'driver_code': driver2}
                    )
                    if profile2:
                        st.markdown(f"**{profile2[0]['metadata']['driver_name']}**")
                        st.text(profile2[0]['text'])

# Tab 4: Ask AI
with tab4:
    st.header("❓ Ask the AI Assistant")

    question = st.text_area(
        "Ask any F1 fantasy question:",
        placeholder="e.g., Who should I pick for Monaco?\nWhich drivers are in good form?\nIs qualifying important at Monza?",
        height=100
    )

    if st.button("Get Answer", type="primary", use_container_width=True):
        if not question:
            st.warning("Please enter a question")
        else:
            with st.spinner("Thinking..."):
                # Search for relevant info
                results = vdb.search_similar_races(
                    query=question,
                    top_k=5
                )

                if not results:
                    st.error("I don't have enough data to answer that question.")
                else:
                    st.success("Here's what I found:")

                    # Display top 3 results
                    for i, result in enumerate(results[:3], 1):
                        with st.expander(f"Source {i} (Relevance: {result['score']:.3f})", expanded=True):
                            st.markdown(f"**Type:** {result['metadata'].get('type', 'Unknown')}")
                            st.markdown("---")
                            st.write(result['text'])

                    # Simple summary
                    st.markdown("---")
                    st.subheader("💡 Quick Answer")
                    st.info("Based on the data above, consider the drivers and circuits with the highest relevance scores for your fantasy picks.")

# Sidebar stats
st.sidebar.markdown("---")
st.sidebar.header("Database Stats")
stats = vdb.get_stats()
st.sidebar.metric("Total Vectors", f"{stats['total_vectors']:,}")
st.sidebar.metric("Dimension", stats['dimension'])

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
This F1 Fantasy Assistant uses AI and historical race data to provide intelligent driver recommendations.

Features:
- Circuit-specific analysis
- Recent form tracking
- Head-to-head comparisons
- Strategy recommendations
""")
