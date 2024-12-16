import streamlit as st
import streamlit.components.v1 as components
import os

def main():
    # Set page config
    st.set_page_config(
        page_title="K-Monitor - EKR Graphs",
        page_icon="🔍",
        layout="wide"
    )

    # Title and description
    st.title("K-Monitor - EKR Graphs 🎯")
    st.markdown("""
    This application displays interactive visualizations of public procurement networks.
    Hovering over nodes and edges reveals details. The edges represent the size of the contract part.
    Community-based coloring (clustering) was performed using the Louvain method.
    """)

    # Styling
    st.markdown("""
        <style>
        .stSelectbox {
            margin-bottom: 2rem;
        }
        .stDownloadButton {
            margin-top: 1rem;
        }
        .main .block-container {
            padding-top: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Get list of HTML files from output/visualizations
    vis_path = "output/visualizations"
    html_files = [f for f in os.listdir(vis_path) 
                 if f.endswith('.html') 
                 and not f.startswith(('main_network', 'full_network', 'total_network'))]
    
    if not html_files:
        st.error("Nem található vizualizációs fájl az output/visualizations mappában.")
        return

    # Group visualizations by type
    type_colors = [f for f in html_files if 'type_colors' in f]
    community_colors = [f for f in html_files if 'community_colors' in f]

    # Create tabs for different visualization types
    tab1, tab2 = st.tabs(["Type based coloring", "Community based coloring"])

    with tab1:
        if type_colors:
            selected_vis = st.selectbox(
                "Válasszon típus alapú vizualizációt:",
                type_colors,
                format_func=lambda x: x.replace('_type_colors.html', '')
                            .replace('_', ' ')
                            .replace('network', '')
                            .strip()
                            .title()
            )
            display_visualization(selected_vis, vis_path)
        else:
            st.warning("No type based visualization found.")

    with tab2:
        if community_colors:
            selected_vis = st.selectbox(
                "Select a community based visualization:",
                community_colors,
                format_func=lambda x: x.replace('_community_colors.html', '')
                            .replace('_', ' ')
                            .replace('network', '')
                            .strip()
                            .title()
            )
            display_visualization(selected_vis, vis_path)
        else:
            st.warning("Nem található közösség alapú vizualizáció.")

def display_visualization(selected_vis: str, vis_path: str):
    """Display the selected visualization and add download button"""
    if selected_vis:
        # Read the HTML file
        with open(os.path.join(vis_path, selected_vis), 'r', encoding='utf-8') as f:
            html_content = f.read()
            
        # Display in full width
        with st.container():
            # Use components.html with scrolling and full height
            components.html(
                html_content,
                height=800,
                scrolling=True
            )

        # Add download button
        st.download_button(
            label="Download visualization",
            data=html_content,
            file_name=selected_vis,
            mime="text/html"
        )

if __name__ == "__main__":
    main() 