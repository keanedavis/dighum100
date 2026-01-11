import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os 

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="WNBA Attendance & Trends Dashboard", page_icon="🏀")

# --- Professional Color Palette ---
COLORS = {
    'primary': '#FF6B35',      # Vibrant orange (WNBA-inspired)
    'secondary': '#004E89',    # Deep blue
    'accent': '#F77F00',       # Warm amber
    'success': '#06A77D',      # Teal green
    'text_dark': '#2C3E50',    # Dark slate
    'text_light': '#7F8C8D',   # Light gray
    'background': '#F8F9FA',   # Off-white
    'card_bg': '#FFFFFF',      # Pure white
    'grid': '#E9ECEF'          # Light grid
}

# Team colors - professional palette
TEAM_COLORS = [
    '#FF6B35', '#004E89', '#F77F00', '#06A77D', '#8338EC', 
    '#FB5607', '#3A86FF', '#06FFA5', '#C1121F', '#FF006E',
    '#8338EC', '#FFB703', '#023047', '#2A9D8F', '#E63946'
] * 3  # Repeat to ensure enough colors

def get_team_color_map(teams):
    color_map = {}
    for i, team in enumerate(teams):
        color_map[team] = TEAM_COLORS[i % len(TEAM_COLORS)]
    return color_map

# --- Load Attendance Data ---
@st.cache_data
def load_attendance_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df['Attendance'] = pd.to_numeric(df['Attendance'], errors='coerce')
        df.dropna(subset=['Attendance'], inplace=True)
        df['Day'] = pd.to_numeric(df['Day'], errors='coerce')
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df.dropna(subset=['Year', 'Month', 'Day'], inplace=True)
        df['Year'] = df['Year'].astype(int)
        df['Day'] = df['Day'].astype(int)
        df['Date_Str'] = df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-' + df['Day'].astype(str)
        df['Date'] = pd.to_datetime(df['Date_Str'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        df['MonthYear'] = df['Date'].dt.to_period('M').astype(str)
        df['DayOfWeek'] = df['Date'].dt.day_name()
        df['MonthName'] = df['Date'].dt.strftime('%b')
        df = df.sort_values(by='Date')
        return df
    except FileNotFoundError:
        st.error(f"Error: The attendance file '{file_path}' was not found.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred while loading the attendance data: {e}")
        return pd.DataFrame()

# --- Load Media Data ---
@st.cache_data
def load_media_data(file_path):
    try:
        df = pd.read_csv(file_path)
        if 'publish_date' in df.columns:
            df['publish_date'] = pd.to_datetime(df['publish_date'], errors='coerce')
            df.dropna(subset=['publish_date'], inplace=True)
            df.rename(columns={'publish_date': 'Date'}, inplace=True)
        else:
            st.warning("Column 'publish_date' not found in media data.")
            return pd.DataFrame()
        df['Mentions'] = 1 
        df['MonthYear'] = df['Date'].dt.to_period('M').astype(str)
        df = df.sort_values(by='Date')
        return df
    except FileNotFoundError:
        st.error(f"Error: The media file '{file_path}' was not found.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred while loading the media data: {e}")
        return pd.DataFrame()

# --- Load Google Search Trend Data ---
@st.cache_data
def load_search_trends_data(file_path):
    try:
        df = pd.read_csv(file_path)
        if 'Month' in df.columns:
            df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
            df.dropna(subset=['Month'], inplace=True)
            df.rename(columns={'Month': 'Date'}, inplace=True)
        else:
            st.warning(f"Column 'Month' not found in search trends data.")
            return pd.DataFrame()
        if 'WNBA' in df.columns:
            df['WNBA'] = pd.to_numeric(df['WNBA'], errors='coerce')
            df.dropna(subset=['WNBA'], inplace=True)
        else:
            st.warning(f"Column 'WNBA' not found in search trends data.")
            return pd.DataFrame()
        df['MonthYear'] = df['Date'].dt.to_period('M').astype(str)
        df = df.sort_values(by='Date')
        return df
    except FileNotFoundError:
        st.error(f"Error: The search trends file '{file_path}' was not found.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred while loading the search trends data: {e}")
        return pd.DataFrame()

attendance_file = "All Game Attendance.csv" 
df_attendance = load_attendance_data(attendance_file)

media_file = "media.csv" 
df_media = load_media_data(media_file)

search_trends_file = "google.csv" 
df_search_trends = load_search_trends_data(search_trends_file)

# --- Professional CSS Styling ---
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    
    .main {{
        background-color: {COLORS['background']};
    }}
    
    h1 {{
        color: {COLORS['text_dark']};
        font-weight: 700;
        font-size: 2.8rem !important;
        margin-bottom: 0.5rem !important;
        letter-spacing: -0.5px;
    }}
    
    h2 {{
        color: {COLORS['text_dark']};
        font-weight: 600;
        font-size: 1.8rem !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
        border-left: 4px solid {COLORS['primary']};
        padding-left: 1rem;
    }}
    
    h3 {{
        color: {COLORS['secondary']};
        font-weight: 600;
        font-size: 1.3rem !important;
    }}
    
    .stMetric {{
        background-color: {COLORS['card_bg']};
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid {COLORS['grid']};
    }}
    
    .stMetric label {{
        color: {COLORS['text_light']} !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    .stMetric [data-testid="stMetricValue"] {{
        color: {COLORS['primary']} !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }}
    
    .stMultiSelect, .stSelectbox {{
        background-color: {COLORS['card_bg']};
        border-radius: 8px;
    }}
    
    div[data-testid="stExpander"] {{
        background-color: {COLORS['card_bg']};
        border-radius: 8px;
        border: 1px solid {COLORS['grid']};
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }}
    
    .streamlit-expanderHeader {{
        font-weight: 600;
        color: {COLORS['secondary']};
    }}
    
    hr {{
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(to right, transparent, {COLORS['grid']}, transparent);
    }}
    
    .stButton>button {{
        background-color: {COLORS['primary']};
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 2px 6px rgba(255,107,53,0.3);
        transition: all 0.3s ease;
    }}
    
    .stButton>button:hover {{
        background-color: {COLORS['accent']};
        box-shadow: 0 4px 12px rgba(247,127,0,0.4);
        transform: translateY(-2px);
    }}
    
    .intro-text {{
        background: linear-gradient(135deg, {COLORS['card_bg']} 0%, #F0F4F8 100%);
        padding: 2rem;
        border-radius: 12px;
        border-left: 4px solid {COLORS['primary']};
        margin-bottom: 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }}
    
    .intro-text p {{
        color: {COLORS['text_dark']};
        font-size: 1.1rem;
        line-height: 1.6;
        margin: 0;
    }}
    
    div[data-testid="stSidebar"] {{
        background-color: {COLORS['card_bg']};
        border-right: 1px solid {COLORS['grid']};
    }}
    
    div[data-testid="stSidebar"] h2 {{
        color: {COLORS['secondary']};
        border-left: none;
        padding-left: 0;
    }}
    
    .css-1d391kg {{
        padding: 2rem 1rem;
    }}
</style>
""", unsafe_allow_html=True)

# --- Streamlit App Layout ---
if not df_attendance.empty:
    # --- Sidebar Settings ---
    st.sidebar.markdown(f"<h2 style='color: {COLORS['secondary']}; font-size: 1.5rem; margin-bottom: 1.5rem;'>⚙️ Dashboard Controls</h2>", unsafe_allow_html=True)
    
    font_size = st.sidebar.slider("Text Size", min_value=12, max_value=24, value=15, step=1)

    # Inject font size CSS
    css = f"""
    <style>
        body, p, li, table, .stMarkdown, .stText {{
            font-size: {font_size}px !important;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    # --- Header ---
    st.markdown(f"<h1 style='text-align: center; color: {COLORS['text_dark']};'>🏀 WNBA Attendance & Trends Dashboard</h1>", unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class='intro-text'>
            <p>Explore comprehensive WNBA attendance data, media coverage trends, and search interest metrics. 
            Use the interactive controls below to filter and analyze patterns across seasons, teams, and venues.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Year Slider ---
    st.markdown(f"<h2>📅 Time Period Selection</h2>", unsafe_allow_html=True)
    all_years_in_data = sorted(df_attendance['Year'].unique().tolist())
    
    if len(all_years_in_data) > 1:
        min_year, max_year = st.slider(
            "Select Year Range",
            min_value=min(all_years_in_data),
            max_value=max(all_years_in_data),
            value=(min(all_years_in_data), max(all_years_in_data)),
            step=1
        )
        selected_years_slider = list(range(min_year, max_year + 1))
    elif len(all_years_in_data) == 1:
        st.info(f"Data available for {all_years_in_data[0]} only")
        selected_years_slider = all_years_in_data
    else:
        st.warning("No year data available")
        selected_years_slider = []

    # --- Sidebar Filters ---
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"<h3 style='color: {COLORS['secondary']}; font-size: 1.1rem;'>🎯 Filter Options</h3>", unsafe_allow_html=True)

    filter_offseason_smooth = st.sidebar.checkbox(
        "Show Offseason Trend",
        value=True,
        help="Display continuous trend line including offseason months"
    )

    all_game_types = df_attendance['Game Type'].unique().tolist()
    selected_game_types = st.sidebar.multiselect(
        "Game Type",
        options=all_game_types,
        default=all_game_types
    )

    all_home_teams = sorted(df_attendance['Home Team'].unique().tolist())
    selected_home_teams = st.sidebar.multiselect(
        "Home Team",
        options=all_home_teams,
        default=all_home_teams
    )

    all_away_teams = sorted(df_attendance['Away Team'].unique().tolist())
    selected_away_teams = st.sidebar.multiselect(
        "Away Team",
        options=all_away_teams,
        default=all_away_teams
    )

    all_cities = sorted(df_attendance['City'].unique().tolist())
    selected_cities = st.sidebar.multiselect(
        "City",
        options=all_cities,
        default=all_cities
    )

    all_states = sorted(df_attendance['State'].unique().tolist())
    selected_states = st.sidebar.multiselect(
        "State",
        options=all_states,
        default=all_states
    )
    
    all_arenas = sorted(df_attendance['Arena'].unique().tolist())
    selected_arenas = st.sidebar.multiselect(
        "Arena",
        options=all_arenas,
        default=all_arenas
    )

    st.sidebar.markdown("---")

    aggregation_level = st.sidebar.radio(
        "Trend Aggregation",
        ('Daily', 'Monthly', 'Yearly'),
        index=2
    )
    
    offseason_months = [11, 12, 1, 2, 3, 4]

    # Filter logic
    teams_to_exclude = ['Team Delle Donne', 'Team WNBA']
    
    initial_filtered_df = df_attendance[
        (~df_attendance['Home Team'].isin(teams_to_exclude)) &
        (~df_attendance['Away Team'].isin(teams_to_exclude))
    ]

    if not filter_offseason_smooth:
        initial_filtered_df = initial_filtered_df[~initial_filtered_df['Date'].dt.month.isin(offseason_months)]

    filtered_df_attendance = initial_filtered_df[
        (initial_filtered_df['Year'].isin(selected_years_slider)) &
        (initial_filtered_df['Game Type'].isin(selected_game_types)) &
        (initial_filtered_df['Home Team'].isin(selected_home_teams)) &
        (initial_filtered_df['Away Team'].isin(selected_away_teams)) &
        (initial_filtered_df['City'].isin(selected_cities)) &
        (initial_filtered_df['State'].isin(selected_states)) &
        (initial_filtered_df['Arena'].isin(selected_arenas))
    ]

    st.markdown("---")

    # --- Main Content ---
    if filtered_df_attendance.empty:
        st.error("No data matches your filters. Please adjust your selections.")
    else:
        line_shape_value = 'spline' if filter_offseason_smooth else 'linear'

        # --- Key Metrics ---
        st.markdown(f"<h2>📊 Performance Overview</h2>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Games", f"{len(filtered_df_attendance):,}")
        with col2:
            st.metric("Avg Attendance", f"{filtered_df_attendance['Attendance'].mean():,.0f}")
        with col3:
            st.metric("Peak Attendance", f"{filtered_df_attendance['Attendance'].max():,.0f}")
        with col4:
            st.metric("Min Attendance", f"{filtered_df_attendance['Attendance'].min():,.0f}")

        st.markdown("---")

        # --- Attendance Trends ---
        st.markdown(f"<h2>📈 Attendance Trends Over Time</h2>", unsafe_allow_html=True)

        if aggregation_level == 'Daily':
            grouped_attendance = filtered_df_attendance.groupby('Date')['Attendance'].mean().reset_index()
            x_axis_col = 'Date'
        elif aggregation_level == 'Monthly':
            grouped_attendance = filtered_df_attendance.groupby('MonthYear')['Attendance'].mean().reset_index()
            grouped_attendance['MonthYear'] = pd.to_datetime(grouped_attendance['MonthYear'])
            grouped_attendance = grouped_attendance.sort_values(by='MonthYear')
            x_axis_col = 'MonthYear'
        else:
            grouped_attendance = filtered_df_attendance.groupby('Year')['Attendance'].mean().reset_index()
            x_axis_col = 'Year'
        
        grouped_attendance.rename(columns={'Attendance': 'Average Attendance'}, inplace=True)

        fig_line_attendance = go.Figure()
        fig_line_attendance.add_trace(go.Scatter(
            x=grouped_attendance[x_axis_col],
            y=grouped_attendance['Average Attendance'],
            mode='lines+markers',
            name='Attendance',
            line=dict(color=COLORS['primary'], width=3, shape=line_shape_value),
            marker=dict(size=8, color=COLORS['primary'], line=dict(width=2, color='white')),
            fill='tozeroy',
            fillcolor=f'rgba(255, 107, 53, 0.1)',
            hovertemplate='<b>%{x}</b><br>Attendance: %{y:,.0f}<extra></extra>'
        ))

        fig_line_attendance.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Inter', color=COLORS['text_dark']),
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(
                title=x_axis_col.replace('MonthYear', 'Month & Year'),
                showgrid=True,
                gridcolor=COLORS['grid'],
                linecolor=COLORS['grid']
            ),
            yaxis=dict(
                title='Average Attendance',
                showgrid=True,
                gridcolor=COLORS['grid'],
                linecolor=COLORS['grid']
            ),
            hovermode="x unified",
            hoverlabel=dict(bgcolor='white', font_size=13, font_family='Inter')
        )
        st.plotly_chart(fig_line_attendance, use_container_width=True)

        st.markdown("---")
        
        # --- Google Search Trends Overlay ---
        st.markdown(f"<h2>🔍 Attendance vs. Search Interest</h2>", unsafe_allow_html=True)

        if not df_search_trends.empty:
            if 'Date' in df_search_trends.columns:
                filtered_df_search_trends = df_search_trends[df_search_trends['Date'].dt.year.isin(selected_years_slider)]
                if not filter_offseason_smooth:
                    filtered_df_search_trends = filtered_df_search_trends[~filtered_df_search_trends['Date'].dt.month.isin(offseason_months)]
            else:
                filtered_df_search_trends = pd.DataFrame()

            if not filtered_df_search_trends.empty:
                if filter_offseason_smooth:
                    monthly_avg_attendance_for_overlay = initial_filtered_df.groupby('MonthYear')['Attendance'].mean().reset_index()
                else:
                    monthly_avg_attendance_for_overlay = filtered_df_attendance.groupby('MonthYear')['Attendance'].mean().reset_index()
                
                monthly_avg_attendance_for_overlay['MonthYear'] = pd.to_datetime(monthly_avg_attendance_for_overlay['MonthYear'])
                monthly_avg_attendance_for_overlay = monthly_avg_attendance_for_overlay.sort_values(by='MonthYear')
                monthly_avg_attendance_for_overlay.rename(columns={'Attendance': 'Average Attendance'}, inplace=True)

                monthly_search_trends = filtered_df_search_trends.groupby('MonthYear')['WNBA'].mean().reset_index()
                monthly_search_trends['MonthYear'] = pd.to_datetime(monthly_search_trends['MonthYear'])
                monthly_search_trends = monthly_search_trends.sort_values(by='MonthYear')
                monthly_search_trends.rename(columns={'WNBA': 'Average WNBA Searches'}, inplace=True)

                combined_df_search = pd.merge(monthly_avg_attendance_for_overlay, monthly_search_trends, on='MonthYear', how='outer')
                combined_df_search = combined_df_search.sort_values(by='MonthYear')

                if not combined_df_search.empty:
                    fig_search_overlay = make_subplots(specs=[[{"secondary_y": True}]])

                    fig_search_overlay.add_trace(
                        go.Scatter(
                            x=combined_df_search['MonthYear'], 
                            y=combined_df_search['Average Attendance'], 
                            name='Attendance', 
                            mode='lines+markers', 
                            line=dict(color=COLORS['primary'], width=3, shape=line_shape_value),
                            marker=dict(size=7, color=COLORS['primary']),
                            hovertemplate='<b>%{x|%b %Y}</b><br>Attendance: %{y:,.0f}<extra></extra>'
                        ),
                        secondary_y=False,
                    )

                    fig_search_overlay.add_trace(
                        go.Scatter(
                            x=combined_df_search['MonthYear'], 
                            y=combined_df_search['Average WNBA Searches'], 
                            name='Search Interest', 
                            mode='lines+markers', 
                            line=dict(color=COLORS['success'], width=3, shape=line_shape_value),
                            marker=dict(size=7, color=COLORS['success']),
                            hovertemplate='<b>%{x|%b %Y}</b><br>Searches: %{y:,.0f}<extra></extra>'
                        ),
                        secondary_y=True,
                    )

                    fig_search_overlay.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(family='Inter', color=COLORS['text_dark']),
                        margin=dict(l=20, r=20, t=40, b=20),
                        hovermode="x unified",
                        hoverlabel=dict(bgcolor='white', font_size=13, font_family='Inter'),
                        legend=dict(
                            x=0.01, y=0.99, 
                            bgcolor='rgba(255,255,255,0.9)', 
                            bordercolor=COLORS['grid'], 
                            borderwidth=1,
                            font=dict(size=12)
                        )
                    )

                    fig_search_overlay.update_xaxes(
                        title_text="Month & Year", 
                        showgrid=True, 
                        gridcolor=COLORS['grid'],
                        tickformat="%b %Y"
                    )
                    fig_search_overlay.update_yaxes(
                        title_text="Average Attendance", 
                        secondary_y=False, 
                        showgrid=True, 
                        gridcolor=COLORS['grid']
                    )
                    fig_search_overlay.update_yaxes(
                        title_text="Search Interest", 
                        secondary_y=True, 
                        showgrid=False
                    )

                    st.plotly_chart(fig_search_overlay, use_container_width=True)

        st.markdown("---")

        # --- Media Coverage Overlay ---
        st.markdown(f"<h2>📰 Attendance vs. Media Coverage</h2>", unsafe_allow_html=True)

        if not df_media.empty:
            if 'Date' in df_media.columns:
                filtered_df_media = df_media[df_media['Date'].dt.year.isin(selected_years_slider)]
                if not filter_offseason_smooth:
                    filtered_df_media = filtered_df_media[~filtered_df_media['Date'].dt.month.isin(offseason_months)]
            else:
                filtered_df_media = pd.DataFrame()

            if not filtered_df_media.empty:
                if filter_offseason_smooth:
                    monthly_avg_attendance = initial_filtered_df.groupby('MonthYear')['Attendance'].mean().reset_index()
                else:
                    monthly_avg_attendance = filtered_df_attendance.groupby('MonthYear')['Attendance'].mean().reset_index()

                monthly_avg_attendance['MonthYear'] = pd.to_datetime(monthly_avg_attendance['MonthYear'])
                monthly_avg_attendance = monthly_avg_attendance.sort_values(by='MonthYear')
                monthly_avg_attendance.rename(columns={'Attendance': 'Average Attendance'}, inplace=True)

                monthly_media_mentions = filtered_df_media.groupby('MonthYear')['Mentions'].sum().reset_index()
                monthly_media_mentions['MonthYear'] = pd.to_datetime(monthly_media_mentions['MonthYear'])
                monthly_media_mentions = monthly_media_mentions.sort_values(by='MonthYear')
                monthly_media_mentions.rename(columns={'Mentions': 'Total Media Mentions'}, inplace=True)

                combined_df = pd.merge(monthly_avg_attendance, monthly_media_mentions, on='MonthYear', how='outer')
                combined_df = combined_df.sort_values(by='MonthYear')

                if not combined_df.empty:
                    fig_overlay = make_subplots(specs=[[{"secondary_y": True}]])

                    fig_overlay.add_trace(
                        go.Scatter(
                            x=combined_df['MonthYear'], 
                            y=combined_df['Average Attendance'], 
                            name='Attendance', 
                            mode='lines+markers', 
                            line=dict(color=COLORS['primary'], width=3, shape=line_shape_value),
                            marker=dict(size=7, color=COLORS['primary']),
                            hovertemplate='<b>%{x|%b %Y}</b><br>Attendance: %{y:,.0f}<extra></extra>'
                        ),
                        secondary_y=False,
                    )

                    fig_overlay.add_trace(
                        go.Scatter(
                            x=combined_df['MonthYear'], 
                            y=combined_df['Total Media Mentions'], 
                            name='Media Coverage', 
                            mode='lines+markers', 
                            line=dict(color=COLORS['accent'], width=3, shape=line_shape_value),
                            marker=dict(size=7, color=COLORS['accent']),
                            hovertemplate='<b>%{x|%b %Y}</b><br>Mentions: %{y:,.0f}<extra></extra>'
                        ),
                        secondary_y=True,
                    )

                    fig_overlay.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(family='Inter', color=COLORS['text_dark']),
                        margin=dict(l=20, r=20, t=40, b=20),
                        hovermode="x unified",
                        hoverlabel=dict(bgcolor='white', font_size=13, font_family='Inter'),
                        legend=dict(
                            x=0.01, y=0.99, 
                            bgcolor='rgba(255,255,255,0.9)', 
                            bordercolor=COLORS['grid'], 
                            borderwidth=1,
                            font=dict(size=12)
                        )
                    )

                    fig_overlay.update_xaxes(
                        title_text="Month & Year", 
                        showgrid=True, 
                        gridcolor=COLORS['grid'],
                        tickformat="%b %Y"
                    )
                    fig_overlay.update_yaxes(
                        title_text="Average Attendance", 
                        secondary_y=False, 
                        showgrid=True, 
                        gridcolor=COLORS['grid']
                    )
                    fig_overlay.update_yaxes(
                        title_text="Media Mentions", 
                        secondary_y=True, 
                        showgrid=False
                    )

                    st.plotly_chart(fig_overlay, use_container_width=True)

        st.markdown("---")

        # --- Attendance by Home Team ---
        st.markdown(f"<h2>🏟️ Team Performance Analysis</h2>", unsafe_allow_html=True)
        avg_attendance_by_home_team = filtered_df_attendance.groupby('Home Team')['Attendance'].mean().reset_index()
        avg_attendance_by_
