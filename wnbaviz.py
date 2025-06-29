import streamlit as st
import pandas as pd
import plotly.express as px

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="WNBA Attendance Dashboard", page_icon="🏀")

# --- Define a custom color palette for teams (using Plotly's qualitative colors) ---
# This ensures a variety of distinct colors for different teams
TEAM_COLORS = px.colors.qualitative.Alphabet + px.colors.qualitative.G10 + px.colors.qualitative.D3

# Function to assign a unique color to each team
def get_team_color_map(teams):
    color_map = {}
    for i, team in enumerate(teams):
        color_map[team] = TEAM_COLORS[i % len(TEAM_COLORS)] # Cycle through available colors
    return color_map

# --- Load Attendance Data ---
@st.cache_data # Cache data to improve performance
def load_attendance_data(file_path):
    """
    Loads the WNBA attendance data from a CSV file.
    Performs basic data cleaning: ensures Attendance is numeric and creates a datetime column.
    """
    try:
        df = pd.read_csv(file_path)
        
        # Convert 'Attendance' to numeric, coercing errors to NaN
        df['Attendance'] = pd.to_numeric(df['Attendance'], errors='coerce')
        
        # Drop rows where Attendance is NaN (if any non-numeric values were present)
        df.dropna(subset=['Attendance'], inplace=True)
        
        # --- Robust Date Creation ---
        # Ensure 'Day' and 'Year' are numeric first, coercing errors
        df['Day'] = pd.to_numeric(df['Day'], errors='coerce')
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

        # Drop rows where Year, Month, or Day are missing after initial loading,
        # as these are critical for date creation.
        # Assuming 'Month' column exists as either number or month name string.
        df.dropna(subset=['Year', 'Month', 'Day'], inplace=True)

        # Convert 'Year' and 'Day' to integer types after dropping NaNs
        df['Year'] = df['Year'].astype(int)
        df['Day'] = df['Day'].astype(int)

        # Create a combined date string in 'YYYY-MonthName-DD' or 'YYYY-MM-DD' format.
        # pandas.to_datetime is smart enough to parse various month formats (names or numbers).
        df['Date_Str'] = df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-' + df['Day'].astype(str)
        df['Date'] = pd.to_datetime(df['Date_Str'], errors='coerce')
        
        # Drop rows where date conversion failed (e.g., invalid date combinations)
        df.dropna(subset=['Date'], inplace=True) 
        # --- End Robust Date Creation ---

        # Create 'MonthYear' for monthly aggregation and 'DayOfWeek' for insights
        df['MonthYear'] = df['Date'].dt.to_period('M').astype(str)
        df['DayOfWeek'] = df['Date'].dt.day_name() # Get day name (Monday, Tuesday, etc.)
        df['MonthName'] = df['Date'].dt.strftime('%b') # Get short month name (Jan, Feb, etc.)

        # Sort by date for better time-series plotting
        df = df.sort_values(by='Date')

        return df
    except FileNotFoundError:
        st.error(f"Error: The attendance file '{file_path}' was not found. Please make sure it's in the same directory as this script.")
        return pd.DataFrame() # Return an empty DataFrame on error
    except Exception as e:
        st.error(f"An error occurred while loading or processing the attendance data: {e}")
        return pd.DataFrame()


attendance_file = "All Game Attendance.csv"
df_attendance = load_attendance_data(attendance_file)


# --- Streamlit App Layout ---
if not df_attendance.empty:
    st.title("🏀 WNBA Game Attendance Dashboard")

    st.markdown("""
        Welcome to the interactive WNBA Attendance Dashboard! Explore game attendance trends
        using the filters below and in the sidebar.
        """)

    st.markdown("---") # Visual separator

    # --- Year Slider (Main Content Area) ---
    st.header("Filter Data by Year Range")
    all_years_in_data = sorted(df_attendance['Year'].unique().tolist())
    
    if len(all_years_in_data) > 1:
        min_year, max_year = st.slider(
            "Select a Year Range",
            min_value=min(all_years_in_data),
            max_value=max(all_years_in_data),
            value=(min(all_years_in_data), max(all_years_in_data)),
            step=1,
            help="Drag the handles to select a specific range of years."
        )
        selected_years_slider = list(range(min_year, max_year + 1))
    elif len(all_years_in_data) == 1:
        st.info(f"Only data for year {all_years_in_data[0]} is available.")
        selected_years_slider = all_years_in_data
    else:
        st.warning("No year data available for filtering.")
        selected_years_slider = []


    # --- Sidebar Filters ---
    st.sidebar.header("Additional Filter Options")
    st.sidebar.markdown("Adjust these filters to refine the data displayed in the charts.")

    # Game Type Selector
    all_game_types = df_attendance['Game Type'].unique().tolist()
    selected_game_types = st.sidebar.multiselect(
        "Select Game Type(s)",
        options=all_game_types,
        default=all_game_types,
        help="Filter by Regular Season, Playoffs, All-Star, etc."
    )

    # Home Team Selector
    all_home_teams = sorted(df_attendance['Home Team'].unique().tolist())
    selected_home_teams = st.sidebar.multiselect(
        "Select Home Team(s)",
        options=all_home_teams,
        default=all_home_teams,
        help="Include or exclude specific home teams."
    )

    # Away Team Selector
    all_away_teams = sorted(df_attendance['Away Team'].unique().tolist())
    selected_away_teams = st.sidebar.multiselect(
        "Select Away Team(s)",
        options=all_away_teams,
        default=all_away_teams,
        help="Include or exclude specific away teams."
    )

    # City Selector
    all_cities = sorted(df_attendance['City'].unique().tolist())
    selected_cities = st.sidebar.multiselect(
        "Select City(ies)",
        options=all_cities,
        default=all_cities,
        help="Filter games by the city they were played in."
    )

    # State Selector
    all_states = sorted(df_attendance['State'].unique().tolist())
    selected_states = st.sidebar.multiselect(
        "Select State(s)",
        options=all_states,
        default=all_states,
        help="Filter games by the state they were played in."
    )
    
    # Arena Selector
    all_arenas = sorted(df_attendance['Arena'].unique().tolist())
    selected_arenas = st.sidebar.multiselect(
        "Select Arena(s)",
        options=all_arenas,
        default=all_arenas,
        help="Filter games by specific arenas."
    )

    st.sidebar.markdown("---")

    # Attendance Aggregation Level
    aggregation_level = st.sidebar.radio(
        "Aggregate Attendance Trends By",
        ('Daily', 'Monthly', 'Yearly'),
        help="Choose the granularity for the attendance trend line chart."
    )
    
    # Filter the DataFrame based on ALL selections
    filtered_df_attendance = df_attendance[
        (df_attendance['Year'].isin(selected_years_slider)) & # Apply slider filter
        (df_attendance['Game Type'].isin(selected_game_types)) &
        (df_attendance['Home Team'].isin(selected_home_teams)) &
        (df_attendance['Away Team'].isin(selected_away_teams)) &
        (df_attendance['City'].isin(selected_cities)) &
        (df_attendance['State'].isin(selected_states)) &
        (df_attendance['Arena'].isin(selected_arenas))
    ]

    st.markdown("---") # Visual separator

    # --- Main Content Area - Display Graphs ---

    if filtered_df_attendance.empty:
        st.error("No attendance data matches the selected filters. Please adjust your selections in the slider and sidebar.")
    else:
        # --- Key Metrics ---
        st.subheader("📊 Key Attendance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Games", f"{len(filtered_df_attendance):,}")
        with col2:
            st.metric("Average Attendance", f"{filtered_df_attendance['Attendance'].mean():,.0f}")
        with col3:
            st.metric("Max Attendance", f"{filtered_df_attendance['Attendance'].max():,.0f}")
        with col4:
            st.metric("Min Attendance", f"{filtered_df_attendance['Attendance'].min():,.0f}")

        st.markdown("---")

        # --- Attendance Trends ---
        st.subheader(f"📈 Attendance Trends ({aggregation_level} Average)")

        # Aggregate attendance data based on selected level
        if aggregation_level == 'Daily':
            grouped_attendance = filtered_df_attendance.groupby('Date')['Attendance'].mean().reset_index()
            x_axis_col = 'Date'
        elif aggregation_level == 'Monthly':
            grouped_attendance = filtered_df_attendance.groupby('MonthYear')['Attendance'].mean().reset_index()
            grouped_attendance['MonthYear'] = pd.to_datetime(grouped_attendance['MonthYear']) # Convert back for proper sorting
            grouped_attendance = grouped_attendance.sort_values(by='MonthYear')
            x_axis_col = 'MonthYear'
        else: # Yearly
            grouped_attendance = filtered_df_attendance.groupby('Year')['Attendance'].mean().reset_index()
            x_axis_col = 'Year'
        
        grouped_attendance.rename(columns={'Attendance': 'Average Attendance'}, inplace=True)

        fig_line_attendance = px.line(
            grouped_attendance,
            x=x_axis_col,
            y='Average Attendance',
            title=f'Average WNBA Game Attendance Over Time ({aggregation_level})',
            labels={x_axis_col: x_axis_col.replace('MonthYear', 'Month & Year'), 'Average Attendance': 'Average Attendance'},
            hover_data={'Average Attendance': ':.0f'},
            template="plotly_white" # Use a clean white template
        )

        fig_line_attendance.update_xaxes(showgrid=True, gridcolor='lightgray')
        fig_line_attendance.update_yaxes(showgrid=True, gridcolor='lightgray')

        fig_line_attendance.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='black',
            margin=dict(l=20, r=20, t=60, b=20), # Adjust margins for better spacing
            xaxis_title_font_size=14,
            yaxis_title_font_size=14,
            title_font_size=20,
            hovermode="x unified",
            hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial")
        )
        st.plotly_chart(fig_line_attendance, use_container_width=True)

        st.markdown("---")

        # --- Attendance by Home Team ---
        st.subheader("🏟️ Average Attendance by Home Team")
        avg_attendance_by_home_team = filtered_df_attendance.groupby('Home Team')['Attendance'].mean().reset_index()
        avg_attendance_by_home_team = avg_attendance_by_home_team.sort_values(by='Attendance', ascending=False)

        # Generate team color map based on current filtered teams
        current_home_teams = avg_attendance_by_home_team['Home Team'].unique().tolist()
        team_color_map = get_team_color_map(current_home_teams)

        fig_bar_team = px.bar(
            avg_attendance_by_home_team,
            x='Home Team',
            y='Attendance',
            title='Average Attendance per Home Team',
            labels={'Home Team': 'Home Team', 'Attendance': 'Average Attendance'},
            hover_data={'Attendance': ':.0f'},
            color='Home Team', # Color bars by team
            color_discrete_map=team_color_map, # Apply custom color map
            template="plotly_white"
        )
        fig_bar_team.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='black',
            margin=dict(l=20, r=20, t=60, b=20),
            xaxis_title_font_size=14,
            yaxis_title_font_size=14,
            title_font_size=20,
            hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial")
        )
        fig_bar_team.update_xaxes(tickangle=-45, showgrid=False) # Angle labels for readability
        fig_bar_team.update_yaxes(showgrid=True, gridcolor='lightgray')
        st.plotly_chart(fig_bar_team, use_container_width=True)

        st.markdown("---")

        # --- Attendance by Day of Week ---
        st.subheader("🗓️ Average Attendance by Day of Week")
        # Define a consistent order for days of the week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        avg_attendance_by_day = filtered_df_attendance.groupby('DayOfWeek')['Attendance'].mean().reindex(day_order).reset_index()
        
        fig_bar_day = px.bar(
            avg_attendance_by_day,
            x='DayOfWeek',
            y='Attendance',
            title='Average Attendance by Day of the Week',
            labels={'DayOfWeek': 'Day of Week', 'Attendance': 'Average Attendance'},
            hover_data={'Attendance': ':.0f'},
            template="plotly_white",
            color='DayOfWeek', # Color by day of week
            color_discrete_sequence=px.colors.qualitative.Pastel # Use a pastel palette for days
        )
        fig_bar_day.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='black',
            margin=dict(l=20, r=20, t=60, b=20),
            xaxis_title_font_size=14,
            yaxis_title_font_size=14,
            title_font_size=20,
            xaxis={'categoryorder':'array', 'categoryarray':day_order}, # Ensure consistent order
            hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial")
        )
        fig_bar_day.update_xaxes(showgrid=False)
        fig_bar_day.update_yaxes(showgrid=True, gridcolor='lightgray')
        st.plotly_chart(fig_bar_day, use_container_width=True)

        st.markdown("---")

        # --- Attendance by Month ---
        st.subheader("📅 Average Attendance by Month")
        # Define a consistent order for months
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        avg_attendance_by_month = filtered_df_attendance.groupby('MonthName')['Attendance'].mean().reindex(month_order).reset_index()
        
        fig_bar_month = px.bar(
            avg_attendance_by_month,
            x='MonthName',
            y='Attendance',
            title='Average Attendance by Month',
            labels={'MonthName': 'Month', 'Attendance': 'Average Attendance'},
            hover_data={'Attendance': ':.0f'},
            template="plotly_white",
            color='MonthName', # Color by month
            color_discrete_sequence=px.colors.qualitative.Set2 # Use another palette for months
        )
        fig_bar_month.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='black',
            margin=dict(l=20, r=20, t=60, b=20),
            xaxis_title_font_size=14,
            yaxis_title_font_size=14,
            title_font_size=20,
            xaxis={'categoryorder':'array', 'categoryarray':month_order}, # Ensure consistent order
            hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial")
        )
        fig_bar_month.update_xaxes(showgrid=False)
        fig_bar_month.update_yaxes(showgrid=True, gridcolor='lightgray')
        st.plotly_chart(fig_bar_month, use_container_width=True)

        st.markdown("---")

        # --- Detailed Data Section (Collapsible) ---
        with st.expander("Show Detailed Attendance Data 📋"):
            st.dataframe(filtered_df_attendance.style.highlight_max(axis=0, subset=['Attendance'], color='lightblue'))

else:
    st.info("Attendance data (`All Game Attendance.csv`) not found or is empty. Please ensure it's in the correct directory.")
