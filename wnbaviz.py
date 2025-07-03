import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if not df_attendance.empty: 
    # --- DEBUGGING FILE PATH START ---
    st.error("--- DEBUGGING FILE PATH ---")
    current_working_directory = os.getcwd()
    st.error(f"Current working directory (where Python is looking): {current_working_directory}")
    
    files_in_directory = os.listdir(current_working_directory)
    st.error(f"Files found in this directory: {files_in_directory}")
    
    # Check for the specific file
    target_filename_lower = 'trends.csv'.lower()
    found_it = False
    for f_name in files_in_directory:
        if f_name.lower() == target_filename_lower:
            st.success(f"'{f_name}' (trends.csv) **FOUND** in the current directory.")
            found_it = True
            break
    
    if not found_it:
        st.error(f"'{target_filename_lower}' **NOT FOUND** in the current directory. Please ensure it's there and spelled correctly (case-sensitive if your OS requires).")
        st.error("Also, check for hidden file extensions (e.g., 'trends.csv.txt' or 'trends.csv.csv').")
        st.info("You can try renaming your file to something very simple like `my_trends.csv` and updating the script.")

    st.error("--- END DEBUGGING FILE PATH ---")

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="WNBA Attendance & Trends Dashboard", page_icon="🏀")

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

# --- Load Media Data ---
@st.cache_data # Cache data to improve performance
def load_media_data(file_path):
    """
    Loads the media coverage data from a CSV file.
    Uses 'publish_date' for dates and infers 'mentions' by counting rows.
    """
    try:
        df = pd.read_csv(file_path)
        
        # Ensure 'publish_date' is datetime
        if 'publish_date' in df.columns:
            df['publish_date'] = pd.to_datetime(df['publish_date'], errors='coerce')
            df.dropna(subset=['publish_date'], inplace=True)
            df.rename(columns={'publish_date': 'Date'}, inplace=True) # Align column name
        else:
            st.warning("Column 'publish_date' not found in media data. Please ensure the CSV has a 'publish_date' column.")
            return pd.DataFrame()

        # Infer mentions by counting rows, useful if no explicit 'Mentions' column
        # Each row is assumed to be one mention/article
        df['Mentions'] = 1 
        df['MonthYear'] = df['Date'].dt.to_period('M').astype(str)
        df = df.sort_values(by='Date')

        return df
    except FileNotFoundError:
        st.error(f"Error: The media file '{file_path}' was not found. Please make sure it's in the same directory as this script.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred while loading or processing the media data: {e}")
        return pd.DataFrame()

# --- Load Google Search Trend Data ---
@st.cache_data # Cache data to improve performance
def load_search_trends_data(file_path):
    """
    Loads Google Search Trend data from a CSV file.
    Assumes 'Month' column for date and 'WNBA' for search volume.
    Adjusted to skip initial metadata rows common in Google Trends exports.
    """
    try:
        # --- IMPORTANT ADJUSTMENT: Skipping rows ---
        # Google Trends CSVs often have metadata in the first few rows.
        # We assume the actual header is on the 3rd row (index 2).
        df = pd.read_csv(file_path, skiprows=2) 
        
        # Convert 'Month' to datetime
        if 'Month' in df.columns:
            df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
            df.dropna(subset=['Month'], inplace=True)
            df.rename(columns={'Month': 'Date'}, inplace=True) # Align column name to 'Date'
        else:
            st.warning(f"Column 'Month' not found in search trends data from '{file_path}'. Please check its exact name and casing.")
            return pd.DataFrame()

        # Ensure 'WNBA' is numeric
        if 'WNBA' in df.columns:
            df['WNBA'] = pd.to_numeric(df['WNBA'], errors='coerce')
            df.dropna(subset=['WNBA'], inplace=True)
        else:
            st.warning(f"Column 'WNBA' not found in search trends data from '{file_path}'. Please check its exact name and casing.")
            return pd.DataFrame()

        df['MonthYear'] = df['Date'].dt.to_period('M').astype(str)
        df = df.sort_values(by='Date')

        return df
    except FileNotFoundError:
        st.error(f"Error: The search trends file '{file_path}' was not found. Please make sure it's in the same directory as this script.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred while loading or processing the search trends data: {e}. You might need to adjust 'skiprows' or column names.")
        return pd.DataFrame()

attendance_file = "All Game Attendance.csv" 
df_attendance = load_attendance_data(attendance_file)

media_file = "media.csv" 
df_media = load_media_data(media_file)

# --- Corrected filename ---
search_trends_file = "trends.csv" 
df_search_trends = load_search_trends_data(search_trends_file)


# --- Streamlit App Layout ---
if not df_attendance.empty:
    # --- Text Size Slider ---
    st.sidebar.header("Dashboard Settings")
    font_size = st.sidebar.slider("Adjust Global Text Size (px)", min_value=12, max_value=24, value=16, step=1)

    # Inject custom CSS for global text size adjustment
    css = f"""
    <style>
        :root {{
            --global-font-size: {font_size}px;
        }}

        /* Apply font size to common elements */
        body, p, li, table, .stMarkdown, .stText, .stNumberInput, .stSelectbox, .stRadio, .stCheckbox, .stButton, .stTextInput, .stMultiSelect {{
            font-size: var(--global-font-size) !important;
        }}

        h1 {{
            font-size: calc(var(--global-font-size) * 2.2) !important;
        }}
        h2 {{
            font-size: calc(var(--global-font-size) * 1.8) !important;
        }}
        h3 {{
            font-size: calc(var(--global-font-size) * 1.5) !important;
        }}
        h4 {{
            font-size: calc(var(--global-font-size) * 1.2) !important;
        }}
        h5, h6 {{
            font-size: var(--global-font-size) !important;
        }}

        /* Adjust labels and internal text for specific widgets if needed */
        div[data-testid="stSlider"] label p,
        div[data-testid="stTextInput"] label p,
        div[data-testid="stMultiSelect"] label p,
        div[data-testid="stRadio"] label p,
        div[data-testid="stCheckbox"] label p,
        div[data-testid="stSelectbox"] label p {{
            font-size: var(--global-font-size) !important;
        }}

        /* Adjust Streamlit specific elements */
        .stButton > button {{
            font-size: var(--global-font-size) !important;
        }}
        .streamlit-expanderHeader {{
            font-size: var(--global-font-size) !important;
        }}
        .metric-value {{
            font-size: calc(var(--global-font-size) * 1.5) !important; /* For st.metric values */
        }}
        .metric-label {{
            font-size: var(--global-font-size) !important; /* For st.metric labels */
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    st.title("🏀 WNBA Attendance & Trends Dashboard")

    st.markdown("""
        Welcome to the interactive WNBA Attendance and Trends Dashboard! Explore game attendance,
        media coverage, and Google search interest using the filters below and in the sidebar.
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

    # Offseason filter option - True means "smooth line (include offseason)", False means "exclude (gap)"
    filter_offseason_smooth = st.sidebar.checkbox(
        "Show Offseason Trend (Smooth Line)",
        value=True,  # Default is ON, means show smooth line over offseason
        help="Check to show a continuous trend line including offseason months. Uncheck to filter out offseason data and create gaps."
    )

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
    
    # Define offseason months (November to April)
    offseason_months = [11, 12, 1, 2, 3, 4]

    # Filter the DataFrame based on ALL selections
    # --- Start of Team and Offseason Exclusion Filter ---
    teams_to_exclude = ['Team Delle Donne', 'Team WNBA']
    
    # Filter out rows where either Home Team or Away Team is in the exclusion list (always applies)
    initial_filtered_df = df_attendance[
        (~df_attendance['Home Team'].isin(teams_to_exclude)) &
        (~df_attendance['Away Team'].isin(teams_to_exclude))
    ]

    # Apply offseason filter if the checkbox is UNCHECKED (i.e., we want to exclude offseason/create gaps)
    if not filter_offseason_smooth:
        initial_filtered_df = initial_filtered_df[~initial_filtered_df['Date'].dt.month.isin(offseason_months)]
    # --- End of Team and Offseason Exclusion Filter ---

    filtered_df_attendance = initial_filtered_df[
        (initial_filtered_df['Year'].isin(selected_years_slider)) & # Apply slider filter
        (initial_filtered_df['Game Type'].isin(selected_game_types)) &
        (initial_filtered_df['Home Team'].isin(selected_home_teams)) &
        (initial_filtered_df['Away Team'].isin(selected_away_teams)) &
        (initial_filtered_df['City'].isin(selected_cities)) &
        (initial_filtered_df['State'].isin(selected_states)) &
        (initial_filtered_df['Arena'].isin(selected_arenas))
    ]

    st.markdown("---") # Visual separator

    # --- Main Content Area - Display Graphs ---

    if filtered_df_attendance.empty:
        st.error("No attendance data matches the selected filters. Please adjust your selections in the slider and sidebar.")
    else:
        # Define line shape based on checkbox state
        line_shape_value = 'spline' if filter_offseason_smooth else 'linear'

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

        # --- Attendance Trends --- (FIRST GRAPH)
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

        fig_line_attendance.update_traces(mode='lines+markers', line=dict(shape=line_shape_value)) # Apply shape here
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
        
        # --- Google Search Trends Overlay (NEW POSITION: UNDER FIRST GRAPH) ---
        st.subheader("📊 Attendance vs. Google Search Trends (Monthly Trends)")

        if not df_search_trends.empty:
            # Filter search trends data by selected years
            if 'Date' in df_search_trends.columns:
                filtered_df_search_trends = df_search_trends[df_search_trends['Date'].dt.year.isin(selected_years_slider)]
                # Apply offseason filter to search trends data if checkbox is UNCHECKED
                if not filter_offseason_smooth:
                    filtered_df_search_trends = filtered_df_search_trends[~filtered_df_search_trends['Date'].dt.month.isin(offseason_months)]
            else:
                filtered_df_search_trends = pd.DataFrame() # No 'Date' column in search trends, so no filter

            if not filtered_df_search_trends.empty:
                # Aggregate attendance by MonthYear (using appropriate filtered data for smooth/gap logic)
                if filter_offseason_smooth: # If smooth line is on, use initial_filtered_df for aggregation
                    monthly_avg_attendance_for_overlay = initial_filtered_df.groupby('MonthYear')['Attendance'].mean().reset_index()
                else: # If smooth line is off (gaps), use already filtered_df_attendance
                     monthly_avg_attendance_for_overlay = filtered_df_attendance.groupby('MonthYear')['Attendance'].mean().reset_index()
                
                monthly_avg_attendance_for_overlay['MonthYear'] = pd.to_datetime(monthly_avg_attendance_for_overlay['MonthYear'])
                monthly_avg_attendance_for_overlay = monthly_avg_attendance_for_overlay.sort_values(by='MonthYear')
                monthly_avg_attendance_for_overlay.rename(columns={'Attendance': 'Average Attendance'}, inplace=True)

                # Aggregate search trends by MonthYear (average 'WNBA' searches for the month)
                monthly_search_trends = filtered_df_search_trends.groupby('MonthYear')['WNBA'].mean().reset_index()
                monthly_search_trends['MonthYear'] = pd.to_datetime(monthly_search_trends['MonthYear'])
                monthly_search_trends = monthly_search_trends.sort_values(by='MonthYear')
                monthly_search_trends.rename(columns={'WNBA': 'Average WNBA Searches'}, inplace=True)

                # Merge the two aggregated dataframes
                combined_df_search = pd.merge(monthly_avg_attendance_for_overlay, monthly_search_trends, on='MonthYear', how='outer')
                combined_df_search = combined_df_search.sort_values(by='MonthYear')

                if not combined_df_search.empty:
                    fig_search_overlay = make_subplots(specs=[[{"secondary_y": True}]])

                    # Add Attendance trace
                    fig_search_overlay.add_trace(
                        go.Scatter(
                            x=combined_df_search['MonthYear'], 
                            y=combined_df_search['Average Attendance'], 
                            name='Average Attendance', 
                            mode='lines+markers', 
                            line=dict(color='blue', width=2, shape=line_shape_value),
                            hovertemplate='<b>Month:</b> %{x|%b %Y}<br><b>Avg. Attendance:</b> %{y:,.0f}<extra></extra>'
                        ),
                        secondary_y=False,
                    )

                    # Add Google Search Trends trace
                    fig_search_overlay.add_trace(
                        go.Scatter(
                            x=combined_df_search['MonthYear'], 
                            y=combined_df_search['Average WNBA Searches'], 
                            name='Average WNBA Searches', 
                            mode='lines+markers', 
                            line=dict(color='green', width=2, shape=line_shape_value),
                            hovertemplate='<b>Month:</b> %{x|%b %Y}<br><b>Avg. Searches:</b> %{y:,.0f}<extra></extra>'
                        ),
                        secondary_y=True,
                    )

                    # Add titles and labels
                    fig_search_overlay.update_layout(
                        title_text='Average WNBA Attendance vs. Google Search Trends Over Time',
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='black',
                        margin=dict(l=20, r=20, t=60, b=20),
                        xaxis_title_font_size=14,
                        yaxis_title_font_size=14,
                        title_font_size=20,
                        hovermode="x unified",
                        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
                        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)', bordercolor='lightgray', borderwidth=1)
                    )

                    fig_search_overlay.update_xaxes(
                        title_text="Month & Year", 
                        showgrid=True, 
                        gridcolor='lightgray',
                        tickformat="%b %Y" # Format date ticks for month and year
                    )
                    fig_search_overlay.update_yaxes(
                        title_text="Average Attendance", 
                        secondary_y=False, 
                        showgrid=True, 
                        gridcolor='lightgray',
                        tickformat=",0f" # Format attendance with commas
                    )
                    fig_search_overlay.update_yaxes(
                        title_text="Average WNBA Searches", 
                        secondary_y=True, 
                        showgrid=False, # Disable grid for secondary y-axis to avoid clutter
                        tickformat=",0f" # Format mentions with commas
                    )

                    st.plotly_chart(fig_search_overlay, use_container_width=True)
                else:
                    st.info("No combined attendance and search trends data available for the selected filters (after merging).")
            else:
                st.info("No Google Search Trends data matches the selected year or offseason filters. Please adjust your selections.")
        else:
            st.info("Google Search Trends data (`trends.csv`) not found or is empty. Please ensure it's in the correct directory, and has 'Month' and 'WNBA' columns.")

        st.markdown("---")

        # --- Overlay: Attendance vs. Media Coverage (Monthly) ---
        st.subheader("📈 Attendance vs. Media Coverage (Monthly Trends)")

        if not df_media.empty:
            # Filter media data to only include years also present in filtered attendance data
            # This aligns the time range for comparison
            # Ensure 'Date' column exists in df_media before filtering by year
            if 'Date' in df_media.columns:
                filtered_df_media = df_media[df_media['Date'].dt.year.isin(selected_years_slider)]
                # Apply offseason filter to media data if checkbox is UNCHECKED
                if not filter_offseason_smooth:
                    filtered_df_media = filtered_df_media[~filtered_df_media['Date'].dt.month.isin(offseason_months)]
            else:
                filtered_df_media = pd.DataFrame() # No 'Date' column in media, so no media filter

            if not filtered_df_media.empty:
                # Aggregate attendance by MonthYear (using appropriate filtered data for smooth/gap logic)
                if filter_offseason_smooth: # If smooth line is on, use initial_filtered_df for aggregation
                    monthly_avg_attendance = initial_filtered_df.groupby('MonthYear')['Attendance'].mean().reset_index()
                else: # If smooth line is off (gaps), use already filtered_df_attendance
                     monthly_avg_attendance = filtered_df_attendance.groupby('MonthYear')['Attendance'].mean().reset_index()


                monthly_avg_attendance['MonthYear'] = pd.to_datetime(monthly_avg_attendance['MonthYear']) # Convert to datetime for proper sorting
                monthly_avg_attendance = monthly_avg_attendance.sort_values(by='MonthYear')
                monthly_avg_attendance.rename(columns={'Attendance': 'Average Attendance'}, inplace=True)

                # Aggregate media mentions by MonthYear (sum of mentions for the month)
                monthly_media_mentions = filtered_df_media.groupby('MonthYear')['Mentions'].sum().reset_index()
                monthly_media_mentions['MonthYear'] = pd.to_datetime(monthly_media_mentions['MonthYear']) # Convert to datetime for proper sorting
                monthly_media_mentions = monthly_media_mentions.sort_values(by='MonthYear')
                monthly_media_mentions.rename(columns={'Mentions': 'Total Media Mentions'}, inplace=True)

                # Merge the two aggregated dataframes
                # Use 'outer' merge to keep all months from both datasets
                combined_df = pd.merge(monthly_avg_attendance, monthly_media_mentions, on='MonthYear', how='outer')
                combined_df = combined_df.sort_values(by='MonthYear')

                if not combined_df.empty:
                    fig_overlay = make_subplots(specs=[[{"secondary_y": True}]])

                    # Add Attendance trace
                    fig_overlay.add_trace(
                        go.Scatter(
                            x=combined_df['MonthYear'], 
                            y=combined_df['Average Attendance'], 
                            name='Average Attendance', 
                            mode='lines+markers', 
                            line=dict(color='blue', width=2, shape=line_shape_value), # Apply shape here
                            hovertemplate='<b>Month:</b> %{x|%b %Y}<br><b>Avg. Attendance:</b> %{y:,.0f}<extra></extra>'
                        ),
                        secondary_y=False,
                    )

                    # Add Media Mentions trace
                    fig_overlay.add_trace(
                        go.Scatter(
                            x=combined_df['MonthYear'], 
                            y=combined_df['Total Media Mentions'], 
                            name='Total Media Mentions', 
                            mode='lines+markers', 
                            line=dict(color='red', width=2, shape=line_shape_value), # Apply shape here
                            hovertemplate='<b>Month:</b> %{x|%b %Y}<br><b>Total Mentions:</b> %{y:,.0f}<extra></extra>'
                        ),
                        secondary_y=True,
                    )

                    # Add titles and labels
                    fig_overlay.update_layout(
                        title_text='Average WNBA Attendance vs. Total Media Mentions Over Time',
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='black',
                        margin=dict(l=20, r=20, t=60, b=20),
                        xaxis_title_font_size=14,
                        yaxis_title_font_size=14,
                        title_font_size=20,
                        hovermode="x unified",
                        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
                        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)', bordercolor='lightgray', borderwidth=1)
                    )

                    fig_overlay.update_xaxes(
                        title_text="Month & Year", 
                        showgrid=True, 
                        gridcolor='lightgray',
                        tickformat="%b %Y" # Format date ticks for month and year
                    )
                    fig_overlay.update_yaxes(
                        title_text="Average Attendance", 
                        secondary_y=False, 
                        showgrid=True, 
                        gridcolor='lightgray',
                        tickformat=",0f" # Format attendance with commas
                    )
                    fig_overlay.update_yaxes(
                        title_text="Total Media Mentions", 
                        secondary_y=True, 
                        showgrid=False, # Disable grid for secondary y-axis to avoid clutter
                        tickformat=",0f" # Format mentions with commas
                    )

                    st.plotly_chart(fig_overlay, use_container_width=True)
                else:
                    st.info("No combined attendance and media data available for the selected filters (after merging).")
            else:
                st.info("No media coverage data matches the selected year or offseason filters. Please adjust your selections.")
        else:
            st.info("Media coverage data (`media.csv`) not found or is empty. Please ensure it's in the correct directory, and has a 'publish_date' column.")

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
        
        # Determine the DataFrame to use for monthly aggregation based on filter_offseason_smooth
        if filter_offseason_smooth:
            # Use initial_filtered_df (which retains offseason data if smooth line is desired)
            avg_attendance_by_month = initial_filtered_df.groupby('MonthName')['Attendance'].mean().reset_index()
        else:
            # Use filtered_df_attendance (which has offseason filtered out if gaps are desired)
            avg_attendance_by_month = filtered_df_attendance.groupby('MonthName')['Attendance'].mean().reset_index()

        # Ensure only relevant months appear in the chart, and in correct order
        avg_attendance_by_month['MonthName'] = pd.Categorical(avg_attendance_by_month['MonthName'], categories=month_order, ordered=True)
        avg_attendance_by_month = avg_attendance_by_month.sort_values('MonthName')
        
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

        with st.expander("Show Detailed Media Data 📰"):
            # Display original media data with relevant columns for user review
            # Check if 'Date' column exists before trying to display it
            display_media_df = df_media[['id', 'Date', 'title', 'media_name', 'url']] if 'Date' in df_media.columns else df_media[['id', 'title', 'media_name', 'url']]
            st.dataframe(display_media_df)
        
        with st.expander("Show Detailed Google Search Trends Data 🔎"):
            # Display original search trends data with relevant columns for user review
            display_search_df = df_search_trends[['Date', 'WNBA']] if 'Date' in df_search_trends.columns else df_search_trends[['WNBA']]
            st.dataframe(display_search_df)

else:
    st.info("Attendance data (`All Game Attendance.csv`) not found or is empty. Please ensure it's in the correct directory.")

    if df_media.empty:
        st.info("Media coverage data (`media.csv`) also not found or is empty.")
    if df_search_trends.empty:
        st.info("Google Search Trends data (`trends.csv`) also not found or is empty.")
