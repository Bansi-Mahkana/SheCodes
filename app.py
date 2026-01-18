import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="CityBAND - Health Intelligence Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .risk-high {
        background-color: #fee;
        border-left: 5px solid #e74c3c;
        padding: 10px;
        border-radius: 5px;
    }
    .risk-medium {
        background-color: #fef9e7;
        border-left: 5px solid #f39c12;
        padding: 10px;
        border-radius: 5px;
    }
    .risk-low {
        background-color: #eafaf1;
        border-left: 5px solid #27ae60;
        padding: 10px;
        border-radius: 5px;
    }
    h1 {
        color: #2c3e50;
    }
    h2 {
        color: #34495e;
    }
    </style>
""", unsafe_allow_html=True)

# Database connection
@st.cache_resource
def get_connection():
    return sqlite3.connect('database/final.db', check_same_thread=False)

# Initialize database with sample data
def init_database():
    conn = get_connection()
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS traffic_data (
            date TEXT,
            city TEXT,
            vehicle_count INTEGER,
            avg_speed REAL,
            congestion_level TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS weather_data (
            date TEXT,
            city TEXT,
            temperature REAL,
            humidity REAL,
            rainfall REAL,
            aqi INTEGER,
            heat_index REAL,
            air_quality_category TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS health_data (
            date TEXT,
            city TEXT,
            total_hospital_visits INTEGER,
            respiratory_cases INTEGER,
            heat_related_cases INTEGER
        )
    ''')
    
    # Check if data exists
    cursor.execute("SELECT COUNT(*) FROM health_data")
    if cursor.fetchone()[0] == 0:
        # Generate sample data
        cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune', 'Ahmedabad']
        dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
        
        for city in cities:
            for date in dates:
                date_str = date.strftime('%Y-%m-%d')
                
                # Traffic data
                vehicle_count = np.random.randint(50000, 200000)
                avg_speed = np.random.uniform(20, 60)
                congestion = 'High' if avg_speed < 30 else 'Medium' if avg_speed < 45 else 'Low'
                cursor.execute(
                    "INSERT INTO traffic_data VALUES (?, ?, ?, ?, ?)",
                    (date_str, city, vehicle_count, avg_speed, congestion)
                )
                
                # Weather data
                temp = np.random.uniform(25, 42)
                humidity = np.random.uniform(40, 90)
                rainfall = np.random.uniform(0, 50) if np.random.random() > 0.7 else 0
                aqi = np.random.randint(50, 350)
                heat_index = temp + 0.5 * humidity / 10
                air_quality = 'Good' if aqi < 100 else 'Moderate' if aqi < 200 else 'Poor' if aqi < 300 else 'Hazardous'
                cursor.execute(
                    "INSERT INTO weather_data VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (date_str, city, temp, humidity, rainfall, aqi, heat_index, air_quality)
                )
                
                # Health data
                base_visits = np.random.randint(500, 2000)
                resp_cases = int(base_visits * np.random.uniform(0.1, 0.3) * (aqi / 100))
                heat_cases = int(base_visits * np.random.uniform(0.05, 0.15) * (temp / 30))
                cursor.execute(
                    "INSERT INTO health_data VALUES (?, ?, ?, ?, ?)",
                    (date_str, city, base_visits, resp_cases, heat_cases)
                )
        
        conn.commit()
    
    return conn

# Load data
@st.cache_data(ttl=300)
def load_data():
    conn = get_connection()
    
    query = '''
        SELECT 
            h.date,
            h.city,
            t.vehicle_count,
            t.avg_speed,
            t.congestion_level,
            w.temperature,
            w.heat_index,
            w.humidity,
            w.rainfall,
            w.aqi,
            w.air_quality_category,
            h.respiratory_cases,
            h.heat_related_cases,
            h.total_hospital_visits
        FROM health_data h
        LEFT JOIN traffic_data t ON h.date = t.date AND h.city = t.city
        LEFT JOIN weather_data w ON h.date = w.date AND h.city = w.city
    '''
    
    df = pd.read_sql_query(query, conn)
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate health risk score
    df['health_risk_score'] = calculate_risk_score(df)
    df['risk_category'] = df['health_risk_score'].apply(categorize_risk)
    
    return df

def calculate_risk_score(df):
    # Normalize factors (0-100 scale)
    aqi_norm = np.clip(df['aqi'] / 3, 0, 100)
    temp_norm = np.clip((df['temperature'] - 25) * 2, 0, 100)
    resp_norm = np.clip(df['respiratory_cases'] / df['respiratory_cases'].max() * 100, 0, 100)
    heat_norm = np.clip(df['heat_related_cases'] / df['heat_related_cases'].max() * 100, 0, 100)
    
    # Weighted risk score
    risk_score = (
        0.35 * aqi_norm +
        0.25 * temp_norm +
        0.25 * resp_norm +
        0.15 * heat_norm
    )
    
    return risk_score

def categorize_risk(score):
    if score < 40:
        return 'Low'
    elif score < 70:
        return 'Medium'
    else:
        return 'High'

# Initialize database
conn = init_database()

# Sidebar
# st.sidebar.image("https://img.icons8.com/fluency/96/000000/health-graph.png", width=80)
st.sidebar.title("🏥 CityBAND")
st.sidebar.markdown("### Data-Driven Health Analytics")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "📊 City Analysis", "🗺️ Geographic Insights", "🔮 What-If Scenarios", "⚠️ Early Warning"]
)

# Load data
df = load_data()

# Date filter
st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
date_range = st.sidebar.date_input(
    "Date Range",
    value=(df['date'].min(), df['date'].max()),
    min_value=df['date'].min(),
    max_value=df['date'].max()
)

if len(date_range) == 2:
    df_filtered = df[(df['date'] >= pd.to_datetime(date_range[0])) & 
                     (df['date'] <= pd.to_datetime(date_range[1]))]
else:
    df_filtered = df

# City filter
cities = st.sidebar.multiselect(
    "Select Cities",
    options=sorted(df['city'].unique()),
    default=sorted(df['city'].unique())
)

if cities:
    df_filtered = df_filtered[df_filtered['city'].isin(cities)]

# PAGE 1: OVERVIEW
if page == "🏠 Overview":
    st.title("🏥 CityBAND - Public Health Intelligence Dashboard")
    st.markdown("### Real-time Urban Health Risk Monitoring & Analytics")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    latest_data = df_filtered.groupby('city').tail(1)
    
    with col1:
        avg_risk = latest_data['health_risk_score'].mean()
        st.metric(
            "Average Risk Score",
            f"{avg_risk:.1f}",
            delta=f"{avg_risk - df.groupby('city').tail(7)['health_risk_score'].mean():.1f}",
            delta_color="inverse"
        )
    
    with col2:
        high_risk_cities = (latest_data['risk_category'] == 'High').sum()
        st.metric(
            "High Risk Cities",
            high_risk_cities,
            delta=f"{high_risk_cities - (df.groupby('city').tail(7)['risk_category'] == 'High').sum()}"
        )
    
    with col3:
        avg_aqi = latest_data['aqi'].mean()
        st.metric(
            "Average AQI",
            f"{avg_aqi:.0f}",
            delta=f"{avg_aqi - df.groupby('city').tail(7)['aqi'].mean():.0f}",
            delta_color="inverse"
        )
    
    with col4:
        total_cases = latest_data['respiratory_cases'].sum() + latest_data['heat_related_cases'].sum()
        st.metric(
            "Total Health Cases",
            f"{total_cases:,}",
            delta=f"{total_cases - (df.groupby('city').tail(7)['respiratory_cases'].sum() + df.groupby('city').tail(7)['heat_related_cases'].sum()):,}",
            delta_color="inverse"
        )
    
    st.markdown("---")
    
    # Risk Distribution
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Risk Score Trends")
        
        daily_risk = df_filtered.groupby(['date', 'city'])['health_risk_score'].mean().reset_index()
        
        fig = px.line(
            daily_risk,
            x='date',
            y='health_risk_score',
            color='city',
            title='Health Risk Score Over Time',
            labels={'health_risk_score': 'Risk Score', 'date': 'Date'}
        )
        fig.add_hline(y=40, line_dash="dash", line_color="green", annotation_text="Low Risk Threshold")
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="High Risk Threshold")
        fig.update_layout(height=400, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Risk Distribution")
        
        risk_counts = latest_data['risk_category'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            hole=.4,
            marker_colors=['#27ae60', '#f39c12', '#e74c3c']
        )])
        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    # City Rankings
    st.subheader("🏙️ City Health Risk Rankings")
    
    city_summary = latest_data[['city', 'health_risk_score', 'risk_category', 'aqi', 
                                  'temperature', 'respiratory_cases', 'heat_related_cases']].copy()
    city_summary = city_summary.sort_values('health_risk_score', ascending=False)
    city_summary['health_risk_score'] = city_summary['health_risk_score'].round(2)
    city_summary.columns = ['City', 'Risk Score', 'Risk Level', 'AQI', 'Temperature', 
                             'Respiratory Cases', 'Heat Cases']
    
    st.dataframe(city_summary, hide_index=True, use_container_width=True)

# PAGE 2: CITY ANALYSIS
elif page == "📊 City Analysis":
    st.title("📊 Detailed City Analysis")
    
    selected_city = st.selectbox("Select City for Analysis", sorted(df_filtered['city'].unique()))
    
    city_data = df_filtered[df_filtered['city'] == selected_city].sort_values('date')
    
    # City metrics
    latest = city_data.iloc[-1]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Risk Score", f"{latest['health_risk_score']:.1f}")
    with col2:
        st.metric("Risk Level", latest['risk_category'])
    with col3:
        st.metric("AQI", f"{latest['aqi']:.0f}")
    with col4:
        st.metric("Temperature", f"{latest['temperature']:.1f}°C")
    with col5:
        st.metric("Hospital Visits", f"{latest['total_hospital_visits']:,}")
    
    st.markdown("---")
    
    # Time series analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Environmental Factors")
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Air Quality Index', 'Temperature & Heat Index'),
            vertical_spacing=0.15
        )
        
        fig.add_trace(
            go.Scatter(x=city_data['date'], y=city_data['aqi'], name='AQI', 
                      line=dict(color='#e74c3c')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=city_data['date'], y=city_data['temperature'], 
                      name='Temperature', line=dict(color='#f39c12')),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=city_data['date'], y=city_data['heat_index'], 
                      name='Heat Index', line=dict(color='#e67e22', dash='dash')),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="AQI", row=1, col=1)
        fig.update_yaxes(title_text="Temperature (°C)", row=2, col=1)
        fig.update_layout(height=500, showlegend=True, hovermode='x unified')
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Health Indicators")
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Health Cases', 'Hospital Visits'),
            vertical_spacing=0.15
        )
        
        fig.add_trace(
            go.Bar(x=city_data['date'], y=city_data['respiratory_cases'], 
                   name='Respiratory Cases', marker_color='#3498db'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=city_data['date'], y=city_data['heat_related_cases'], 
                   name='Heat Cases', marker_color='#e74c3c'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=city_data['date'], y=city_data['total_hospital_visits'], 
                      name='Total Visits', line=dict(color='#9b59b6')),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Cases", row=1, col=1)
        fig.update_yaxes(title_text="Visits", row=2, col=1)
        fig.update_layout(height=500, showlegend=True, barmode='stack', hovermode='x unified')
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.subheader("📊 Factor Correlation Analysis")
    
    correlation_data = city_data[['aqi', 'temperature', 'humidity', 'respiratory_cases', 
                                    'heat_related_cases', 'health_risk_score']].corr()
    
    fig = px.imshow(
        correlation_data,
        text_auto='.2f',
        aspect='auto',
        color_continuous_scale='RdYlGn_r',
        title=f'Correlation Matrix - {selected_city}'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# PAGE 3: GEOGRAPHIC INSIGHTS
elif page == "🗺️ Geographic Insights":
    st.title("🗺️ Geographic Health Risk Distribution")
    
    # City coordinates (sample data)
    city_coords = {
        'Mumbai': {'lat': 19.0760, 'lon': 72.8777},
        'Delhi': {'lat': 28.7041, 'lon': 77.1025},
        'Bangalore': {'lat': 12.9716, 'lon': 77.5946},
        'Chennai': {'lat': 13.0827, 'lon': 80.2707},
        'Kolkata': {'lat': 22.5726, 'lon': 88.3639},
        'Hyderabad': {'lat': 17.3850, 'lon': 78.4867},
        'Pune': {'lat': 18.5204, 'lon': 73.8567},
        'Ahmedabad': {'lat': 23.0225, 'lon': 72.5714}
    }
    
    latest_data = df_filtered.groupby('city').tail(1).copy()
    latest_data['lat'] = latest_data['city'].map(lambda x: city_coords.get(x, {}).get('lat', 0))
    latest_data['lon'] = latest_data['city'].map(lambda x: city_coords.get(x, {}).get('lon', 0))
    
    # Map visualization
    st.subheader("🗺️ Interactive Risk Map")
    
    fig = px.scatter_mapbox(
        latest_data,
        lat='lat',
        lon='lon',
        size='health_risk_score',
        color='risk_category',
        hover_name='city',
        hover_data={
            'health_risk_score': ':.2f',
            'aqi': ':.0f',
            'temperature': ':.1f',
            'respiratory_cases': ':,',
            'lat': False,
            'lon': False
        },
        color_discrete_map={'Low': '#27ae60', 'Medium': '#f39c12', 'High': '#e74c3c'},
        zoom=4,
        height=600,
        title='City Health Risk Distribution'
    )
    
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)
    
    # Regional comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("City Comparison - AQI vs Risk Score")
        
        fig = px.scatter(
            latest_data,
            x='aqi',
            y='health_risk_score',
            size='total_hospital_visits',
            color='risk_category',
            hover_name='city',
            color_discrete_map={'Low': '#27ae60', 'Medium': '#f39c12', 'High': '#e74c3c'},
            title='AQI Impact on Health Risk'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Temperature vs Heat Cases")
        
        fig = px.scatter(
            latest_data,
            x='temperature',
            y='heat_related_cases',
            size='health_risk_score',
            color='risk_category',
            hover_name='city',
            color_discrete_map={'Low': '#27ae60', 'Medium': '#f39c12', 'High': '#e74c3c'},
            title='Temperature Impact on Heat Cases'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# PAGE 4: WHAT-IF SCENARIOS
elif page == "🔮 What-If Scenarios":
    st.title("🔮 What-If Scenario Analysis")
    st.markdown("### Simulate changes in environmental and health parameters")
    
    selected_city = st.selectbox("Select City for Simulation", sorted(df_filtered['city'].unique()))
    
    city_data = df_filtered[df_filtered['city'] == selected_city]
    baseline = city_data.iloc[-1].copy()
    
    st.markdown("---")
    st.subheader("🎛️ Adjust Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        aqi_change = st.slider(
            "AQI Change (%)",
            min_value=-50,
            max_value=100,
            value=0,
            step=5,
            help="Simulate AQI increase or decrease"
        )
        
        temp_change = st.slider(
            "Temperature Change (°C)",
            min_value=-10.0,
            max_value=10.0,
            value=0.0,
            step=0.5,
            help="Simulate temperature change"
        )
    
    with col2:
        resp_change = st.slider(
            "Respiratory Cases Change (%)",
            min_value=-50,
            max_value=100,
            value=0,
            step=5,
            help="Simulate change in respiratory cases"
        )
        
        heat_change = st.slider(
            "Heat Cases Change (%)",
            min_value=-50,
            max_value=100,
            value=0,
            step=5,
            help="Simulate change in heat-related cases"
        )
    
    with col3:
        traffic_change = st.slider(
            "Vehicle Count Change (%)",
            min_value=-50,
            max_value=100,
            value=0,
            step=5,
            help="Simulate traffic volume change"
        )
        
        humidity_change = st.slider(
            "Humidity Change (%)",
            min_value=-30,
            max_value=30,
            value=0,
            step=5,
            help="Simulate humidity change"
        )
    
    # Calculate scenario
    scenario = baseline.copy()
    scenario['aqi'] = baseline['aqi'] * (1 + aqi_change / 100)
    scenario['temperature'] = baseline['temperature'] + temp_change
    scenario['respiratory_cases'] = baseline['respiratory_cases'] * (1 + resp_change / 100)
    scenario['heat_related_cases'] = baseline['heat_related_cases'] * (1 + heat_change / 100)
    scenario['vehicle_count'] = baseline['vehicle_count'] * (1 + traffic_change / 100)
    scenario['humidity'] = np.clip(baseline['humidity'] + humidity_change, 0, 100)
    
    # Recalculate risk score
    scenario_df = pd.DataFrame([scenario])
    scenario['health_risk_score'] = calculate_risk_score(scenario_df).iloc[0]
    scenario['risk_category'] = categorize_risk(scenario['health_risk_score'])
    
    st.markdown("---")
    st.subheader("📊 Scenario Results")
    
    # Comparison metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        risk_delta = scenario['health_risk_score'] - baseline['health_risk_score']
        st.metric(
            "Risk Score",
            f"{scenario['health_risk_score']:.1f}",
            delta=f"{risk_delta:.1f}",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            "Risk Category",
            scenario['risk_category'],
            delta=None if scenario['risk_category'] == baseline['risk_category'] else "Changed"
        )
    
    with col3:
        st.metric(
            "AQI",
            f"{scenario['aqi']:.0f}",
            delta=f"{scenario['aqi'] - baseline['aqi']:.0f}",
            delta_color="inverse"
        )
    
    with col4:
        total_cases = scenario['respiratory_cases'] + scenario['heat_related_cases']
        baseline_cases = baseline['respiratory_cases'] + baseline['heat_related_cases']
        st.metric(
            "Total Cases",
            f"{total_cases:.0f}",
            delta=f"{total_cases - baseline_cases:.0f}",
            delta_color="inverse"
        )
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Parameter Comparison")
        
        comparison_data = pd.DataFrame({
            'Parameter': ['AQI', 'Temperature', 'Respiratory Cases', 'Heat Cases', 'Vehicle Count'],
            'Baseline': [
                baseline['aqi'],
                baseline['temperature'],
                baseline['respiratory_cases'],
                baseline['heat_related_cases'],
                baseline['vehicle_count']
            ],
            'Scenario': [
                scenario['aqi'],
                scenario['temperature'],
                scenario['respiratory_cases'],
                scenario['heat_related_cases'],
                scenario['vehicle_count']
            ]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Baseline', x=comparison_data['Parameter'], 
                            y=comparison_data['Baseline'], marker_color='#3498db'))
        fig.add_trace(go.Bar(name='Scenario', x=comparison_data['Parameter'], 
                            y=comparison_data['Scenario'], marker_color='#e74c3c'))
        fig.update_layout(barmode='group', height=400, title='Baseline vs Scenario')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Risk Score Impact")
        
        risk_data = pd.DataFrame({
            'Scenario': ['Baseline', 'Simulated'],
            'Risk Score': [baseline['health_risk_score'], scenario['health_risk_score']],
            'Category': [baseline['risk_category'], scenario['risk_category']]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=risk_data['Scenario'],
            y=risk_data['Risk Score'],
            text=risk_data['Category'],
            textposition='auto',
            marker_color=['#3498db', '#e74c3c']
        ))
        fig.add_hline(y=40, line_dash="dash", line_color="green")
        fig.add_hline(y=70, line_dash="dash", line_color="red")
        fig.update_layout(height=400, title='Health Risk Score Comparison')
        st.plotly_chart(fig, use_container_width=True)
    
    # Insights
    st.markdown("---")
    st.subheader("💡 Scenario Insights")
    
    if risk_delta > 10:
        st.error(f"⚠️ **High Risk Increase**: The simulated scenario shows a significant risk increase of {risk_delta:.1f} points. Immediate intervention may be required.")
    elif risk_delta > 0:
        st.warning(f"⚡ **Moderate Risk Increase**: Risk score increased by {risk_delta:.1f} points. Monitor the situation closely.")
    elif risk_delta < -10:
        st.success(f"✅ **Significant Improvement**: Risk score decreased by {abs(risk_delta):.1f} points. Positive impact expected.")
    else:
        st.info(f"📊 **Minimal Change**: Risk score changed by {risk_delta:.1f} points. Stable conditions.")

# PAGE 5: EARLY WARNING
elif page == "⚠️ Early Warning":
    st.title("⚠️ Early Warning System")
    st.markdown("### Real-time alerts and predictive indicators")
    
    # Alert thresholds
    latest_data = df_filtered.groupby('city').tail(1)
    
    # Generate alerts
    alerts = []
    
    for _, row in latest_data.iterrows():
        if row['health_risk_score'] > 70:
            alerts.append({
                'severity': 'High',
                'city': row['city'],
                'message': f"High health risk detected (Score: {row['health_risk_score']:.1f})",
                'type': 'Risk Score'
            })
        
        if row['aqi'] > 250:
            alerts.append({
                'severity': 'High',
                'city': row['city'],
                'message': f"Hazardous air quality (AQI: {row['aqi']:.0f})",
                'type': 'Air Quality'
            })
        elif row['aqi'] > 150:
            alerts.append({
                'severity': 'Medium',
                'city': row['city'],
                'message': f"Unhealthy air quality (AQI: {row['aqi']:.0f})",
                'type': 'Air Quality'
            })
        
        if row['temperature'] > 38:
            alerts.append({
                'severity': 'High',
                'city': row['city'],
                'message': f"Extreme heat warning (Temperature: {row['temperature']:.1f}°C)",
                'type': 'Temperature'
            })
        
        if row['respiratory_cases'] > latest_data['respiratory_cases'].quantile(0.75):
            alerts.append({
                'severity': 'Medium',
                'city': row['city'],
                'message': f"Elevated respiratory cases ({row['respiratory_cases']:.0f})",
                'type': 'Health'
            })
    
    alerts_df = pd.DataFrame(alerts)
    
    # Display alerts
    if not alerts_df.empty:
        st.subheader(f"🚨 Active Alerts ({len(alerts_df)})")
        
        # Filter by severity
        severity_filter = st.multiselect(
            "Filter by Severity",
            options=['High', 'Medium'],
            default=['High', 'Medium']
        )
        
        filtered_alerts = alerts_df[alerts_df['severity'].isin(severity_filter)]
        
        for _, alert in filtered_alerts.iterrows():
            if alert['severity'] == 'High':
                st.markdown(f"""
                <div class="risk-high">
                    <strong>🔴 {alert['city']}</strong> - {alert['type']}<br>
                    {alert['message']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-medium">
                    <strong>🟡 {alert['city']}</strong> - {alert['type']}<br>
                    {alert['message']}
                </div>
                """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.success("✅ No active alerts. All cities within normal parameters.")
    
    st.markdown("---")
    
    # Trend analysis for predictions
    st.subheader("📈 Predictive Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 7-day moving average
        st.markdown("**7-Day Risk Trend**")
        
        trend_data = df_filtered.groupby(['date', 'city'])['health_risk_score'].mean().reset_index()
        trend_data['ma7'] = trend_data.groupby('city')['health_risk_score'].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )
        
        fig = px.line(
            trend_data,
            x='date',
            y='ma7',
            color='city',
            title='7-Day Moving Average - Risk Score'
        )
        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Threshold violations
        st.markdown("**Threshold Violations (Last 7 Days)**")
        
        recent_data = df_filtered[df_filtered['date'] >= df_filtered['date'].max() - timedelta(days=7)]
        
        violations = recent_data.groupby('city').apply(
            lambda x: pd.Series({
                'High AQI Days': (x['aqi'] > 200).sum(),
                'High Temp Days': (x['temperature'] > 35).sum(),
                'High Risk Days': (x['health_risk_score'] > 70).sum()
            })
        ).reset_index()
        
        fig = px.bar(
            violations.melt(id_vars='city', var_name='Metric', value_name='Days'),
            x='city',
            y='Days',
            color='Metric',
            barmode='group',
            title='Recent Threshold Violations'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Recommended actions
    st.markdown("---")
    st.subheader("📋 Recommended Actions")
    
    high_risk_cities = latest_data[latest_data['risk_category'] == 'High']['city'].tolist()
    
    if high_risk_cities:
        st.warning(f"**Cities Requiring Immediate Attention**: {', '.join(high_risk_cities)}")
        
        st.markdown("""
        **Recommended Interventions:**
        - 🏥 Increase hospital preparedness and staff allocation
        - 📢 Issue public health advisories for vulnerable populations
        - 🚗 Implement traffic restrictions to reduce air pollution
        - 🌡️ Activate cooling centers for heat-related emergencies
        - 💊 Ensure adequate supply of respiratory medications
        - 📊 Increase monitoring frequency
        """)
    else:
        st.success("All cities are within acceptable risk levels. Continue routine monitoring.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #7f8c8d;'>
        <p>CityBAND - Data-Driven Adaptive Analytics for Better Neighborhood Health</p>
        <p>Powered by Streamlit | Data updated in real-time</p>
    </div>
""", unsafe_allow_html=True)

# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import sqlite3
# import os

# # ---------------- PAGE CONFIG ----------------
# st.set_page_config(
#     page_title="Public Health Intelligence Dashboard",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # ---------------- CUSTOM UI STYLE ----------------
# st.markdown("""
# <style>
# div[data-testid="metric-container"] {
#     background-color: #f5f7fa;
#     border-radius: 10px;
#     padding: 14px;
#     border-left: 6px solid #4c78a8;
# }
# </style>
# """, unsafe_allow_html=True)

# # ---------------- LOAD DATA ----------------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DB_PATH = os.path.join(BASE_DIR, "database/final.db")

# conn = sqlite3.connect(DB_PATH)
# df = pd.read_sql_query("SELECT * FROM health_data", conn)
# conn.close()

# # ---------------- DATA CLEANING ----------------
# df.columns = df.columns.str.strip().str.lower()
# df["date"] = pd.to_datetime(df["date"])

# # ---------------- CITY COORDINATES ----------------
# city_coords = {
#     "Ahmedabad": (23.0225, 72.5714),
#     "Delhi": (28.6139, 77.2090),
#     "Mumbai": (19.0760, 72.8777),
#     "Bengaluru": (12.9716, 77.5946),
#     "Chennai": (13.0827, 80.2707)
# }

# df["lat"] = df["city"].map(lambda x: city_coords.get(x, (None, None))[0])
# df["lon"] = df["city"].map(lambda x: city_coords.get(x, (None, None))[1])

# # ---------------- FEATURE ENGINEERING ----------------
# df["respiratory_ratio"] = df["respiratory_cases"] / df["total_hospital_visits"]
# df["heat_ratio"] = df["heat_related_cases"] / df["total_hospital_visits"]

# df["health_risk_score"] = (
#     0.6 * df["respiratory_ratio"] +
#     0.4 * df["heat_ratio"]
# )

# def risk_level(score):
#     if score < 0.05:
#         return "Low"
#     elif score < 0.15:
#         return "Medium"
#     else:
#         return "High"

# df["risk_level"] = df["health_risk_score"].apply(risk_level)

# # ---------------- SIDEBAR FILTERS ----------------
# st.sidebar.header("🔍 Filters")

# city = st.sidebar.selectbox(
#     "Select City",
#     sorted(df["city"].unique())
# )
# # Default city comes from sidebar
# top_city = city

# date_range = st.sidebar.date_input(
#     "Select Date Range",
#     [df["date"].min(), df["date"].max()]
# )

# filtered_df = df[
#     (df["city"] == top_city) &
#     (df["date"] >= pd.to_datetime(date_range[0])) &
#     (df["date"] <= pd.to_datetime(date_range[1]))
# ].copy()


# # ---------------- TITLE ----------------


# st.title("🌍 Public Health Intelligence Dashboard")
# st.markdown(
#     "Data-driven insights for **urban health risk assessment** and **policy decision-making**."
# )

# # 🔍 City selector at top (UX improvement)
# top_city = st.selectbox(
#     "🔎 Search & Select City",
#     sorted(df["city"].unique()),
#     index=sorted(df["city"].unique()).index(city)
# )

# st.markdown("---")



# # ---------------- KPI METRICS ----------------
# st.subheader("📌 City Health Overview")

# k1, k2, k3, k4 = st.columns(4)

# k1.metric("🏥 Avg Hospital Visits", int(filtered_df["total_hospital_visits"].mean()))
# k2.metric("🌬 Avg Respiratory Cases", int(filtered_df["respiratory_cases"].mean()))
# k3.metric("🔥 Avg Heat Cases", int(filtered_df["heat_related_cases"].mean()))
# k4.metric("⚠️ Risk Score", round(filtered_df["health_risk_score"].mean(), 3))

# # ---------------- RISK STATUS ----------------
# st.markdown("### ⚠️ Current Risk Status")

# risk_mode = filtered_df["risk_level"].mode()[0]

# if risk_mode == "Low":
#     st.success("🟢 LOW RISK — Conditions are stable")
# elif risk_mode == "Medium":
#     st.warning("🟡 MEDIUM RISK — Monitoring recommended")
# else:
#     st.error("🔴 HIGH RISK — Immediate intervention required")

# st.markdown("---")

# # ---------------- MAP ----------------
# st.subheader("🗺️ City Health Risk Map")

# map_df = filtered_df.dropna(subset=["lat", "lon"])

# fig_map = px.scatter_mapbox(
#     map_df,
#     lat="lat",
#     lon="lon",
#     size="health_risk_score",
#     color="risk_level",
#     hover_name="city",
#     hover_data={
#         "health_risk_score": True,
#         "respiratory_cases": True,
#         "heat_related_cases": True
#     },
#     zoom=6,
#     height=520
# )

# fig_map.update_layout(
#     mapbox_style="carto-positron",
#     margin=dict(l=0, r=0, t=30, b=0),
#     legend=dict(bgcolor="rgba(255,255,255,0.8)")
# )

# st.plotly_chart(fig_map, use_container_width=True)

# st.caption("💡 Higher bubble size and darker colors indicate elevated public health risk.")

# st.markdown("---")

# # ---------------- VISUAL GRID ----------------
# st.subheader("📊 Health Analytics")

# col1, col2 = st.columns(2)

# # --- PIE: Risk Distribution ---
# risk_pie = filtered_df["risk_level"].value_counts().reset_index()
# risk_pie.columns = ["Risk Level", "Days"]

# fig_pie = px.pie(
#     risk_pie,
#     names="Risk Level",
#     values="Days",
#     hole=0.4,
#     title="Health Risk Distribution"
# )

# col1.plotly_chart(fig_pie, use_container_width=True)

# # --- LINE: Trends ---
# fig_trend = px.line(
#     filtered_df,
#     x="date",
#     y=["respiratory_cases", "heat_related_cases"],
#     title="Disease Trend Over Time"
# )

# col2.plotly_chart(fig_trend, use_container_width=True)

# st.markdown("---")

# # ---------------- BAR: DAILY LOAD ----------------
# st.subheader("🏥 Daily Hospital Load")

# fig_bar = px.bar(
#     filtered_df,
#     x="date",
#     y=["respiratory_cases", "heat_related_cases"],
#     title="Hospital Case Composition",
#     labels={"value": "Cases", "variable": "Case Type"}
# )

# st.plotly_chart(fig_bar, use_container_width=True)

# st.caption("💡 Respiratory cases contribute more consistently to hospital load.")

# st.markdown("---")

# # ---------------- WHAT-IF SCENARIO ----------------
# with st.expander("🔮 What-If Scenario: Heatwave Impact", expanded=True):

#     increase = st.slider(
#         "Increase Heat-Related Cases (%)",
#         0, 50, 10
#     )

#     filtered_df["simulated_heat"] = (
#         filtered_df["heat_related_cases"] * (1 + increase / 100)
#     )

#     fig_sim = px.line(
#         filtered_df,
#         x="date",
#         y=["heat_related_cases", "simulated_heat"],
#         title="Simulated Heatwave Scenario"
#     )

#     st.plotly_chart(fig_sim, use_container_width=True)

#     st.caption(
#         "Simulation highlights potential surge in healthcare burden during extreme heat events."
#     )

# # ---------------- ROLLING AVERAGE ----------------
# st.markdown("---")
# st.subheader("📉 Early Warning Indicator")

# filtered_df = filtered_df.sort_values("date")
# filtered_df["resp_7d_avg"] = filtered_df["respiratory_cases"].rolling(7).mean()

# fig_roll = px.line(
#     filtered_df,
#     x="date",
#     y="resp_7d_avg",
#     title="7-Day Rolling Average of Respiratory Cases"
# )

# st.plotly_chart(fig_roll, use_container_width=True)

# # ---------------- HEATMAP ----------------
# st.markdown("---")
# st.subheader("🔥 Weekly Risk Intensity Heatmap")

# filtered_df["week"] = filtered_df["date"].dt.isocalendar().week

# heatmap_df = (
#     filtered_df
#     .groupby(["week", "risk_level"])
#     .size()
#     .reset_index(name="days")
# )

# fig_heat = px.density_heatmap(
#     heatmap_df,
#     x="week",
#     y="risk_level",
#     z="days",
#     title="Weekly Health Risk Intensity"
# )

# st.plotly_chart(fig_heat, use_container_width=True)

# # ---------------- FOOTER ----------------
# st.sidebar.markdown("---")
# st.sidebar.caption("⏱ Live Simulation")
# st.sidebar.write(pd.Timestamp.now())
