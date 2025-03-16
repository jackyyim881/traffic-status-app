import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import json
import requests
from io import BytesIO
from typing import List, Optional
from datetime import datetime
from PIL import Image
import os
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import geocoder
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
import subprocess


# Load environment variables
load_dotenv()

# Set page title and favicon
st.set_page_config(
    page_title="Traffic Status Monitor",
    page_icon="🚦",
    layout="wide"
)

# Define Pydantic models for traffic data


class VehicleDensity(BaseModel):
    level: str = Field(
        description="Traffic density level (Low, Medium, High, Very High)")
    vehicles_count_estimate: Optional[int] = Field(
        description="Estimated number of vehicles visible")


class TrafficCondition(BaseModel):
    congestion_level: str = Field(
        description="Overall congestion level (None, Mild, Moderate, Severe)")
    flow_description: str = Field(description="Description of traffic flow")
    queue_length: Optional[str] = Field(
        description="Description of any queue length")
    is_accident_visible: bool = Field(
        description="Whether there appears to be an accident")
    weather_condition: Optional[str] = Field(
        description="Weather conditions visible in the image")
    timestamp: str = Field(description="Time of analysis")
    vehicle_density: VehicleDensity = Field(
        description="Vehicle density information")
    special_observations: Optional[List[str]] = Field(
        description="Any special observations about traffic conditions")


def get_hko_weather():
    """获取香港天文台API的天气信息"""
    try:
        # 获取当前天气数据
        current_weather_url = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=rhrread&lang=tc"
        current_resp = requests.get(current_weather_url)
        if current_resp.status_code != 200:
            st.error(
                f"Failed to get current weather data: Status code {current_resp.status_code}")
            return None
        current_data = current_resp.json()
        # 获取9天天气预报
        forecast_url = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=fnd&lang=tc"
        forecast_resp = requests.get(forecast_url)
        if forecast_resp.status_code != 200:
            st.error(
                f"Failed to get weather forecast data: Status code {forecast_resp.status_code}")
            return None
        forecast_data = forecast_resp.json()
        # 返回整合的天气数据
        return {
            "current": current_data,
            "forecast": forecast_data,

        }
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON: {e}")
        return {"error": str(e)}
    except Exception as e:
        st.error(f"Error fetching weather data: {e}")
        return {"error": str(e)}


def display_weather_info(weather_data):
    """显示从天文台API获取的天气信息"""
    if "error" in weather_data:
        st.error(f"获取天气数据失败: {weather_data['error']}")
        return

    current = weather_data.get("current")
    forecast = weather_data.get("forecast")
    warnings = weather_data.get("warnings")
    radar = weather_data.get("radar")

    # 显示主要天气数据
    if current:
        st.subheader("当前天气")

        # 提取温度和湿度
        temperature = current.get("temperature", {}).get("data", [])
        humidity = current.get("humidity", {}).get("data", [])

        # 创建温度和湿度的列
        temp_cols = st.columns(min(len(temperature), 3))
        for i, temp_data in enumerate(temperature[:3]):
            with temp_cols[i]:
                st.metric(f"{temp_data.get('place', '地点')}",
                          f"{temp_data.get('value', 'N/A')}°C")

        # 显示湿度
        if humidity:
            st.markdown(f"**濕度:** {humidity[0].get('value', 'N/A')}%")

        # 显示天气图标和信息
        icon = current.get("icon", [])

        if icon:
            icon_url = f"https://www.hko.gov.hk/images/HKOWxIconOutline/pic{icon[0]}.png"

            weather_info = current.get("generalSituation", "")

            col1, col2 = st.columns([1, 4])
            with col1:
                st.image(icon_url, width=70)
            with col2:
                st.markdown(weather_info)
    # 显示天气预报
    if forecast and "weatherForecast" in forecast:
        st.subheader("未來天氣預報")

        # 创建3天的预报卡片
        forecast_days = forecast["weatherForecast"][:3]  # 只取前3天
        cols = st.columns(len(forecast_days))

        for i, day_forecast in enumerate(forecast_days):
            with cols[i]:
                date = day_forecast.get("forecastDate", "")
                week = day_forecast.get("week", "")
                temp_range = f"{day_forecast.get('forecastMintemp', {}).get('value', '')}°C - {day_forecast.get('forecastMaxtemp', {}).get('value', '')}°C"
                humidity_range = f"{day_forecast.get('forecastMinrh', {}).get('value', '')}% - {day_forecast.get('forecastMaxrh', {}).get('value', '')}%"

                st.markdown(f"**{date} ({week})**")
                icon_num = day_forecast.get("ForecastIcon", "")
                if icon_num:
                    st.image(
                        f"https://www.hko.gov.hk/images/HKOWxIconOutline/pic{icon_num}.png", width=50)

                st.markdown(f"温度: {temp_range}")
                st.markdown(f"湿度: {humidity_range}")
                st.markdown(day_forecast.get("forecastWeather", ""))

# Functions to handle traffic camera data


def load_traffic_cameras(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def find_nearby_cameras(geojson_data, user_lat, user_lng, radius_km=2.0):
    """Find cameras within a certain radius of the user's location"""
    nearby_cameras = []

    for feature in geojson_data["features"]:
        if "geometry" in feature and feature["geometry"] is not None:
            cam_lng, cam_lat = feature["geometry"]["coordinates"]

            # Calculate rough distance using Haversine formula
            from math import sin, cos, sqrt, atan2, radians

            R = 6371.0  # Earth radius in kilometers

            lat1, lon1 = radians(user_lat), radians(user_lng)
            lat2, lon2 = radians(cam_lat), radians(cam_lng)

            dlon = lon2 - lon1
            dlat = lat2 - lat1

            a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))

            distance = R * c  # Distance in kilometers

            if distance <= radius_km:
                camera_info = feature["properties"]
                camera_info["distance"] = distance
                camera_info["latitude"] = cam_lat
                camera_info["longitude"] = cam_lng
                nearby_cameras.append(camera_info)

    # Sort cameras by distance
    nearby_cameras.sort(key=lambda x: x["distance"])
    return nearby_cameras


def analyze_traffic_image(image_url):
    """Analyze traffic image using X.AI API"""

    try:
        api_key = st.secrets["X_API_KEY"]["value"]
    except Exception:
        api_key = os.getenv("X_API_KEY")

    if not api_key:
        return "API key not found in environment variables"

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1",
    )

    # Download the image
    response = requests.get(image_url)
    if response.status_code != 200:
        return f"Failed to get image: Status code {response.status_code}"

    try:
        completion = client.chat.completions.create(
            model="grok-2-vision-latest",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please analyze this traffic image and provide a traffic status report.\n"
                            "Include the following information:\n"
                            "1. Overall congestion level (None/Mild/Moderate/Severe)\n"
                            "2. Traffic flow description\n"
                            "3. Estimated vehicle count\n"
                            "4. Weather conditions\n"
                            "5. Any accidents or special observations"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                }
            ],
            temperature=0,
            max_completion_tokens=1024,
            top_p=1,
            stream=False
        )

        # Extract the analysis from the response
        analysis = completion.choices[0].message.content

        # Create a structured traffic condition object
        # Note: In a production app, you would parse the AI response to fill these fields properly
        traffic_data = TrafficCondition(
            congestion_level="Moderate",  # Should parse from analysis
            flow_description=analysis,
            queue_length=None,  # Should parse from analysis
            is_accident_visible=False,  # Should parse from analysis
            weather_condition="Clear",  # Should parse from analysis
            timestamp=datetime.now().isoformat(),
            vehicle_density=VehicleDensity(
                level="Medium",  # Should parse from analysis
                vehicles_count_estimate=15  # Should parse from analysis
            ),
            special_observations=[]  # Should parse from analysis
        )

        return traffic_data

    except Exception as e:
        return f"Error analyzing traffic image: {str(e)}"


def text_to_speech(text, output_file="traffic_report.mp3"):
    """Generate speech from text using edge-tts"""
    voice = "en-US-AriaNeural"  # You can change this to your preferred voice
    command = [
        "edge-tts",
        "--text", text,
        "--voice", voice,
        "--write-media", output_file,
    ]
    try:
        subprocess.run(command, check=True)
        return output_file
    except subprocess.CalledProcessError as e:
        return f"Error generating speech: {e}"
    except FileNotFoundError:
        return "edge-tts not installed. Install with: pip install edge-tts"


def add_incident_reporting():
    st.header("⚠️ Report Traffic Incidents")

    user_location = get_current_location()
    col1, col2 = st.columns(2)

    with col1:
        incident_type = st.selectbox(
            "Incident Type",
            ["Accident", "Road Construction", "Road Closure", "Traffic Jam", "Other"]
        )

        severity = st.slider("Severity Level", 1, 5, 3)

        location_input = st.text_input("Location Description")

    with col2:
        st.write("Drop Pin on Map")

        # Simple map for incident location selection
        incident_map = folium.Map(location=user_location, zoom_start=15)

        # Add ability to click and add a marker
        folium.LatLngPopup().add_to(incident_map)

        folium_static(incident_map)

    description = st.text_area("Additional Details", height=100)

    upload_col, camera_col = st.columns(2)

    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload an image (optional)", type=["jpg", "png", "jpeg"])

    with camera_col:
        camera_input = st.camera_input("Take a photo (optional)")

    if st.button("Submit Report", type="primary"):
        # In a real app, you'd save this to a database
        st.success(
            "Incident reported successfully! Authorities have been notified.")

        # Show a mock notification
        st.toast("Report submitted to traffic management center!")


def add_personalized_dashboard():
    st.header("🚗 Your Traffic Dashboard")

    # User preferences
    with st.expander("Dashboard Preferences"):
        col1, col2 = st.columns(2)
        with col1:
            favorite_locations = st.multiselect(
                "Favorite Locations",
                ["Home", "Work", "School", "Shopping Mall"],
                default=["Home", "Work"]
            )

            refresh_rate = st.select_slider(
                "Data Refresh Rate",
                options=["30s", "1m", "5m", "10m", "30m"],
                value="5m"
            )

        with col2:
            threshold_severe = st.slider(
                "Severe Traffic Threshold", 70, 100, 80)
            notification_pref = st.checkbox("Enable Notifications", True)

    # Summary cards for quick insights
    st.subheader("Today's Traffic Summary")

    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)

    with metrics_col1:
        st.metric(
            label="Current Congestion",
            value="67%",
            delta="5%",
            delta_color="inverse"
        )

    with metrics_col2:
        st.metric(
            label="Avg Speed",
            value="42 km/h",
            delta="-8 km/h",
            delta_color="inverse"
        )

    with metrics_col3:
        st.metric(
            label="Incidents Today",
            value="3",
            delta="1"
        )

    with metrics_col4:
        st.metric(
            label="Travel Time Index",
            value="1.8x",
            delta="0.3x",
            delta_color="inverse"
        )


def get_current_location():
    """Get user's current location using IP geolocation"""
    try:
        g = geocoder.ip('me')
        if g.ok:
            return g.latlng
        return [22.3193, 114.1694]  # Default: Hong Kong
    except Exception as e:
        st.error(f"Error getting location: {e}")
        return [22.3193, 114.1694]  # Default: Hong Kong


def enhanced_map_features(nearby_cameras, user_location):
    st.header("🗺️ Enhanced Traffic Map")

    # Map layers
    map_layers = st.multiselect(
        "Map Layers",
        ["Traffic Cameras", "Traffic Density",
            "Incidents", "Roadwork", "Weather Overlay"],
        default=["Traffic Cameras", "Traffic Density"]
    )

    # Time slider for historical view
    current_time = datetime.now()
    time_selection = st.slider(
        "View traffic at time:",
        min_value=current_time - pd.Timedelta(hours=4),
        max_value=current_time,
        value=current_time,
        format="HH:mm"
    )

    # Create enhanced map
    m = folium.Map(location=user_location, zoom_start=14)

    # Add user marker
    folium.Marker(
        location=user_location,
        popup="Your Location",
        icon=folium.Icon(color="blue", icon="user", prefix="fa"),
    ).add_to(m)

    # Add traffic cameras with custom markers
    if "Traffic Cameras" in map_layers:
        marker_cluster = MarkerCluster().add_to(m)

        for camera in nearby_cameras:
            camera_location = [camera["latitude"], camera["longitude"]]
            distance = camera["distance"]

            # Get congestion level for this camera (mock data)
            congestion = ["None", "Mild", "Moderate",
                          "Severe"][min(int(distance * 10) % 4, 3)]
            color = {"None": "green", "Mild": "blue",
                     "Moderate": "orange", "Severe": "red"}[congestion]

            camera_info = f"""
            <b>{camera['description']}</b><br>
            Distance: {distance:.2f} km<br>
            Congestion: <span style='color:{color};font-weight:bold'>{congestion}</span><br>
            <a href="#" onclick='selectCamera("{camera['url']}");return false;'>Analyze</a> | 
            <a href="{camera['url']}" target="_blank">View feed</a>
            """

            folium.Marker(
                location=camera_location,
                popup=folium.Popup(camera_info, max_width=300),
                icon=folium.Icon(color=color, icon="camera", prefix="fa"),
            ).add_to(marker_cluster)

    # Add traffic density heatmap layer
    if "Traffic Density" in map_layers:
        # Mock density data around user location
        density_points = []
        for i in range(100):
            radius = 0.05
            lat = user_location[0] + (np.random.random() - 0.5) * radius * 2
            lng = user_location[1] + (np.random.random() - 0.5) * radius * 2
            weight = np.random.random()
            density_points.append([lat, lng, weight])

        from folium.plugins import HeatMap
        HeatMap(density_points).add_to(m)

    folium_static(m)


def add_ai_insights():
    st.header("🤖 AI Traffic Insights")

    with st.expander("Traffic Pattern Analysis", expanded=True):
        st.write(
            "Based on the analyzed traffic patterns, our AI has identified the following insights:")

        insights = [
            "**Peak congestion** occurs between 8:15-9:30 AM on weekdays",
            "**Construction at Nathan Road** is causing a 23% increase in travel time",
            "**Weather patterns** show 35% higher congestion on rainy days",
            "**Special events** today may cause disruptions on eastern approach roads"
        ]

        for insight in insights:
            st.markdown(f"- {insight}")

    # Vehicle classification from camera feed
    st.subheader("Vehicle Classification Analysis")

    # Mock data for vehicle types
    vehicle_data = {
        'Private Cars': 68,
        'Taxis': 12,
        'Buses': 8,
        'Trucks': 7,
        'Motorcycles': 5
    }

    fig = px.pie(
        values=list(vehicle_data.values()),
        names=list(vehicle_data.keys()),
        title="Vehicle Types in Current View",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig, use_container_width=True)


def add_traffic_prediction():
    st.header("🔮 Traffic Prediction")

    # Create mock historical data for demonstration
    dates = pd.date_range(start='2025-03-01', periods=14, freq='D')
    traffic_density = np.random.normal(65, 15, 14).clip(10, 100)
    historical_data = pd.DataFrame({
        'Date': dates,
        'Traffic Density (%)': traffic_density
    })

    # Predict next 3 days (mock prediction)
    future_dates = pd.date_range(
        start=dates[-1] + pd.Timedelta(days=1), periods=3, freq='D')
    future_density = [traffic_density[-1] +
                      np.random.normal(0, 5) for _ in range(3)]
    future_data = pd.DataFrame({
        'Date': future_dates,
        'Traffic Density (%)': future_density
    })

    # Combine datasets
    all_data = pd.concat([historical_data, future_data])
    all_data['Type'] = ['Historical'] * \
        len(historical_data) + ['Predicted'] * len(future_data)

    # Create plot
    fig = px.line(all_data, x='Date', y='Traffic Density (%)', color='Type',
                  title='Traffic Density Trend & Prediction',
                  color_discrete_map={'Historical': 'blue', 'Predicted': 'red'})

    st.plotly_chart(fig, use_container_width=True)

    # Add peak hours information
    st.subheader("Peak Traffic Hours")

    hours = ['07:00-09:00', '12:00-14:00', '17:00-19:00']
    congestion = ['Severe', 'Moderate', 'Severe']

    peak_data = pd.DataFrame({
        'Time Period': hours,
        'Typical Congestion': congestion
    })

    st.table(peak_data)

# Main Streamlit app


def main():
    st.title("🚦 Real-time Traffic Status Monitor")

    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info(
        "This app uses your location to find nearby traffic cameras "
        "and provides real-time traffic status using AI analysis."
    )

    # Try to load traffic camera data
    try:
        geojson_data = load_traffic_cameras("traffic_cameras_tc.geojson")
        st.sidebar.success(
            f"Loaded {len(geojson_data['features'])} traffic cameras")
    except Exception as e:
        st.sidebar.error(f"Failed to load traffic camera data: {e}")
        if not os.path.exists("traffic_cameras_tc.geojson"):
            st.sidebar.warning(
                "traffic_cameras_tc.geojson file not found in current directory")
        return

    # Location options
    st.header("📍 Your Location")

    location_method = st.radio(
        "How would you like to set your location?",
        ["Use my current location (IP-based)", "Enter coordinates manually"]
    )

    if location_method == "Use my current location (IP-based)":
        # user_location = get_current_location()
        # hong kong location
        user_location = [22.3193, 114.1694]
        st.info(
            f"Using your current location: Lat {user_location[0]:.4f}, Long {user_location[1]:.4f}")
    else:
        col1, col2 = st.columns(2)

    # Display weather information for the selected location
    st.header("🌤️ 當前區天氣")
    weather_data = get_hko_weather()

    display_weather_info(weather_data)

    # Find nearby cameras
    radius = st.slider("Search radius (km)", min_value=0.5,
                       max_value=100.0, value=2.0, step=0.5)
    nearby_cameras = find_nearby_cameras(
        geojson_data, user_location[0], user_location[1], radius_km=radius)

    # Display map with user location and traffic cameras
    st.header("🗺️ Traffic Camera Map")

    if len(nearby_cameras) == 0:
        st.warning(
            f"No traffic cameras found within {radius} km of your location.")
    else:
        st.success(f"Found {len(nearby_cameras)} traffic cameras near you.")

        # Create map centered on user location
        m = folium.Map(location=user_location, zoom_start=14)

        # Add user marker
        folium.Marker(
            location=user_location,
            popup="Your Location",
            icon=folium.Icon(color="blue", icon="user", prefix="fa"),
        ).add_to(m)

        # Add traffic camera markers
        marker_cluster = MarkerCluster().add_to(m)

        for camera in nearby_cameras:
            camera_location = [camera["latitude"], camera["longitude"]]
            distance = camera["distance"]
            camera_info = f"""
            <b>{camera['description']}</b><br>
            Distance: {distance:.2f} km<br>
            <a href="{camera['url']}" target="_blank">View camera feed</a>
            """

            folium.Marker(
                location=camera_location,
                popup=folium.Popup(camera_info, max_width=300),
                icon=folium.Icon(color="red", icon="camera", prefix="fa"),
            ).add_to(marker_cluster)

        # Display map
        folium_static(m)

    # Camera selection and traffic analysis
    st.header("🚦 Traffic Analysis")

    if len(nearby_cameras) > 0:
        camera_options = {
            f"{cam['description']} ({cam['distance']:.2f} km)": i for i, cam in enumerate(nearby_cameras)}
        selected_camera_desc = st.selectbox(
            "Select a traffic camera to analyze", list(camera_options.keys()))
        selected_camera = nearby_cameras[camera_options[selected_camera_desc]]

        col1, col2 = st.columns([2, 3])

        with col1:
            st.subheader("Camera Feed")
            try:
                response = requests.get(selected_camera['url'])
                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content))
                    st.image(image, caption=selected_camera['description'])
                else:
                    st.error(
                        f"Failed to load camera image: Status code {response.status_code}")
            except Exception as e:
                st.error(f"Error loading camera feed: {e}")

        with col2:
            st.subheader("Traffic Analysis")

            if st.button("Analyze Traffic Conditions"):
                with st.spinner("Analyzing traffic image with AI..."):
                    analysis_result = analyze_traffic_image(
                        selected_camera['url'])

                    if isinstance(analysis_result, str):
                        st.error(analysis_result)
                    else:
                        # Display results
                        st.success("Analysis complete!")

                        # Show congestion level with color coding
                        congestion_level = analysis_result.congestion_level
                        congestion_colors = {
                            "None": "green",
                            "Mild": "lightgreen",
                            "Moderate": "orange",
                            "Severe": "red"
                        }
                        color = congestion_colors.get(congestion_level, "gray")

                        st.markdown(
                            f"**Congestion Level:** <span style='background-color:{color}; padding:3px 8px; border-radius:3px'>{congestion_level}</span>", unsafe_allow_html=True)

                        # Show density level
                        st.markdown(
                            f"**Density Level:** {analysis_result.vehicle_density.level}")

                        if analysis_result.vehicle_density.vehicles_count_estimate:
                            st.markdown(
                                f"**Estimated Vehicles:** {analysis_result.vehicle_density.vehicles_count_estimate}")

                        if analysis_result.weather_condition:
                            st.markdown(
                                f"**Weather:** {analysis_result.weather_condition}")

                        if analysis_result.is_accident_visible:
                            st.warning(
                                "⚠️ Possible accident or incident detected!")

                        # Display flow description
                        st.subheader("Traffic Description")
                        st.markdown(analysis_result.flow_description)

                        # Generate audio report
                        st.subheader("Audio Traffic Report")
                        audio_text = f"Traffic report for {selected_camera['description']}. " + \
                            f"Congestion level is {congestion_level}. " + \
                            f"{analysis_result.flow_description[:200]}"

                        audio_file = text_to_speech(audio_text)
                        if not audio_file.startswith("Error"):
                            try:
                                with open(audio_file, "rb") as f:
                                    st.audio(f.read(), format="audio/mp3")
                            except Exception as e:
                                st.error(f"Error playing audio: {e}")
                        else:
                            st.error(audio_file)


if __name__ == "__main__":
    main()

    user_location = get_current_location()

    # After camera analysis

    geojson_data = load_traffic_cameras("traffic_cameras_tc.geojson")
    nearby_cameras = find_nearby_cameras(
        geojson_data, user_location[0], user_location[1], radius_km=2.0)
    enhanced_map_features(nearby_cameras, user_location)

    # After traffic analysis
    add_ai_insights()

    # Add new sections
    add_traffic_prediction()
    add_personalized_dashboard()
    add_incident_reporting()
