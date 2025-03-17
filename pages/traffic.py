import streamlit as st
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import plotly.express as px


def fetch_traffic_data(url):
    try:
        # Fetch the XML data
        response = requests.get(url)
        response.raise_for_status()

        # Parse the XML content
        root = ET.fromstring(response.content)

        # Get the date from the root level
        date_elem = root.find('date')
        date = date_elem.text if date_elem is not None else "N/A"

        # List to store parsed data
        data = []

        # Iterate through each 'period' inside 'periods'
        for period in root.findall('.//periods/period'):
            period_from_elem = period.find('period_from')
            period_to_elem = period.find('period_to')

            period_from = period_from_elem.text if period_from_elem is not None else "N/A"
            period_to = period_to_elem.text if period_to_elem is not None else "N/A"

            # Find all detectors within this period
            detectors = period.findall('.//detectors/detector')
            for detector in detectors:
                detector_id_elem = detector.find('detector_id')
                direction_elem = detector.find('direction')

                detector_id = detector_id_elem.text if detector_id_elem is not None else "N/A"
                direction = direction_elem.text if direction_elem is not None else "N/A"

                # Find all lanes within this detector
                lanes = detector.findall('.//lanes/lane')
                for lane in lanes:
                    lane_data = {
                        'Date': date,
                        'Period From': period_from,
                        'Period To': period_to,
                        'Detector ID': detector_id,
                        'Direction': direction,
                        'Lane ID': lane.find('lane_id').text if lane.find('lane_id') is not None else "N/A",
                        'Speed (km/h)': int(lane.find('speed').text) if lane.find('speed') is not None else 0,
                        'Occupancy (%)': int(lane.find('occupancy').text) if lane.find('occupancy') is not None else 0,
                        'Volume': int(lane.find('volume').text) if lane.find('volume') is not None else 0,
                        'Speed Std Dev': float(lane.find('s.d.').text) if lane.find('s.d.') is not None and lane.find('s.d.').text else 0.0,
                        'Valid': lane.find('valid').text if lane.find('valid') is not None else "N/A"
                    }
                    data.append(lane_data)

        # Convert to DataFrame
        df = pd.DataFrame(data)
        return df

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data: {e}")
        return None
    except ET.ParseError as e:
        st.error(f"Error parsing XML: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


def main():
    st.title("Traffic Speed, Volume, and Road Occupancy Dashboard")
    url = "https://resource.data.one.gov.hk/td/traffic-detectors/rawSpeedVol-all.xml"

    with st.spinner("Fetching and parsing data..."):
        df = fetch_traffic_data(url)

    if df is not None and not df.empty:
        # Convert time strings to datetime for better plotting
        df['Period From'] = pd.to_datetime(df['Period From'])
        df['Period To'] = pd.to_datetime(df['Period To'])

        # Add tabs for better organization
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Raw Data", "Interactive Maps", "Speed Analysis", "Volume & Occupancy"])

        with tab1:
            st.subheader("Raw Traffic Data")
            st.dataframe(df)

            # Advanced filtering options
            st.subheader("Filter Data")
            col1, col2 = st.columns(2)

            with col1:
                detector_ids = df['Detector ID'].unique()
                selected_detector = st.selectbox("Select Detector ID", options=[
                                                 "All"] + list(detector_ids))

            with col2:
                directions = df['Direction'].unique()
                selected_direction = st.selectbox("Select Direction", options=[
                                                  "All"] + list(directions))

            filtered_df = df
            if selected_detector != "All":
                filtered_df = filtered_df[filtered_df['Detector ID']
                                          == selected_detector]
            if selected_direction != "All":
                filtered_df = filtered_df[filtered_df['Direction']
                                          == selected_direction]

            st.subheader("Filtered Data")
            st.dataframe(filtered_df)

            # Download option
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download filtered data as CSV",
                data=csv,
                file_name="traffic_data.csv",
                mime="text/csv"
            )

        with tab2:
            st.subheader("Traffic Detector Map")
            # Note: For a real map, you'd need detector coordinates
            # This is a placeholder. In practice, you'd use actual lat/long of detectors
            st.warning(
                "Note: This is a placeholder map. Actual implementation requires detector coordinates.")

            # Create sample map data for demonstration
            map_data = pd.DataFrame({
                'Detector ID': df['Detector ID'].unique(),
                # Hong Kong coords
                'lat': np.random.uniform(22.2, 22.4, len(df['Detector ID'].unique())),
                'lon': np.random.uniform(114.1, 114.3, len(df['Detector ID'].unique())),
                'Speed': [df[df['Detector ID'] == detector]['Speed (km/h)'].mean() for detector in df['Detector ID'].unique()]
            })

            # Limit the scale of the map for better performance
            map_data = map_data[(map_data['lat'] >= 22.2) & (map_data['lat'] <= 22.4) &
                                (map_data['lon'] >= 114.1) & (map_data['lon'] <= 114.3)]

            fig_map = px.scatter_mapbox(
                map_data,
                lat="lat",
                lon="lon",
                color="Speed",
                size="Speed",
                hover_name="Detector ID",
                hover_data=["Speed"],
                color_continuous_scale=px.colors.cyclical.IceFire,
                size_max=15,
                zoom=10,
                mapbox_style="carto-positron"
            )

            st.plotly_chart(fig_map, use_container_width=True)

        with tab3:
            st.subheader("Speed Analysis")

            # Speed distribution histogram
            fig_hist = px.histogram(
                filtered_df,
                x="Speed (km/h)",
                nbins=20,
                color="Direction",
                title="Speed Distribution by Direction"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            # Speed over time
            st.subheader("Speed Trends")
            fig_line = px.line(
                filtered_df,
                x="Period From",
                y="Speed (km/h)",
                color="Detector ID",
                line_group="Lane ID",
                hover_data=["Volume", "Occupancy (%)"],
                title="Speed Over Time by Detector and Lane"
            )
            st.plotly_chart(fig_line, use_container_width=True)

            # Speed by lane comparison
            st.subheader("Speed by Lane")
            fig_box = px.box(
                filtered_df,
                x="Lane ID",
                y="Speed (km/h)",
                color="Direction",
                title="Speed Distribution by Lane"
            )
            st.plotly_chart(fig_box, use_container_width=True)

        with tab4:
            st.subheader("Volume & Occupancy Analysis")

            # Volume vs Occupancy scatter plot
            fig_scatter = px.scatter(
                filtered_df,
                x="Occupancy (%)",
                y="Volume",
                color="Speed (km/h)",
                size="Speed (km/h)",
                hover_data=["Detector ID", "Lane ID", "Direction"],
                title="Volume vs Occupancy Relationship"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            # Volume by detector
            st.subheader("Traffic Volume by Detector")
            fig_bar = px.bar(
                filtered_df.groupby("Detector ID")[
                    "Volume"].sum().reset_index(),
                x="Detector ID",
                y="Volume",
                title="Total Volume by Detector"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Occupancy heatmap
            st.subheader("Road Occupancy Heatmap")
            pivot_data = filtered_df.pivot_table(
                index="Detector ID",
                columns="Period From",
                values="Occupancy (%)",
                aggfunc='mean'
            ).fillna(0)

            fig_heatmap = px.imshow(
                pivot_data,
                labels=dict(x="Time", y="Detector ID", color="Occupancy (%)"),
                title="Occupancy Heatmap by Detector and Time"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)

            # Traffic metrics summary
            st.subheader("Traffic Metrics Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Avg Speed", f"{filtered_df['Speed (km/h)'].mean():.1f} km/h")
            with col2:
                st.metric("Avg Volume", f"{filtered_df['Volume'].mean():.1f}")
            with col3:
                st.metric("Avg Occupancy",
                          f"{filtered_df['Occupancy (%)'].mean():.1f}%")
    else:
        st.warning("No data parsed successfully.")


if __name__ == "__main__":
    main()
