import json
import requests
from io import BytesIO
from typing import List, Optional
from datetime import datetime
from PIL import Image
from openai import OpenAI
import subprocess
from pydantic import BaseModel, Field
from groq import Groq
from dotenv import load_dotenv
import os


load_dotenv()
# Load the geojson file with traffic camera data


def load_traffic_cameras(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# Find the Aberdeen Tunnel Wan Chai entrance camera


XAI_API_KEY = os.getenv("X_API_KEY")


def find_aberdeen_tunnel_camera(geojson_data):
    for feature in geojson_data["features"]:
        description = feature["properties"].get("description", "")
        if "青朗公路近帝景軒 - 北行" in description:
            return feature["properties"]
    return None

# Define a Pydantic model for traffic analysis


client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1",
)


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

# Analyze traffic image using Grok


def text_to_speech_command_line(text, output_file="traffic_report.mp3"):
    voice = "zh-HK-HiuGaaiNeural"
    command = [
        "edge-tts",
        "--text", text,
        "--voice", voice,
        "--write-media", output_file,
        # "--write-subtitles", f"{output_file.replace('.mp3', '.srt')}"  # 可選：生成字幕
    ]
    try:
        subprocess.run(command, check=True)
        print(f"語音已生成並保存為 {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"生成語音時出錯: {e}")


def analyze_traffic_image(image_url, api_key):
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
                            "text": "請分析這張香港交通照片，提供交通狀況報告。\n"
                            "請提供以下資訊：\n"
                            "1. 整體擠塞程度（無/輕微/中度/嚴重）\n"
                            "2. 交通流量描述\n"
                            "3. 可見車輛數量估計\n"
                            "4. 天氣狀況\n"
                            "5. 是否有意外或其他特別觀察"
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

        # Parse the analysis into structured data
        traffic_data = TrafficCondition(
            congestion_level="Moderate",  # You should parse this from analysis
            flow_description=analysis,
            queue_length=None,  # Parse from analysis
            is_accident_visible=False,  # Parse from analysis
            weather_condition=None,  # Parse from analysis
            timestamp=datetime.now().isoformat(),
            vehicle_density=VehicleDensity(
                level="Medium",  # Parse from analysis
                vehicles_count_estimate=None  # Parse from analysis
            ),
            special_observations=[]  # Parse from analysis
        )

        return traffic_data

    except Exception as e:
        print(f"Error analyzing traffic image: {str(e)}")
        return None


def main():
    geojson_data = load_traffic_cameras("traffic_cameras_tc.geojson")
    camera = find_aberdeen_tunnel_camera(geojson_data)

    if not camera:
        print("未找到指定的攝影機。")
        return

    print(f"找到攝影機: {camera['description']}")
    print(f"攝影機 URL: {camera['url']}")

    # 分析交通狀況
    result = analyze_traffic_image(camera['url'], XAI_API_KEY)

    if result:
        print("\n交通分析結果:")
        result_dict = result.dict()
        print(json.dumps(result_dict, indent=2, default=str, ensure_ascii=False))

        # 將分析結果轉為粵語語音描述
        traffic_summary = (
            f"交通狀況報告：擠塞程度係 {result.congestion_level}，"
            f"流量描述如下：{result.flow_description}。"
            f"時間係 {result.timestamp}。"
        )
        text_to_speech_command_line(traffic_summary)


if __name__ == "__main__":
    main()
