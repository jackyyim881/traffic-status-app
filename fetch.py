import requests
import json

# WFS 請求
wfs_url = "https://portal.csdi.gov.hk/server/services/common/td_rcd_1638952287148_39267/MapServer/WFSServer"
params = {
    "service": "WFS",
    "request": "GetFeature",
    "typeName": "Traffic_Camera_Locations_Tc",
    "outputFormat": "GeoJSON",
    "bbox": "22.15,113.83,22.56,114.44"
}

response = requests.get(wfs_url, params=params)
if response.status_code == 200:
    # 確保以 UTF-8 解碼，避免中文顯示為轉義序列
    response.encoding = 'utf-8'
    data = response.json()
    # 打印第一個要素的 properties
    print("樣本 properties:", data["features"][0]["properties"])
else:
    print(f"請求失敗：{response.status_code}")
    print(response.text)
