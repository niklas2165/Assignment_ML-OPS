import time
import subprocess
from datetime import datetime

def run_prediction():
    print(f"[{datetime.now()}] Running prediction...")
    subprocess.run(["python", "daily_prediction.py"])

if __name__ == "__main__":
    while True:
        run_prediction()
        time.sleep(24 * 60 * 60)  # wait 24 hours
