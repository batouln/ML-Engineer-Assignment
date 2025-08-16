# simple_predict.py
import sys
import requests

URL = "http://127.0.0.1:8000/predict"
DEFAULT_TEXT = "الخدمة رائعة جدا ومميزة للغاية"

def main():
    # 1) text from command line, else use a nice default
    text = " ".join(sys.argv[1:]).strip() or DEFAULT_TEXT

    print(f"→ Server: {URL}")
    print(f"→ Text  : {text}\n")

    try:
        r = requests.post(URL, json={"text": text}, timeout=10)
    except Exception as e:
        print(f" Unable to reach server: {e}")
        return

    if r.status_code != 200:
        print(f" Server error {r.status_code}: {r.text}")
        return

    try:
        data = r.json()
    except Exception:
        print(f"✖ Invalid JSON from server:\n{r.text}")
        return

    sentiment = data.get("sentiment")
    if sentiment:
        print(f" Predicted Sentiment: {sentiment}")
    else:
        print(f" Response missing 'sentiment': {data}")

if __name__ == "__main__":
    main()
