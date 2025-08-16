
import sys
import requests

def main():
    url = "http://localhost:8000/predict"

    text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "الخدمة رائعة جدا ومميزة للغاية"

    response = requests.post(url, json={"text": text})

    if response.status_code == 200:
        try:
            sentiment = response.json()["sentiment"]  
            print("Predicted Sentiment:", sentiment)
        except Exception as e:
            print("Failed to parse JSON:", response.text)
    else:
        print("Error:", response.status_code, response.text)

if __name__ == "__main__":
    main()
