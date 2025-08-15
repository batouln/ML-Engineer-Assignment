def build_prompt(text: str) -> str:
    return f'''You are a helpful assistant. Your task is to classify the sentiment of the given text into one of the following categories:
["Positive", "Neutral", "Negative"]

Text: {text}

Sentiment:
'''
