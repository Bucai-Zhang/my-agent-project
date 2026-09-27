from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

response = client.responses.create(
    model="deepseek-flash",
    input="你好，请只回复一句：API调用成功"
)

print(response.output_text)