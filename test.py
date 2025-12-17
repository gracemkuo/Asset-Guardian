from openai import AzureOpenAI
from dotenv import load_dotenv
import os

# 載入環境變數
load_dotenv()
def init_azure_openai():
    """初始化 Azure OpenAI 客戶端"""
    try:
        client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )
        print("✓ Azure OpenAI 客戶端初始化成功")
        print(f"Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
        return client
    except Exception as e:
        print(f"✗ Azure OpenAI 初始化失敗: {str(e)}")
        return None

if __name__ == "__main__":
    print("開始測試 Azure OpenAI 連接...")
    client = init_azure_openai()

    if client:
        print("\n客戶端已成功建立！")
        # 可以在這裡添加進一步的測試，例如：
        # response = client.chat.completions.create(...)
    else:
        print("\n客戶端初始化失敗，請檢查環境變數設定")

