import os
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

print("=== Azure OpenAI Connection Test ===")
print(f"Endpoint: {endpoint}")
print(f"Deployment: {deployment}")
print(f"API Version: {api_version}")
print(f"API Key: {subscription_key[:10]}...{subscription_key[-5:] if subscription_key else 'MISSING'}")
print("=" * 40)

try:
    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )

    print("\nSending test request...")
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": "Say 'Connection successful!' in one sentence.",
            }
        ],
        max_tokens=100,
        temperature=0.7,
        model=deployment
    )

    print("\n✅ SUCCESS!")
    print(f"Response: {response.choices[0].message.content}")
    print(f"\nModel used: {response.model}")

except Exception as e:
    print("\n❌ ERROR!")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print("\n=== Troubleshooting Tips ===")
    print("1. Check if your API key is valid and active")
    print("2. Verify the deployment name matches exactly in Azure Portal")
    print("3. Ensure the endpoint URL is correct")
    print("4. Try different API versions: 2024-08-01-preview, 2024-02-15-preview, or 2023-05-15")
