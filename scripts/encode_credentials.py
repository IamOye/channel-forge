import base64

files = {
    'YOUTUBE_CLIENT_SECRET_B64':
        '.credentials/money_debate_client_secret.json',
    'YOUTUBE_TOKEN_B64':
        '.credentials/money_debate_token.json'
}

for var_name, file_path in files.items():
    try:
        with open(file_path, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode()
        print(f"{var_name}={encoded}")
        print()
    except FileNotFoundError:
        print(f"ERROR: {file_path} not found")
