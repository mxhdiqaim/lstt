from huggingface_hub import login, snapshot_download

login(os.getenv("HUGGINGFACE_TOKEN"))

models = [
    "NigerianAccentedEnglish",
    "Hausa-ASR",
    "Yoruba-ASR",
    "Igbo-ASR"
]

for model in models:
    print(f"\n--- Downloading {model} ---")
    snapshot_download(repo_id=f"NCAIR1/{model}", local_dir=model)

print("\nSuccess! All models downloaded.")