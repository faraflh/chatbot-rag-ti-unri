"""Entrypoint root untuk Hugging Face Spaces / Streamlit Cloud.

HF Spaces dan Streamlit Cloud secara default mencari file `app.py` di root.
File ini mendelegasikan ke modul utama di `app/streamlit_app.py`.
"""

from app.streamlit_app import main

if __name__ == "__main__":
    main()
