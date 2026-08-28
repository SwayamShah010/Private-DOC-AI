"""
Entrypoint: run the PrivateDocs AI app with `python run.py`.

Equivalent to running `streamlit run app/ui/streamlit_app.py` directly -
this wrapper exists so there's one obvious command to start the app,
matching the "run.py" file named in the PRD's recommended project
structure (Section 16).
"""
import subprocess
import sys
from pathlib import Path


def main():
    app_path = Path(__file__).parent / "app" / "ui" / "streamlit_app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])


if __name__ == "__main__":
    main()
