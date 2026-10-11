from pathlib import Path
from urllib.request import urlopen, Request
from zipfile import ZipFile
from io import BytesIO
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_FILE = DATA_DIR / "student-mat.csv"
PASS_MARK = 10

# Official UCI dataset ZIP. The zip contains student-mat.csv and student-por.csv.
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
# Public mirror fallback if the official archive is temporarily unavailable.
MIRROR_URL = "https://www.stat.cmu.edu/~brian/valerie/617-2022/617-2021/project01/UCI%20ML%20data%20sets/student%20performance/student-mat.csv"

def load_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_FILE.exists():
        errors = []
        try:
            req = Request(UCI_ZIP_URL, headers={"User-Agent": "StudentPerformanceMLProject/1.0"})
            with urlopen(req, timeout=25) as response:
                content = response.read()
            with ZipFile(BytesIO(content)) as archive:
                match = next((n for n in archive.namelist() if n.lower().endswith("student-mat.csv")), None)
                if not match:
                    raise FileNotFoundError("student-mat.csv was not found inside the UCI ZIP.")
                DATA_FILE.write_bytes(archive.read(match))
        except Exception as exc:
            errors.append(f"UCI download failed: {exc}")
            try:
                req = Request(MIRROR_URL, headers={"User-Agent": "Mozilla/5.0"})
                with urlopen(req, timeout=20) as response:
                    DATA_FILE.write_bytes(response.read())
            except Exception as exc2:
                errors.append(f"Mirror download failed: {exc2}")
                raise RuntimeError("Could not download dataset. " + " | ".join(errors)) from exc2

    df = pd.read_csv(DATA_FILE, sep=";")
    if "G3" not in df.columns:
        raise ValueError("Dataset is missing expected target column G3. Check data/student-mat.csv.")
    df["pass_fail"] = (df["G3"] >= PASS_MARK).astype(int)
    return df

# Use all available fields except period grades and the final target to avoid grade leakage.
TARGET_COLUMNS = ["G1", "G2", "G3", "pass_fail"]
def get_feature_columns(df):
    return [col for col in df.columns if col not in TARGET_COLUMNS]

# These names reflect the UCI dataset's schema.
NUMERIC_FEATURES = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"
]
CATEGORICAL_FEATURES = [
    "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
    "reason", "guardian", "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic"
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
