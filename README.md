# Hands-on 1 — MLflow

Implementasi praktik Hands-on 1 pada slide: tracking server lokal, basic logging, logging PyTorch per epoch, autologging, hyperparameter tuning dengan Optuna nested runs, perbandingan run, Model Registry, alias `challenger`/`champion`, dan prediksi dari URI registry.

Tugas CNN pada slide 88 tidak termasuk proyek ini.

## Struktur

```text
src/basic_tracking.py  logging manual Random Forest
src/deep_learning.py   logging loss/F1/AUC/lr per epoch
src/autologging.py     sklearn autolog + konteks manual
src/tune.py            Optuna dan nested child runs
src/compare_runs.py    pencarian kandidat terbaik
src/register.py        registry, alias, load, prediksi
start_server.sh        server MLflow SQLite lokal
run_pipeline.sh        menjalankan seluruh praktik
```

## Setup

Direkomendasikan Python 3.11–3.13.

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pastikan dependency utama dapat diimpor:

```bash
python -c "import mlflow, optuna, torch, sklearn; print(mlflow.__version__)"
```

## 1. Tracking server

Jalankan dari direktori `handson-1` pada terminal pertama:

```bash
source .venv/bin/activate
./start_server.sh
```

Server memakai SQLite `mlflow.db`, artifact store `mlartifacts/`, dan UI <http://127.0.0.1:5000>. Untuk port lain:

```bash
MLFLOW_PORT=5002 ./start_server.sh
```

Kemudian set URI yang sama pada terminal eksperimen:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

Untuk eksekusi lokal tanpa proses server, MLflow dapat diarahkan langsung ke SQLite. Tracking dan Model Registry tetap berfungsi, tetapi UI tidak aktif:

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

## 2. Basic logging

```bash
python -m src.basic_tracking
```

Run `rf-baseline` mencatat parameter Random Forest, hash dan objek dataset, F1, ROC-AUC, `requirements.txt`, signature, contoh input, serta model.

## 3. Deep learning: logging per epoch

```bash
python -m src.deep_learning --epochs 15 --batch-size 64 --lr 0.0003 --dropout 0.2
```

Model MLP PyTorch berjalan pada CPU. Setiap epoch mencatat `train_loss`, `val_loss`, `val_f1`, `val_auc`, dan learning rate. Best checkpoint serta model dengan signature disimpan sebagai artifact. Standardisasi berada di dalam model agar model yang dimuat kembali menerima fitur mentah.

## 4. Autologging

```bash
python -m src.autologging
```

`mlflow.sklearn.autolog()` mencatat informasi estimator secara otomatis. Hash data, tag konteks, metrik validasi domain, dan model bernama konsisten tetap dicatat manual.

## 5. Optuna nested runs

```bash
python -m src.tune --trials 20
```

Satu parent run merangkum sweep. Setiap trial menjadi child run dan mencatat parameter Random Forest serta validation ROC-AUC. Search space dan model terbaik tersimpan pada parent run.

## 6. Bandingkan, register, dan prediksi

```bash
python -m src.compare_runs
python -m src.register
```

Run dengan `val_auc` tertinggi dipilih secara programatik. Model diregistrasikan sebagai `handson-1-breast-cancer-clf`, mendapat alias `challenger` dan `champion`, kemudian dimuat melalui:

```text
models:/handson-1-breast-cancer-clf@champion
```

## Menjalankan seluruh pipeline

Setelah server aktif:

```bash
./run_pipeline.sh
```

Jumlah epoch dan trial dapat diubah untuk debugging:

```bash
EPOCHS=2 TRIALS=3 ./run_pipeline.sh
```

Gunakan nilai default 15 epoch dan 20 trial untuk praktik utama.

## Pemeriksaan

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

Di MLflow UI, pilih beberapa run pada experiment `handson-1-breast-cancer`, lalu buka **Compare** untuk membandingkan parameter, metrik, dan learning curve.
