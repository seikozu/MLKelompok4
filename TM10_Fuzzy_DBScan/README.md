# TM10 Fuzzy C-Means & DBSCAN

Tiap dataset sekarang berupa 1 folder berisi **1 file kode + dataset**.

```text
TM10_Fuzzy_DBScan/
├── adult/
│   ├── adult_clustering.py        # seluruh proses adult
│   └── adult.csv
├── Mall_Customers/
│   ├── mall_clustering.py         # seluruh proses mall
│   └── Mall_Customers.csv
└── requirements.txt
```

## Menjalankan
```bash
pip install -r requirements.txt
python adult/adult_clustering.py
python Mall_Customers/mall_clustering.py
```
Bisa dijalankan dari folder mana saja. Hasil (grafik, metrics JSON, laporan .md)
otomatis dibuat di `outputs/` di dalam masing-masing folder.

Isi tiap file kode: load data → preprocessing → K-Means / Fuzzy C-Means / DBSCAN →
evaluasi (Silhouette, DBI, CH) → visualisasi → laporan → tabel perbandingan.
