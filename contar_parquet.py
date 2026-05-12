from pathlib import Path
import sys

def contar_parquet(ruta: str) -> int:
    base = Path(ruta)
    if not base.exists():
        raise FileNotFoundError(f"La ruta no existe: {ruta}")

    return sum(1 for f in base.rglob("*.parquet") if f.is_file())

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python contar_parquet.py /mnt/c/Users/edert/Downloads/master-thesis/master-thesis/results/full_optimal/rows/")
        sys.exit(1)

    ruta = sys.argv[1]
    total = contar_parquet(ruta)
    print(f"Ficheros .parquet encontrados: {total}")