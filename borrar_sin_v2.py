from pathlib import Path
import sys

def borrar_sin_v2(ruta: str):
    base = Path(ruta)
    if not base.exists():
        raise FileNotFoundError(f"La ruta no existe: {ruta}")

    borrados = 0
    for f in base.rglob("*"):
        if f.is_file() and "FX" not in f.name:
            f.unlink()
            borrados += 1

    print(f"Ficheros borrados que NO contenían 'V2': {borrados}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python borrar_sin_v2.py /mnt/c/Users/edert/Downloads/master-thesis/master-thesis/results/full_optimal/rows/")
        sys.exit(1)

    ruta = sys.argv[1]
    borrar_sin_v2(ruta)