from pathlib import Path
import sys

def borrar_con_v2(ruta: str):
    base = Path(ruta)
    if not base.exists():
        raise FileNotFoundError(f"La ruta no existe: {ruta}")

    borrados = 0
    for f in base.rglob("*"):
        if f.is_file() and "V2" in f.name and "SP50" in f.name:
            f.unlink()
            borrados += 1

    print(f"Ficheros borrados que contenían 'V2': {borrados}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python borrar_v2.py /home/eder/projects/master-thesis/results/full_optimal/rows/")
        sys.exit(1)

    ruta = sys.argv[1]
    borrar_con_v2(ruta)