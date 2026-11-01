"""Roda as etapas do projeto em ordem.

    python run_pipeline.py                 # tudo que não precisa de LLM
    python run_pipeline.py --from index    # recomeça a partir de uma etapa
    python run_pipeline.py --llm           # só as etapas com LLM (precisa de chave no .env)

As etapas com LLM ficam separadas porque têm custo e são mais lentas. As
chamadas ficam em cache (data/cache/), então reexecutar --llm depois de
revisar o gold não gera custo: apenas recalcula as métricas.
"""
import argparse
import subprocess
import sys
import time

PY = sys.executable

STEPS = [
    ("download", [PY, "scripts/download_data.py"]),
    ("prep", [PY, "-m", "src.data_prep"]),
    ("classifier", [PY, "-m", "src.classifier"]),
    ("index", [PY, "-m", "src.index"]),
    ("search_eval", [PY, "-m", "src.eval_search"]),
    ("gold_review", [PY, "-m", "src.eval_gold"]),
    ("report", [PY, "-m", "src.report"]),
]

LLM_STEPS = [
    ("llm_extract", [PY, "-m", "src.llm_extract", "--gold"]),
    ("rag_eval", [PY, "-m", "src.eval_rag"]),
    ("gold_review", [PY, "-m", "src.eval_gold"]),
    ("report", [PY, "-m", "src.report"]),
]


def run(steps):
    for name, cmd in steps:
        print(f"\n=== {name} " + "=" * (60 - len(name)))
        t0 = time.time()
        if subprocess.run(cmd).returncode != 0:
            sys.exit(f"Etapa '{name}' falhou. Corrija e rode de novo.")
        print(f"({time.time() - t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", choices=[s for s, _ in STEPS], default="download")
    ap.add_argument("--llm", action="store_true", help="roda só as etapas que usam LLM")
    args = ap.parse_args()

    if args.llm:
        from src.llm import get_client
        client = get_client()
        if client is None:
            sys.exit("Nenhum LLM configurado. Copie .env.example para .env e preencha a chave.")
        print(f"LLM: {client}")
        run(LLM_STEPS)
        return

    names = [s for s, _ in STEPS]
    run(STEPS[names.index(args.start):])


if __name__ == "__main__":
    main()
