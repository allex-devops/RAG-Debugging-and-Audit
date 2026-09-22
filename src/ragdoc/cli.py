import argparse
import tempfile
from pathlib import Path

from ragchat.rag import ConfigError, default_min_score, embedder, llm_from_env
from shared.cache import DiskCache

from .corpus import ANSWERABLE, UNANSWERABLE
from .faults import BY_NAME, FAULTS
from .logs import summarize
from .probes import LIKELY_CAUSES, run_all
from .profile import profile
from .rig import Config, Rig


def build_rig(fault_names: list[str], live: bool, cache: bool = False) -> Rig:
    embed = model = None
    if live:
        llm = llm_from_env()
        if cache:
            llm.cache = DiskCache("ragdoc-live")  # repeated questions come back from disk, not the model
        embed, model = embedder(llm), llm
    # a real embedding model scores much higher than the toy one, so it needs a higher cutoff
    cfg = Config(min_score=default_min_score() if live else 0.25, latency_budget_ms=5000.0 if live else 60.0)
    for name in fault_names:
        BY_NAME[name].configure(cfg)
    kw = {"embed": embed} if embed else {}
    return Rig(cfg, Path(tempfile.mkdtemp()), model=model, **kw)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="ragdoc")
    sub = ap.add_subparsers(dest="cmd", required=True)

    doc = sub.add_parser("doctor", help="run the probes, optionally with a bug planted")
    doc.add_argument("--fault", choices=list(BY_NAME), action="append", default=[])
    doc.add_argument("--live", action="store_true", help="use your configured provider (see .env.example) instead of the built-in toy embedder")
    sub.add_parser("faults", help="list the plantable bugs")
    prof = sub.add_parser("profile", help="split request time into stages")
    prof.add_argument("--fault", choices=list(BY_NAME), action="append", default=[])
    prof.add_argument("--live", action="store_true")
    prof.add_argument("--cache", action="store_true", help="cache model answers on disk (live only)")
    prof.add_argument("--repeat", type=int, default=3, help="times to ask each question")
    sub.add_parser("incident", help="reproduce the embedding-model-swap incident")
    logs = sub.add_parser("logs", help="summarise a request log")
    logs.add_argument("path", type=Path)
    args = ap.parse_args(argv)

    def rig_for(faults: list[str], live: bool, cache: bool = False) -> Rig:
        try:
            return build_rig(faults, live, cache)
        except ConfigError as e:
            ap.error(str(e))

    if args.cmd == "faults":
        for f in FAULTS:
            print(f"{f.name:28} {f.story}\n{'':28} shows up as: {f.symptom}")
    elif args.cmd == "incident":
        from .incident import main as incident

        incident()
    elif args.cmd == "logs":
        print(summarize(args.path).report())
    elif args.cmd == "profile":
        rig = rig_for(args.fault, args.live, args.cache)
        print(profile(rig.ask, [q for q, _, _ in ANSWERABLE] + UNANSWERABLE, repeat=args.repeat).report())
    else:
        rig = rig_for(args.fault, args.live)
        failed = 0
        for r in run_all(rig):
            print(f"{'ok  ' if r.passed else 'FAIL'} {r.name:26} {r.detail}")
            if not r.passed:
                failed += 1
                print(f"     likely: {LIKELY_CAUSES[r.name]}")
        raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
