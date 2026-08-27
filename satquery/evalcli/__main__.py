import argparse
import json

from satquery.evalcli.smoke import run_dummy_query


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m satquery.evalcli")
    parser.add_argument("--question", required=True)
    parser.add_argument("--index", choices=["ALPHA", "BETA"], default="ALPHA")
    parser.add_argument("--scale", type=float, default=0.5)
    args = parser.parse_args()
    trace = run_dummy_query(args.question, {"index": args.index, "scale": args.scale})
    print(json.dumps(trace, indent=2))


if __name__ == "__main__":
    main()
