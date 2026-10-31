"""CodecCat CLI: detect or strictly decode a local byte stream."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

MAX_BYTES = 16 * 1024 * 1024


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Detect text encoding with CodecCat's bundled alpha model")
    p.add_argument("file", nargs="?", default="-", help="input file or - for stdin")
    p.add_argument("--decode", action="store_true", help="write strictly decoded text to stdout")
    p.add_argument("--json", action="store_true", help="write structured detection metadata")
    p.add_argument("--bom-policy", choices=("strip", "preserve"), default="strip")
    args = p.parse_args(argv)
    if args.json and args.decode:
        p.error("--json and --decode are mutually exclusive")
    try:
        if args.file == "-":
            raw = sys.stdin.buffer.read(MAX_BYTES + 1)
        else:
            with Path(args.file).open("rb") as stream:
                raw = stream.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            p.error("input exceeds 16 MiB limit; never silently truncate a stream")
        from .usable import detect
        result = detect(raw, bom_policy=args.bom_policy)
    except (OSError, RuntimeError, ValueError) as exc:
        p.exit(2, f"codeccat: {exc}\n")
    if args.decode:
        if result.text is None:
            print("codeccat: unable to strictly decode input", file=sys.stderr)
            return 2
        sys.stdout.write(result.text)
    elif args.json:
        print(json.dumps({
            "encoding": result.encoding,
            "accepted": result.accepted,
            "ambiguous": result.ambiguous,
            "candidates": result.candidates,
            "bom_present": result.bom_present,
            "reason": result.reason,
        }, ensure_ascii=False))
    else:
        if not result.accepted:
            print("unknown")
            return 2
        print(result.encoding)
        if result.ambiguous:
            print("codeccat: encoding label is ambiguous", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
