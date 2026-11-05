"""Generate grouped, manifest-driven training smoke fixtures, with no downloads.

Training data here checks pipeline roundtrip only; NOT accuracy or generalization.
"""
from pathlib import Path
import json

def main():
    root=Path("native-prototype/training/smoke-data")
    root.mkdir(parents=True,exist_ok=True)
    records=[]
    for split,indices in (("train",range(24)),("validation",range(24,30))):
        for i in indices:
            for codec,text in (
                ("utf-8",f"日本語の文字列とUTF8データ {i}. 日本語の文字列とUTF8データ。"),
                ("cp1252",f"café crème façade déjà vu numéro {i}. café crème."),
            ):
                filename=f"{split}-{codec}-{i}.dat"
                (root/filename).write_bytes(text.encode(codec))
                records.append({"path":filename,"encoding":codec,
                    "group":f"{split}-{codec}-{i}","split":split})
    (root/"manifest.jsonl").write_text(
        "".join(json.dumps(r,ensure_ascii=False)+"\n" for r in records),
        encoding="utf-8")
    print("TRAINER_SMOKE_FIXTURES_READY",len(records),flush=True)

if __name__=="__main__":main()
