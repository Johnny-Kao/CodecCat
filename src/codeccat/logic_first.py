"""Logic-first validity evidence; no guessed encoding from validity alone.

Pure byte-level structural validation on complete provided buffers.
Truncated prefixes return UNKNOWN instead of INVALID where applicable.
"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Evidence:
    encoding: str
    status: str  # valid / invalid / incomplete
    consumed: int

def _validate(data: bytes, encoding: str) -> Evidence:
    if encoding == "utf-8":
        try:
            data.decode("utf-8","strict")
            return Evidence(encoding,"valid",len(data))
        except UnicodeDecodeError as e:
            return Evidence(encoding,"incomplete" if e.reason=="unexpected end of data" and e.end==len(data) else "invalid",e.start)
    state=0
    for i, v in enumerate(data):
        if encoding=="shift-jis":
            if state==0:
                if v<=0x7f or 0xa1<=v<=0xdf:continue
                if 0x81<=v<=0x9f or 0xe0<=v<=0xfc:state=1;continue
                return Evidence(encoding,"invalid",i)
            if 0x40<=v<=0x7e or 0x80<=v<=0xfc:
                state=0;continue
            return Evidence(encoding,"invalid",i)
        if encoding=="euc-jp":
            if state==0:
                if v<=0x7f:continue
                if v==0x8e:state=1;continue
                if v==0x8f:state=2;continue
                if 0xa1<=v<=0xfe:state=4;continue
                return Evidence(encoding,"invalid",i)
            if state==1:
                if 0xa1<=v<=0xdf:state=0;continue
                return Evidence(encoding,"invalid",i)
            if state in (2,3,4):
                if 0xa1<=v<=0xfe:state=3 if state==2 else 0;continue
                return Evidence(encoding,"invalid",i)
        if encoding=="big5":
            if state==0:
                if v<=0x7f:continue
                if 0x81<=v<=0xfe:state=1;continue
                return Evidence(encoding,"invalid",i)
            if 0x40<=v<=0x7e or 0xa1<=v<=0xfe:state=0;continue
            return Evidence(encoding,"invalid",i)
        if encoding=="gb18030":
            if state==0:
                if v<=0x7f:continue
                if 0x81<=v<=0xfe:state=1;continue
                return Evidence(encoding,"invalid",i)
            if state==1:
                if 0x30<=v<=0x39:state=2;continue
                if 0x40<=v<=0x7e or 0x80<=v<=0xfe:state=0;continue
                return Evidence(encoding,"invalid",i)
            if state==2:
                if 0x81<=v<=0xfe:state=3;continue
                return Evidence(encoding,"invalid",i)
            if state==3:
                if 0x30<=v<=0x39:state=0;continue
                return Evidence(encoding,"invalid",i)
    return Evidence(encoding,"incomplete" if state else "valid",len(data))

SUPPORTED=("utf-8","shift-jis","euc-jp","big5","gb18030")
def evaluate(data: bytes) -> dict[str,Evidence]:
    return {e:_validate(data,e) for e in SUPPORTED}
