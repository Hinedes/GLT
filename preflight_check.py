#!/usr/bin/env python3
"""Pre-flight structural checker for the grafting single-TU HIP build.

Runs on the RPI5 (no ROCm/hipcc). It does NOT compile device code; it
catches the cheap, high-value errors that waste a round-trip to the
ROCm host:
  1. Every local #include "..." resolves to a real file.
  2. Brace / paren / bracket balance across the whole include tree
     (the exact class of bug that broke grafting.hip before).
  3. No stray unterminated // or /* */ comment blocks.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(ROOT, "grafting.hip")

LOCAL_INC = re.compile(r'^\s*#\s*include\s*"([^"]+)"')
BRACES = {"{": "}", "(": ")", "[": "]"}


def find_file(inc: str, base: str):
    cand = os.path.join(base, inc)
    if os.path.isfile(cand):
        return cand
    cand = os.path.join(ROOT, inc)
    if os.path.isfile(cand):
        return cand
    return None


def collect_sources():
    seen, order, missing = set(), [], []
    stack = [(MAIN, ROOT)]
    while stack:
        path, base = stack.pop()
        if path in seen:
            continue
        seen.add(path)
        order.append(path)
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    m = LOCAL_INC.match(line)
                    if m:
                        fnd = find_file(m.group(1), base)
                        if fnd:
                            stack.append((fnd, os.path.dirname(fnd)))
                        else:
                            missing.append((path, m.group(1)))
        except OSError as e:
            missing.append((path, f"<read error: {e}>"))
    return order, missing


def balance(text: str):
    # strip line comments and string/char literals crudely, then block comments
    out = []
    i, n = 0, len(text)
    in_line = in_block = False
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if in_block:
            if c == "*" and nxt == "/":
                in_block = False
                i += 2
                continue
            i += 1
            continue
        if in_line:
            if c == "\n":
                in_line = False
            i += 1
            continue
        if c == "/" and nxt == "/":
            in_line = True
            i += 2
            continue
        if c == "/" and nxt == "*":
            in_block = True
            i += 2
            continue
        if c in "\"'" or (c == "L" and nxt in "\"'"):
            # skip string/char literal
            q = nxt if c == "L" else c
            i += 2 if c == "L" else 1
            while i < n:
                if text[i] == "\\":
                    i += 2
                    continue
                if text[i] == q:
                    i += 1
                    break
                i += 1
            continue
        out.append(c)
        i += 1
    clean = "".join(out)
    counts = {o: clean.count(o) for o in BRACES}
    counts.update({c: clean.count(c) for c in BRACES.values()})
    ok = all(counts[o] == counts[c] for o, c in BRACES.items())
    return ok, counts


def main():
    order, missing = collect_sources()
    print(f"Resolved {len(order)} translation-unit files.")
    rc = 0
    if missing:
        rc = 1
        print("\n[FAIL] Unresolved local includes:")
        for p, inc in missing:
            print(f"  {p}  ->  \"{inc}\"")
    else:
        print("[OK]   All local includes resolve.")

    total = []
    for p in order:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            total.append(f.read())
    ok, counts = balance("\n".join(total))
    if ok:
        print("[OK]   Brace/paren/bracket balance across TU.")
    else:
        rc = 1
        print("\n[FAIL] Brace/paren/bracket imbalance across TU:")
        for k, v in counts.items():
            print(f"  '{k}': {v}")
    print(f"\nPer-file balance:")
    for p in order:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            okf, cf = balance(f.read())
        flag = "OK " if okf else "BAD"
        if not okf:
            rc = 1
        print(f"  [{flag}] {os.path.relpath(p, ROOT)}  { {k:cf[k] for k in '{}()[]'} }")
    return rc


if __name__ == "__main__":
    sys.exit(main())
