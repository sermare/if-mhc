keep = {"A": 180, "B": 99, "D": 115, "E": 120}   # MHC a1a2 + b2m + TCR Va + Vb
out = []
present = {c: [] for c in keep}
for l in open("inputs/2P5E.pdb"):
    if l[:6] in ("ATOM  ", "TER   "):
        c = l[21] if len(l) > 21 else ""
        if c in keep:
            try:
                rn = int(l[22:26])
            except ValueError:
                continue
            if rn <= keep[c]:
                out.append(l)
                if l[:6] == "ATOM  ":
                    present[c].append(rn)
out.append("END\n")
open("inputs/2P5E_trimmed.pdb", "w").writelines(out)
rng = {c: (min(v), max(v)) for c, v in present.items() if v}
print("trimmed ranges:", rng, "| atoms:", sum(1 for x in out if x[:4] == "ATOM"))
