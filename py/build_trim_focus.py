# Trim 6AMU/6AM5 to MHC a1a2 + b2m + TCR Va/Vb + peptide; emit RFdiffusion partial-diffusion contig.
keep={"A":180,"B":100,"C":10,"D":115,"E":120}
out=open("inputs/focus_6am/contigs_focus.txt","w")
for pid in ["6AMU","6AM5"]:
    lines=[]; present={c:[] for c in keep}
    for l in open(f"inputs/focus_6am/{pid}.pdb"):
        if l[:6] in ("ATOM  ","TER   "):
            c=l[21] if len(l)>21 else ""
            if c in keep:
                try: rn=int(l[22:26])
                except: continue
                if rn<=keep[c]:
                    lines.append(l)
                    if l[:6]=="ATOM  ": present[c].append(rn)
    lines.append("END\n")
    open(f"inputs/focus_6am/{pid}_trim.pdb","w").writelines(lines)
    rng={c:(min(v),max(v)) for c,v in present.items() if v}
    contig=" ".join(f"{c}{rng[c][0]}-{rng[c][1]}/0" for c in ["A","B","C","D","E"] if c in rng).rstrip("/0").rstrip()
    # ensure last has no trailing /0
    seg=[f"{c}{rng[c][0]}-{rng[c][1]}" for c in ["A","B","C","D","E"] if c in rng]
    contig="/0 ".join(seg)
    out.write(f"{pid}\t{contig}\n")
    print(f"{pid}: trimmed {sum(1 for x in lines if x[:4]=='ATOM')} atoms | contig=[{contig}]")
out.close()
