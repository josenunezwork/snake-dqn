import re,os,glob
REPO="/Users/josenunez/Projects/ml/snake-dqn"; ART="/Users/josenunez/Projects/ml/snake-dqn-artifacts"
notes=[f for f in glob.glob("*.md")]
names=set(notes)
for f in sorted(notes):
    txt=open(f).read()
    # links
    bad=[l for l in re.findall(r'\]\(([^)#]+)(?:#[^)]*)?\)',txt) if not l.startswith("http") and l not in names and not os.path.exists(l)]
    # citations
    miss=[];n=0
    for m in re.finditer(r'\(src: ([^)]+)\)',txt):
        for part in m.group(1).split(";"):
            part=part.strip()
            if part.startswith("git "): continue
            mm=re.match(r'`?([^\s:`]+)`?(?::(\d+))?',part)
            if not mm: continue
            p=mm.group(1); n+=1
            ap = p if p.startswith("/") else (ART+"/"+p if p.startswith("big-lever/") else REPO+"/"+p)
            if not os.path.exists(ap): miss.append(p)
            elif mm.group(2):
                try:
                    L=len(open(ap,errors="replace").read().split("\n"))
                    if int(mm.group(2))>L: miss.append(f"{p}:{mm.group(2)}>EOF")
                except: pass
    live=len(re.findall(r'\b(is running|are running|is pending|currently running)\b',txt))
    print(f"{f:32s} cites={n} badcites={len(miss)} badlinks={bad[:6]} liveness={live}")
    for x in sorted(set(miss))[:12]: print("   MISS",x)
