"""Independent re-implementation of Tsiokos's discharge diff audit for the R44 Lean development.

The rule audited (from lean/R44/scripts/discharge_diff_audit.sh of ioannist/six-birds-tiles): every one of the 67
files in lean/R44/R44/Proved must differ from its text in the latest external delivery archive only inside hunks
marked `-- [compile-fix ...]`, and every deleted run of delivery lines must sit next to a marked block.

This script does not import or run that script.  It takes two directories:
    ARCHIVES  the seven delivery ZIPs under proof/external_lean at commit 838bca5
    PROVED    the 67 files of lean/R44/R44/Proved at commit d53d224
and, beyond the rule above, reports what each marked hunk touches: declaration headers (theorem, lemma, def,
structure, instance, axiom, abbrev, opaque), and any added `sorry`, `admit`, `axiom` or `native_decide`.

Usage: python3 -I discharge_audit_indep.py ARCHIVES PROVED
"""
import difflib
import re
import sys
import zipfile
from pathlib import Path

ARCH, PROVED = Path(sys.argv[1]), Path(sys.argv[2])
# delivery order, oldest first; a file is compared with the newest delivery that contains it
ORDER = ['r44_F1_exchange2_discharge.zip', 'r44_F2_exchange3_discharge.zip', 'r44_F3_exchange4_discharge.zip',
         'r44_F4_exchange5_discharge.zip', 'r44_exchange6_bridges.zip', 'r44_exchange7_native_interpretation.zip',
         'r44_F1_complete_dihedral_list_draft.zip']
PREFIX = 'lean/R44/R44/Discharge/'
DECL = re.compile(r'^\s*(?:@\[[^\]]*\]\s*)?(?:private\s+|protected\s+|noncomputable\s+)*'
                  r'(theorem|lemma|def|structure|instance|axiom|abbrev|opaque|inductive|class)\b')
RISKY = re.compile(r'\b(sorry|admit|axiom|native_decide)\b')
NAME = re.compile(DECL.pattern + r'\s+(\S+)')


def statements(lines):
    """name -> statement text (from the header to the first `:=`, `where` or `|`), comments and whitespace removed"""
    out = {}
    for i, s in enumerate(lines):
        m = NAME.match(s)
        if not m:
            continue
        kind, name = m.group(1), m.group(2)
        body = []
        for t in lines[i:i + 60]:
            t = t.split('--')[0]
            body.append(t)
            if ':=' in t or re.search(r'\bwhere\b', t) or (kind == 'inductive' and t.strip().startswith('|')):
                break
        text = ' '.join(' '.join(body).split())
        text = re.split(r':=|\bwhere\b', text)[0]
        text = re.sub(r'^(@\[[^\]]*\]\s*)?((private|protected|noncomputable)\s+)*', '', text)
        out[name.strip('«»')] = (kind, text, i + 1)
    return out

latest = {}
for name in ORDER:
    with zipfile.ZipFile(ARCH / name) as z:
        for m in z.namelist():
            if m.startswith(PREFIX) and m.endswith('.lean') and '/' not in m[len(PREFIX):]:
                latest[m[len(PREFIX):]] = (name, z.read(m).decode())

files = sorted(p.name for p in PROVED.glob('*.lean'))
print('files in Proved:', len(files), '; files in some delivery:', len(latest))
assert set(files) == set(latest), (sorted(set(files) - set(latest)), sorted(set(latest) - set(files)))


def marked_lines(lines, fname):
    """True on every line inside or on a `-- [compile-fix begin: ...]` ... `-- [compile-fix end]` block"""
    inside, out = False, []
    for i, s in enumerate(lines):
        b, e = '-- [compile-fix begin:' in s, '-- [compile-fix end]' in s
        if b:
            assert not inside, '%s:%d nested block' % (fname, i + 1)
            inside = True
        out.append(inside or b or e)
        if e:
            assert inside, '%s:%d end without begin' % (fname, i + 1)
            inside = False
    assert not inside, '%s unterminated block' % fname
    return out


def is_marked(line, cov):
    return cov or '-- [compile-fix' in line


npass, tot_changed, decl_hunks, risky, stmt = 0, 0, [], [], []
for f in files:
    arch, old = latest[f]
    old = old.splitlines()
    new = (PROVED / f).read_text().splitlines()
    cov = marked_lines(new, f)
    bad = []
    nchanged = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=old, b=new, autojunk=False).get_opcodes():
        if tag == 'equal':
            continue
        if tag in ('replace', 'insert'):
            nchanged += sum(1 for j in range(j1, j2) if new[j].strip())
            # a hunk counts as marked when any of its new lines is marked
            if not any(is_marked(new[j], cov[j]) for j in range(j1, j2)):
                bad += [(j + 1, new[j].strip()) for j in range(j1, j2) if new[j].strip()]
        else:  # delete
            if not ((j1 > 0 and cov[j1 - 1]) or (j1 < len(new) and cov[j1])):
                bad.append(('delivery %d-%d' % (i1 + 1, i2), 'deleted outside a block'))
        touched = [old[i] for i in range(i1, i2)] + [new[j] for j in range(j1, j2)]
        if any(DECL.match(s) for s in touched):
            decl_hunks.append((f, j1 + 1, [s.strip()[:100] for s in touched if DECL.match(s)]))
        for j in range(j1, j2):
            code = new[j].split('--')[0]
            if RISKY.search(code):
                risky.append((f, j + 1, new[j].strip()[:100]))
    tot_changed += nchanged
    so, sn = statements(old), statements(new)
    for name in sorted(so):
        if name not in sn:
            stmt.append((f, name, 'removed', so[name][1][:160], ''))
        elif so[name][1] != sn[name][1]:
            stmt.append((f, name, 'changed', so[name][1][:160], sn[name][1][:160]))
    for name in sorted(set(sn) - set(so)):
        stmt.append((f, name, 'added', '', sn[name][1][:160]))
    if bad:
        print('FAIL', f, 'against', arch, bad[:5])
    else:
        npass += 1
        print('PASS %-40s %-45s changed lines %d' % (f, arch, nchanged))
print('\naudit: %d/%d files pass; %d changed non-blank lines in all' % (npass, len(files), tot_changed))
print('\nhunks that touch a declaration header (%d):' % len(decl_hunks))
for f, j, ds in decl_hunks:
    print('  %s:%d' % (f, j), ds)
print('\nadded or changed lines with sorry/admit/axiom/native_decide outside comments (%d):' % len(risky))
for r in risky:
    print('  %s:%d %s' % r)
print('\ndeclarations whose statement text differs, ignoring comments, whitespace and private/protected (%d):'
      % len(stmt))
for f, name, how, a, b in stmt:
    print('  %s %s [%s]' % (f, name, how))
    if a:
        print('     delivery:', a)
    if b:
        print('     build:   ', b)
assert npass == len(files)
