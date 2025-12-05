#!/usr/bin/env python3
import sys, math
import matplotlib.pyplot as plt

SOMOS_PHASE1_INT = 17
SOMOS_PHASE2 = 779731
FORCE_INT = '--int' in sys.argv
PLOT = '--plot' in sys.argv
NE = 50
NS = 0
SHOW_END = True
if '--test1' in sys.argv:
    NS = 0
    NE = 30
    FORCE_INT = True
elif '--test2' in sys.argv:
    NS = 0
    NE = 100
    FORCE_INT = True
elif '--test3' in sys.argv:
    NS = 0
    NE = 2500
    PLOT = True
    FORCE_INT = True
elif '--test4' in sys.argv:
    NS = 0
    NE = 500000
    PLOT = True
    FORCE_INT = True
elif '--test5' in sys.argv:
    NS = 779731 - 100
    NE = 779731 + 100
    PLOT = True
    FORCE_INT = True
elif '--test6' in sys.argv:
    NS = 0
    NE = 779731 + 50000
    PLOT = True
    FORCE_INT = True

elif '--test7' in sys.argv:
    NS = 0
    NE = 779731 + 50000
    PLOT = True
    FORCE_INT = False
elif '--test8' in sys.argv:
    NS = 0
    NE = 2000000
    PLOT = True
    FORCE_INT = True
elif '--test9' in sys.argv:
    NS = 0
    NE = 2000000
    PLOT = True
    FORCE_INT = False
elif '--test10' in sys.argv:
    NS = 0
    NE = 4000000
    PLOT = True
    FORCE_INT = False

elif '--end' in sys.argv:
    NE = int(sys.argv[-1])
elif '--start-end' in sys.argv:
    NS = int(sys.argv[-2])
    NE = int(sys.argv[-1])


def plot(x, y, title='none', xlabel='value', ylabel='number', bars=False):
    if FORCE_INT: title += ' (mode=int)'
    else: title += ' (mode=float)'
    if '--verbose' in sys.argv: print('ploting data size:', len(x), title)
    if len(x) <= 1:
        print('data size too small', len(x))
        return
    if len(x) >= 30000:
        print('matplotlib is too slow for this data size:', len(x))
        return
    plt.figure(figsize=(12, 10))
    if bars:
        plt.bar(x,y)
    else:
        plt.plot(x,y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

VALUES = []
def somos_8_sequence(init=1, num_terms=256):
    """Computes the Somos-8 sequence up to a specified number of terms."""
    if num_terms < 8: return [init] * num_terms
    s = [init] * 8
    terms = [init] * 8
    # note: the sequence is 1-indexed for the mathematical definition,but Python lists are 0-indexed. 
    # s[0] corresponds to s_1, s[7] corresponds to s_8.
    for i in range(8, num_terms):
        # Calculate the next term using the recurrence relation:
        # s_n = (s_{n-1}s_{n-7} + s_{n-2}s_{n-6} + s_{n-3}s_{n-5} + s_{n-4}^2) / s_{n-8}
        # In Python indices (where n = i + 1):
        # s[i] = (s[i-1]*s[i-7] + s[i-2]*s[i-6] + s[i-3]*s[i-5] + s[i-4]**2) / s[i-8]
        #next_term = (s[i-1] * s[i-7] + s[i-2] * s[i-6] + s[i-3] * s[i-5] + s[i-4]**2) / s[i-8]
        a = (s[i-1] * s[i-7] + s[i-2] * s[i-6] + s[i-3] * s[i-5] + s[i-4]**2)
        b = s[i-8]
        assert int(a) == a and int(b) == b
        next_term = a / b
        terms.append((int(a),int(b)))
        if int(next_term) != next_term:
            s.append(next_term)
            VALUES.append(math.log(next_term))
            break
        else:
            if FORCE_INT: s.append(int(next_term))
            else: s.append(next_term)

    return s, terms

# --- Example Usage ---
strange = []  ## this contains >= 19th
stranger = [] ## less than 17th step is very rare at first, and only in float mode.
prev = 0
phase_trans = {}
ending_digits = {}
A_digits = {}
B_digits = {}

x = []
y = []
for i in range(NS, NE):
    sequence, terms = somos_8_sequence(i+1)
    l = len(sequence)
    if l not in phase_trans:
        phase_trans[l] = {'total':0}
    s = phase_trans[l]
    s['total'] += 1
    s['real'] = sequence[-1]
    x.append(i+1)
    y.append(len(sequence)-1)

    if NE-NS <= 100: print(sequence, 'initializer:', i+1, 'breaks-at:', len(sequence)-1)
    if i+2 > NE and SHOW_END:
        print('end:', sequence, 'initializer:', i+1, 'breaks-at:', len(sequence)-1)

    ss = str(sequence[-1])
    edigit = int(ss[-1])
    if edigit not in ending_digits:
        ending_digits[ edigit ] = i+1


    if len(sequence) != 18:
        if edigit not in A_digits:
            A_digits[ edigit ] = i+1

        delta = i - prev
        if len(sequence) >= 20:  ## this first happens at Somos(91)
            #assert i+1 < SOMOS_PHASE2 (not valid) breaks at Somos(780234)
            assert i+1 >= 91
            strange.append((i, len(sequence), sequence[-1], terms[-1]))

            ## always true? NO
            #if ss.endswith(('1', '9')):  ## this only happens after Somos(13559)
            #    assert i+1 >= 13559
            #if len(sequence) >= 21:  ## this first happens at Somos(2275)
            #    assert i+1 >= 2275
            #if len(sequence) >= 22:  ## this first happens at Somos(138775)
            #    assert i+1 >= 138775


        prev = i
        if len(sequence) < 18:
            if edigit not in B_digits:
                B_digits[ edigit ] = i+1

            if '--verbose' in sys.argv: print('STRANGER %sth'%(len(sequence)-1), sequence, i)
            ## this first happens at Somos8(779731), with breaking on the 16th step and locked into the even phase.
            assert i+1 >= SOMOS_PHASE2
            stranger.append((i, len(sequence), sequence[-1], terms[-1]))
            strange.append((i, len(sequence), sequence[-1], terms[-1]))

    else:
        if '--debug' in sys.argv:
            print('NormalSomos init=%s bindex=%s seq=%s' % (i, len(sequence)-1, sequence))
            print(terms)



if PLOT:
    vx = list(range(len(VALUES)))
    plot(vx,VALUES, title='Somos8(%s-%s) - Values' % (NS+1, NE), xlabel='Somos8(N)', ylabel='number log scale', bars=True)
    plot(x,y, title='Somos8(%s-%s) - Breaking Points' % (NS+1, NE), xlabel='Somos8(N)', ylabel='breaking point index', bars=True)

print('total-strange:', len(strange))
if not len(strange): print('note: the first strange happens at N=91')
prev = NS - 1
x = []; x2 = []; x3 = []
y = []; y2 = []; y3 = []
for s in strange:
    delta = s[0] - prev
    init,lenseq,real,term = s
    if '--verbose' in sys.argv or len(strange) < 10:
        print('\t'*(lenseq-20), 'integer-break-point:', lenseq-1, 'distance:', delta, 'init:', init, 'real:', real, 'term:',term)
    x.append(init)
    y.append(delta)
    prev = s[0]
    if lenseq >= 21:
        x2.append(init)
        y2.append(lenseq-1)
        x3.append(init)
        y3.append(delta)

if PLOT:
    plot(x,y, title='Somos8(%s-%s) - Strange Group Deltas' % (NS+1, NE), xlabel='Somos(N)', ylabel='delta spacing of N', bars=False)
    y = [math.log(v) for v in y]
    plot(x,y, title='Somos8(%s-%s) - Strange Group Deltas [log]' % (NS+1, NE), xlabel='Somos(N)', ylabel='delta spacing of N (log scale)', bars=False)

plot(x2,y2, title='Somos8(%s-%s) - Strange Group: of integer-break-point over 20' % (NS+1, NE), xlabel='Somos(N)', ylabel='integer-break-point', bars=False)
plot(x3,y3, title='Somos8(%s-%s) - Strange Deltas: of integer-break-point over 20' % (NS+1, NE), xlabel='Somos(N)', ylabel='delta of N-1 and N', bars=False)

if len(strange):
    print('strange-ratio:', len(strange) / NE)
    print(NE / len(strange))

p = 1
phases = list(phase_trans.keys())
phases.sort()
x = []
y = []
for phase in phases:
    print('GROUP:', p, '  integer-break-point:', phase-1)
    p += 1
    s = phase_trans[phase]
    print('    total=', s['total'], 'percentage=', s['total']/NE )
    if '--verbose' in sys.argv: print('    first-real=', s['real'])
    x.append(phase-1)
    y.append(math.log(s['total']))

plot(x,y, title='Somos8(%s-%s) - Breaking Points Ratios' % (NS+1, NE), xlabel='integer-to-real phase index', ylabel='number of solutions (log scale)', bars=True)
print('somos8-search-range: 1 to', NE)
print('total-stranger:', len(stranger))

if '--verbose' in sys.argv: 
    for s in stranger:
        print(s[0],end=',')

if stranger:
    print('first stranger:', stranger[0])
    print('last stranger:', stranger[-1])
else:
    print('note: strangers start at Somos8(779731)')

emap = {'ending-digits-all': ending_digits, 'ending-digits-strange': A_digits, 'ending-digits-stranger':B_digits}
for name in emap:
    endict = emap[name]
    print(name, endict)
    e = list(endict.keys())
    e.sort(); x=[]; y=[]; y2=[]
    #for num in e:
    for i in range(10):
        if i in endict:
            v = endict[i]
        else:
            v = 0.0
        x.append(i)
        if v==0: y.append(v)  ## fixes math domain error
        else: y.append(math.log(v))
        y2.append(v)
    plot(x,y, title='Somos8(%s-%s) - Order of %s' % (NS+1, NE, name), xlabel='ending digit', ylabel='breaks at (log scale)', bars=True)
    plot(x,y2, title='Somos8(%s-%s) - Order of %s' % (NS+1, NE, name), xlabel='ending digit', ylabel='breaks at', bars=False)
