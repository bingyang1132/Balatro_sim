"""
Multi-round discard simulator with exact inner planning (single-step optimal).

Outer loop (simulation):
- For each trial:
  1) Build deck (52 or 40), draw H cards for the starting hand.
  2) For round t = 0..max_rounds:
       - If current hand already has the target, record success at t (0 means pre-discard).
         Stop the trial.
       - If t == max_rounds, record failure; stop.
       - Else choose a discard plan for THIS round via exact single-step success maximization:
           * Enumerate "dominance-pruned" keep candidates K for the target.
           * For each allowed k in [min_discard_per_round, max_discard_per_round] (bounded by hand size),
             compute EXACT success probability P(K,k) using the remaining deck composition.
           * Pick (K*, k*) with maximal P (tie-break smaller k).
         Execute it: discard the others and draw k* cards uniformly without replacement from the remaining deck.

- Report, for r = min_round..max_round:
    * exact_r[r]: fraction of trials whose earliest success happened at exactly r rounds.
    * cum_r[r]:   fraction of trials whose success happened by or before r rounds.

Notes:
- "Rounds" here means exchange cycles. If min_discard=0, round 0 (no exchange) is considered.
- Inner planning is myopic (maximizes current-round success), but uses EXACT probabilities and accounts for current remaining deck.
- Flush: hypergeometric; Two Pair / Full House: rank-DP; Straight: inclusion–exclusion over top-L windows (default L=3).
- All draws are without replacement.

CLI:
  python balatro_multiround_exact.py --deck 52 --hand-size 8 --trials 20000 \
    --min-discard 0 --max-discard 5 --target Straight
"""
import argparse, math, random
from collections import Counter, defaultdict
from typing import List, Tuple, Dict
Card = Tuple[int,int]
TARGETS = ['All', 'Full House', 'Straight', 'Flush', 'Two Pair']

# ---------- Utilities ----------
def comb(n,k):
    if k<0 or k>n: return 0
    return math.comb(n,k)

def build_deck(deck_type:str)->List[Card]:
    if deck_type=='52':
        ranks=list(range(1,14))
    elif deck_type=='40':
        ranks=[1]+list(range(2,11))
    else:
        raise ValueError
    return [(r,s) for r in ranks for s in range(4)]

def hand_has_target(cards: List[Card], target: str, deck_type: str) -> bool:
    if target == 'Flush':
        return any(c>=5 for c in Counter(s for _,s in cards).values())
    elif target == 'Two Pair':
        rc = Counter(r for r,_ in cards)
        return sum(c//2 for c in rc.values()) >= 2
    elif target == 'Full House':
        rc = Counter(r for r,_ in cards)
        return any(c>=3 for c in rc.values()) and sum(1 for c in rc.values() if c>=2) >= 2
    elif target == 'Straight':
        ranks = set(r for r,_ in cards); ranks_aug=set(ranks)
        if 1 in ranks: ranks_aug.add(14)
        starts = range(1,11) if deck_type=='52' else range(1,7)
        return any(all((a+i) in ranks_aug for i in range(5)) for a in starts)
    else:
        raise ValueError

# ---------- Candidate keep sets (dominance-pruned) ----------
def windows_for_straight(deck_type:str)->List[List[int]]:
    starts = range(1,11) if deck_type=='52' else range(1,7)
    wins=[]
    for a in starts:
        w=[a+i for i in range(5)]
        w=[1 if x==14 else x for x in w]
        wins.append(w)
    return wins

def candidate_keeps_for_straight(cards: List[Card], deck_type: str) -> List[List[int]]:
    rank_to_idx=defaultdict(list)
    for i,(r,_) in enumerate(cards):
        rank_to_idx[r].append(i)
    keeps=[]
    for W in windows_for_straight(deck_type):
        kept=[]
        seen=set()
        for r in W:
            if r in rank_to_idx and r not in seen:
                kept.append(rank_to_idx[r][0]); seen.add(r)
        if kept: keeps.append(sorted(kept))
    # union-of-best-two windows
    scores=[]
    for W in windows_for_straight(deck_type):
        d=len(set(r for r,_ in cards if r in W))
        scores.append((d,W))
    scores.sort(reverse=True)
    if scores and scores[0][0]>0:
        W1=scores[0][1]; W2=scores[1][1] if len(scores)>1 else []
        union=set()
        for i,(r,_) in enumerate(cards):
            if r in W1 or r in W2: union.add(i)
        if union: keeps.append(sorted(list(union)))
    if not keeps: keeps=[[]]
    # dedup
    uniq,seen=[],set()
    for k in keeps:
        t=tuple(k)
        if t not in seen:
            uniq.append(k); seen.add(t)
    return uniq

def candidate_keeps_for_flush(cards: List[Card]) -> List[List[int]]:
    suit_to_idx=defaultdict(list)
    for i,(_,s) in enumerate(cards):
        suit_to_idx[s].append(i)
    keeps=[]
    for s,idxs in suit_to_idx.items():
        keeps.append(sorted(idxs))
        if len(idxs)>=4: keeps.append(sorted(idxs[:4]))
        if len(idxs)>=3: keeps.append(sorted(idxs[:3]))
    if not keeps: keeps=[[]]
    uniq,seen=[],set()
    for k in keeps:
        t=tuple(k)
        if t not in seen:
            uniq.append(k); seen.add(t)
    return uniq

def candidate_keeps_for_two_pair(cards: List[Card]) -> List[List[int]]:
    rc=Counter(r for r,_ in cards)
    rank_to_idx=defaultdict(list)
    for i,(r,_) in enumerate(cards):
        rank_to_idx[r].append(i)
    pairs=[r for r,c in rc.items() if c>=2]
    keeps=[]
    if len(pairs)>=2:
        p=pairs[:2]
        keeps.append(sorted(rank_to_idx[p[0]][:2]+rank_to_idx[p[1]][:2]))
    if len(pairs)>=1:
        keeps.append(sorted(rank_to_idx[pairs[0]][:2]))
    keeps.append([])
    uniq,seen=[],set()
    for k in keeps:
        t=tuple(k)
        if t not in seen:
            uniq.append(k); seen.add(t)
    return uniq

def candidate_keeps_for_full_house(cards: List[Card]) -> List[List[int]]:
    rc=Counter(r for r,_ in cards)
    rank_to_idx=defaultdict(list)
    for i,(r,_) in enumerate(cards):
        rank_to_idx[r].append(i)
    trips=[r for r,c in rc.items() if c>=3]
    pairs=[r for r,c in rc.items() if c>=2]
    keeps=[]
    for r in trips:
        keeps.append(sorted(rank_to_idx[r][:3]))
    if len(pairs)>=2:
        p=pairs[:2]
        keeps.append(sorted(rank_to_idx[p[0]][:2]+rank_to_idx[p[1]][:2]))
    if len(pairs)>=1:
        keeps.append(sorted(rank_to_idx[pairs[0]][:2]))
    # triple+another pair if exists
    if trips and pairs:
        r3=trips[0]
        r2 = pairs[0] if pairs[0]!=r3 else (pairs[1] if len(pairs)>1 else None)
        if r2 is not None:
            keeps.append(sorted(rank_to_idx[r3][:3]+rank_to_idx[r2][:2]))
    keeps.append([])
    uniq,seen=[],set()
    for k in keeps:
        t=tuple(k)
        if t not in seen:
            uniq.append(k); seen.add(t)
    return uniq

def candidate_keeps(cards: List[Card], target: str, deck_type: str) -> List[List[int]]:
    if target=='Straight':
        return candidate_keeps_for_straight(cards, deck_type)
    elif target=='Flush':
        return candidate_keeps_for_flush(cards)
    elif target=='Two Pair':
        return candidate_keeps_for_two_pair(cards)
    elif target=='Full House':
        return candidate_keeps_for_full_house(cards)
    else:
        return [[]]

# ---------- Exact success probabilities given remaining deck composition ----------
def remaining_counts_by_rank(deck_rem: List[Card], deck_type: str) -> Dict[int,int]:
    ranks_all=(list(range(1,14)) if deck_type=='52' else [1]+list(range(2,11)))
    cnt=dict.fromkeys(ranks_all,0)
    for r,_ in deck_rem:
        cnt[r]+=1
    return cnt

def remaining_counts_by_suit(deck_rem: List[Card]) -> Dict[int,int]:
    cnt={0:0,1:0,2:0,3:0}
    for _,s in deck_rem:
        cnt[s]+=1
    return cnt

def exact_prob_flush_with_remaining(deck_type: str, cards: List[Card], keep_idx: List[int], k: int, deck_rem: List[Card]) -> float:
    suits_cnt = remaining_counts_by_suit(deck_rem)
    Nrem = len(deck_rem)
    if Nrem < k or k<0: return 0.0
    kept_suits = Counter(cards[i][1] for i in keep_idx)
    # evaluate the best suit (either the one with most kept or the one maximizing prob)
    candidates = [max(kept_suits.items(), key=lambda x:x[1])[0]] if kept_suits else [0,1,2,3]
    best=0.0
    for s in candidates:
        s_kept = kept_suits.get(s,0)
        s_rem = suits_cnt.get(s,0)
        off_rem = Nrem - s_rem
        need = max(0, 5 - s_kept)
        fav=0
        for y in range(need, min(k, s_rem)+1):
            fav += comb(s_rem, y) * comb(off_rem, k - y)
        tot = comb(Nrem, k)
        p = 0.0 if tot==0 else fav/tot
        if p>best: best=p
    return best

def exact_prob_two_pair_with_remaining(deck_type: str, cards: List[Card], keep_idx: List[int], k: int, deck_rem: List[Card]) -> float:
    ranks_all=(list(range(1,14)) if deck_type=='52' else [1]+list(range(2,11)))
    avail = remaining_counts_by_rank(deck_rem, deck_type)
    kept = Counter(cards[i][0] for i in keep_idx)  # 0/1/2 allowed
    Nrem = len(deck_rem)
    if k<0 or k>Nrem: return 0.0
    # DP[t][p] p=0,1,2 (>=2 clipped)
    dp=[[0]*3 for _ in range(k+1)]
    dp[0][0]=1
    for r in ranks_all:
        a=avail[r]
        c0=min(kept.get(r,0),2)
        ndp=[[0]*3 for _ in range(k+1)]
        for t in range(k+1):
            for p in range(3):
                cur=dp[t][p]
                if cur==0: continue
                for x in range(0, min(a, k - t)+1):
                    C=c0+x
                    add_pair = 1 if C>=2 else 0
                    p2=min(2,p+add_pair)
                    ndp[t+x][p2]+=cur*comb(a,x)
        dp=ndp
    fav=sum(dp[k][p] for p in range(2,3))
    tot=comb(Nrem,k)
    return 0.0 if tot==0 else fav/tot

def exact_prob_full_house_with_remaining(deck_type: str, cards: List[Card], keep_idx: List[int], k: int, deck_rem: List[Card]) -> float:
    ranks_all=(list(range(1,14)) if deck_type=='52' else [1]+list(range(2,11)))
    avail = remaining_counts_by_rank(deck_rem, deck_type)
    kept = Counter(cards[i][0] for i in keep_idx)
    Nrem = len(deck_rem)
    if k<0 or k>Nrem: return 0.0
    # DP[t][T][Ponly]; T=#(>=3), Ponly=#(==2)
    dp=[[[0]*3 for _ in range(3)] for __ in range(k+1)]
    dp[0][0][0]=1
    for r in ranks_all:
        a=avail[r]
        c0=min(kept.get(r,0),3)
        ndp=[[[0]*3 for _ in range(3)] for __ in range(k+1)]
        for t in range(k+1):
            for T in range(3):
                for P in range(3):
                    cur=dp[t][T][P]
                    if cur==0: continue
                    for x in range(0, min(a, k - t)+1):
                        C=c0+x
                        addT=1 if C>=3 else 0
                        addP=1 if C==2 else 0
                        T2=min(2,T+addT); P2=min(2,P+addP)
                        ndp[t+x][T2][P2]+=cur*comb(a,x)
        dp=ndp
    fav=0
    for T in range(3):
        for P in range(3):
            if (T>=2) or (T>=1 and P>=1):
                fav+=dp[k][T][P]
    tot=comb(Nrem,k)
    return 0.0 if tot==0 else fav/tot

def exact_prob_straight_with_remaining(deck_type: str, cards: List[Card], keep_idx: List[int], k: int, deck_rem: List[Card], topL:int=3) -> float:
    ranks_all=(list(range(1,14)) if deck_type=='52' else [1]+list(range(2,11)))
    avail = remaining_counts_by_rank(deck_rem, deck_type)
    Nrem = len(deck_rem)
    if k<0 or k>Nrem: return 0.0
    kept_present=set(cards[i][0] for i in keep_idx)
    windows = windows_for_straight(deck_type)
    # rank 14 alias handled by mapping back to 1 in window definition already
    info=[]
    for W in windows:
        miss=[r for r in W if r not in kept_present]
        outs=sum(avail[r] for r in set(miss))
        info.append((len(miss), -outs, W, miss))
    info.sort()
    pick=info[:max(1, min(topL, len(info)))]
    A_out = Nrem - sum(avail[r] for _,_,_,miss in pick for r in set(miss))  # rough; not used per-window, we recompute
    # DP helper: count #ways to cover a set of required ranks with >=1 each, drawing s from those, then fill rest from outside.
    def count_cover_missing(missing: List[int], k:int) -> int:
        if not missing:
            return comb(Nrem, k)
        # DP over missing
        f=[[0]*(k+1) for _ in range(len(missing)+1)]
        f[0][0]=1
        sum_req=0
        for i,r in enumerate(missing, start=1):
            a=avail[r]
            for s in range(k+1):
                cur=f[i-1][s]
                if cur==0: continue
                for x in range(1, min(a, k - s)+1):
                    f[i][s+x]+=cur*comb(a,x)
            sum_req+=a
        Aout = Nrem - sum(avail[r] for r in set(missing))
        total=0
        for s in range(len(missing), min(k, sum(avail[r] for r in set(missing)))+1):
            total += f[len(missing)][s] * comb(Aout, k - s)
        return total
    # Inclusion–exclusion over the picked windows
    from itertools import combinations
    fav=0
    mL=len(pick)
    for m in range(1, mL+1):
        for idxs in combinations(range(mL), m):
            miss_union=sorted(set().union(*[pick[i][3] for i in idxs]))
            cnt=count_cover_missing(miss_union, k)
            fav += cnt if (m%2==1) else -cnt
    tot=comb(Nrem,k)
    return 0.0 if tot==0 else max(0.0, min(1.0, fav/tot))

def exact_success_prob_with_remaining(deck_type: str, target: str, cards: List[Card], keep_idx: List[int], k: int, deck_rem: List[Card]) -> float:
    if target=='Flush':
        return exact_prob_flush_with_remaining(deck_type, cards, keep_idx, k, deck_rem)
    elif target=='Two Pair':
        return exact_prob_two_pair_with_remaining(deck_type, cards, keep_idx, k, deck_rem)
    elif target=='Full House':
        return exact_prob_full_house_with_remaining(deck_type, cards, keep_idx, k, deck_rem)
    elif target=='Straight':
        return exact_prob_straight_with_remaining(deck_type, cards, keep_idx, k, deck_rem, topL=3)
    else:
        return 0.0

# ---------- Inner planner: choose (keep, k) to maximize immediate success ----------
def choose_best_plan(deck_type: str, target: str, hand: List[Card], deck_rem: List[Card], min_k:int, max_k:int) -> Tuple[List[int], int, float]:
    # If already made and min_k==0, keep; else plan for this round.
    if hand_has_target(hand, target, deck_type) and min_k==0:
        return (list(range(len(hand))), 0, 1.0)
    keeps = candidate_keeps(hand, target, deck_type)
    best_p=-1.0; best_k=None; best_keep=None
    H=len(hand)
    for keep in keeps:
        # legal ks for this round
        low=max(min_k, 1 if not hand_has_target(hand, target, deck_type) and min_k==0 else min_k)
        high=min(max_k, H - len(keep))
        if high < low: continue
        for k in range(low, high+1):
            p = exact_success_prob_with_remaining(deck_type, target, hand, keep, k, deck_rem)
            if p > best_p + 1e-18 or (abs(p-best_p)<=1e-18 and (best_k is None or k < best_k)):
                best_p = p; best_k = k; best_keep = keep
    if best_k is None:
        # fallback: discard min positive to avoid stalling
        best_k = max(1, min_k)
        best_keep = [] if best_k <= H else list(range(H))
        best_p = 0.0
    return best_keep, best_k, best_p

# ---------- Multi-round simulator ----------
def simulate_multiround(deck_type: str, target: str, hand_size:int, trials:int, seed:int, min_discard:int, max_discard:int):
    rng = random.Random(seed)
    exact_counts = Counter()  # step -> count of earliest success at exactly step
    deck_full = build_deck(deck_type)
    for _ in range(trials):
        deck = deck_full.copy()
        rng.shuffle(deck)
        hand = [deck.pop() for __ in range(hand_size)]
        earliest = None
        max_rounds = max_discard
        # Round 0 check
        if min_discard==0 and hand_has_target(hand, target, deck_type):
            earliest = 0
        else:
            for t in range(1, max_rounds+1):
                # plan this round
                keep_idx, k, _ = choose_best_plan(deck_type, target, hand, deck, min_k=min_discard, max_k=max_discard)
                # build new hand: keep kept cards, discard others, draw k from deck
                keep_set = set(keep_idx)
                kept = [hand[i] for i in sorted(keep_set)]
                # discard the rest back to nowhere, draw from deck without replacement
                draw_k = min(k, len(deck))
                drawn = [deck.pop() for __ in range(draw_k)]
                hand = kept + drawn
                # success?
                if hand_has_target(hand, target, deck_type):
                    earliest = t
                    break
                if len(deck)==0:
                    # no more cards to draw; stop
                    break
        if earliest is not None:
            exact_counts[earliest]+=1
        else:
            exact_counts[None]+=1  # failure within allowed rounds
    # Build report from min_discard..max_discard
    total = trials
    steps = list(range(min_discard, max_discard+1))
    exact = []
    cum = []
    acc = 0
    for s in steps:
        c = exact_counts[s]
        exact.append(c/total)
        acc += c
        cum.append(acc/total)
    return steps, exact, cum, exact_counts.get(None, 0)/total

def run_cli():
    parser = argparse.ArgumentParser(description="Multi-round discard simulator with exact inner planning.")
    parser.add_argument('--deck', choices=['52','40'], default='52')
    parser.add_argument('--hand-size', type=int, default=8)
    parser.add_argument('--trials', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--min-discard', type=int, default=0, help='Per-round minimum discard. Also defines range start for reporting.')
    parser.add_argument('--max-discard', type=int, default=5, help='Per-round maximum discard. Also defines the #rounds to simulate/report.')
    parser.add_argument('--target', choices=TARGETS, default='All')
    args = parser.parse_args()

    print("="*78)
    print(f"deck={args.deck}, hand size={args.hand_size}, trials={args.trials}, per-round discard in [{args.min_discard},{args.max_discard}]")
    print("="*78)

    if args.target == 'All':
        for tgt in TARGETS[1:]:
            print(f"\n=== Target: {tgt} ===")
            steps, exact, cum, fail = simulate_multiround(args.deck, tgt, args.hand_size, args.trials, args.seed, args.min_discard, args.max_discard)
            print("-"*78)
            print("Rounds  |  Exact Success%  |  Cumulative Success%")
            for s, e, c in zip(steps, exact, cum):
                print(f"{s:>6d} | {100*e:>13.2f}% | {100*c:>20.2f}%")
            print("-"*78)
            print(f"Failed within allowed rounds: {100*fail:.2f}%")
    else:
        print(f"Deck={args.deck}, H={args.hand_size}, trials={args.trials}, per-round discard in [{args.min_discard},{args.max_discard}], target={args.target}")
        steps, exact, cum, fail = simulate_multiround(args.deck, args.target, args.hand_size, args.trials, args.seed, args.min_discard, args.max_discard)
        print("-"*78)
        print("Rounds  |  Exact Success%  |  Cumulative Success%")
        for s, e, c in zip(steps, exact, cum):
            print(f"{s:>6d} | {100*e:>13.2f}% | {100*c:>20.2f}%")
        print("-"*78)
        print(f"Failed within allowed rounds: {100*fail:.2f}%")

if __name__ == '__main__':
    run_cli()