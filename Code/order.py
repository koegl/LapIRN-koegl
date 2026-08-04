order_dsc = [
    "nosy-doe-38788231",
    "popular-sloth-38758804",
    "exultant-hawk-38756587",
    "intelligent-fish-38730451",
    "charming-trout-38863973",
    "secretive-dolphin-38622192",
    "polite-snake-38577202",
    "worried-elk-38863657",
    "rumbling-yak-38789486",
    "sincere-finch-38813192",
    "victorious-flea-38622412",
]
order_hd95 = [
    "nosy-doe-38788231",
    "exultant-hawk-38756587",
    "popular-sloth-38758804",
    "secretive-dolphin-38622192",
    "intelligent-fish-38730451",
    "polite-snake-38577202",
    "charming-trout-38863973",
    "sincere-finch-38813192",
    "worried-elk-38863657",
    "rumbling-yak-38789486",
    "victorious-flea-38622412",
]
order_ndv = [
    "rumbling-yak-38789486",
    "worried-elk-38863657",
    "intelligent-fish-38730451",
    "exultant-hawk-38756587",
    "popular-sloth-38758804",
    "charming-trout-38863973",
    "victorious-flea-38622412",
    "secretive-dolphin-38622192",
    "nosy-doe-38788231",
    "polite-snake-38577202",
    "sincere-finch-38813192",
]
order_mtv = [
    "rumbling-yak-38789486",
    "victorious-flea-38622412",
    "worried-elk-38863657",
    "intelligent-fish-38730451",
    "secretive-dolphin-38622192",
    "charming-trout-38863973",
    "nosy-doe-38788231",
    "popular-sloth-38758804",
    "exultant-hawk-38756587",
    "polite-snake-38577202",
    "sincere-finch-38813192",
]
order_tlg = [
    "worried-elk-38863657",
    "rumbling-yak-38789486",
    "charming-trout-38863973",
    "intelligent-fish-38730451",
    "secretive-dolphin-38622192",
    "victorious-flea-38622412",
    "popular-sloth-38758804",
    "exultant-hawk-38756587",
    "polite-snake-38577202",
    "nosy-doe-38788231",
    "sincere-finch-38813192",
]

# --- learn2reg-style ranking ---
# learn2reg turns each metric into a normalized rank score (best = 1, worst
# ~ 1/N) and combines the metrics with a geometric mean. the official scheme
# ranks per case before averaging; here only the final per-metric orderings are
# available, so this ranks on those directly.

orders = {
    "dsc": order_dsc,
    "hd95": order_hd95,
    "ndv": order_ndv,
    "mtv": order_mtv,
    "tlg": order_tlg,
}

# only models ranked in every metric can be compared
models = sorted(set.intersection(*(set(o) for o in orders.values())))


def rank_score(order, model):
    """normalized rank score in (0, 1]: best model gets 1, worst 1/N."""
    ranked = [m for m in order if m in models]
    return 1.0 - ranked.index(model) / len(ranked)


scores = {}
for model in models:
    per_metric = [rank_score(order, model) for order in orders.values()]
    geometric_mean = 1.0
    for s in per_metric:
        geometric_mean *= s
    geometric_mean **= 1.0 / len(per_metric)
    scores[model] = (geometric_mean, per_metric)

print(f"{'rank':<5}{'model':<30}{'score':<8}" + "".join(f"{m:>7}" for m in orders))
for i, (model, (score, per_metric)) in enumerate(
    sorted(scores.items(), key=lambda kv: -kv[1][0]), start=1
):
    print(f"{i:<5}{model:<30}{score:<8.3f}" + "".join(f"{s:>7.2f}" for s in per_metric))
