"""Why do unsupervised groups not follow the hand sorting?

    python tools/teach_lab/diagnose_grouping.py <ds_features.npz> <ds_video_features.npz>

1. Do the features group by source video (scene, actors, light) or by class?
2. Does removing each video's own mean (possible on new footage too) help?
3. Does a space learned from the sorted dataset (LDA / logreg scores) help
   unsupervised grouping of videos it never saw?

Everything except (1) holds out whole source videos (GroupKFold).
"""
import sys
import warnings

import numpy as np

import eval_grouping as E

warnings.filterwarnings("ignore")
MIX = {"clip8": 1, "r21d": 1, "intel": 1}


def centre_per_video(x, video):
    x = x.copy()
    for v in np.unique(video):
        m = video == v
        x[m] -= x[m].mean(0)
    return x


def kmeans_nmi(x, labels_list, k, seed=0):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import normalized_mutual_info_score as nmi
    x = PCA(n_components=min(32, x.shape[1]), random_state=seed).fit_transform(x)
    g = KMeans(k, n_init=10, random_state=seed).fit_predict(x)
    return [nmi(l, g) for l in labels_list], g


def heldout_grouping(x, y, video, splits, space):
    """Group each held-out fold without labels; purity / NMI vs the sorting."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import normalized_mutual_info_score as nmi
    pur, nm, vid = [], [], []
    for tr, te in splits:
        if space == "raw":
            z = PCA(n_components=32, random_state=0).fit(x[tr]).transform(x[te])
        elif space == "lda":
            p = PCA(n_components=256, random_state=0).fit(x[tr])
            lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit(p.transform(x[tr]), y[tr])
            z = lda.transform(p.transform(x[te]))
        elif space == "logreg":
            m = LogisticRegression(C=1, max_iter=3000, class_weight="balanced").fit(x[tr], y[tr])
            z = m.decision_function(x[te])
        k = len(set(y))
        g = KMeans(k, n_init=10, random_state=0).fit_predict(z)
        pur.append(E.purity(y[te], g))
        nm.append(nmi(y[te], g))
        vid.append(nmi(video[te], g))
    return np.mean(pur), np.mean(nm), np.mean(vid)


def main():
    blocks, y, video, _ = E.load(sys.argv[1], 20, sys.argv[2])
    y, video = y.astype(str), video.astype(str)
    k = len(set(y))
    splits = E.folds(y, video, 5)
    print(f"{len(y)} clips, {k} classes, {len(set(video))} source videos\n")

    print("1. KMeans on all clips: NMI with class vs NMI with source video")
    for name, w in {"clip4": {"clip": 1}, "pose": {"pose": 1}, "motion": {"motion": 1},
                    "clip8+r21d+intel": MIX}.items():
        x = E.mix(blocks, w)
        (nc, nv), _ = kmeans_nmi(x, [y, video], k)
        xc = centre_per_video(x, video)
        (ncc, nvc), _ = kmeans_nmi(xc, [y, video], k)
        print(f"   {name:18} class {nc:.2f}  video {nv:.2f}   | per-video centred: class {ncc:.2f}  video {nvc:.2f}")

    print("\n2/3. Held-out videos grouped without labels (purity / NMI class / NMI video)")
    x = E.mix(blocks, MIX)
    xc = centre_per_video(x, video)
    for space in ("raw", "lda", "logreg"):
        for tag, xx in (("as is", x), ("centred", xc)):
            p, n, v = heldout_grouping(xx, y, video, splits, space)
            print(f"   {space:7} {tag:8} purity {p:.2f}  NMI class {n:.2f}  NMI video {v:.2f}")

    print("\n   classifier on held-out videos, as is vs per-video centred")
    for tag, xx in (("as is", x), ("centred", xc)):
        b = {"all": xx}
        s = E.logreg(b, {"all": 1}, y, splits)
        print(f"   logreg {tag:8} acc {s['acc']:.3f}  bal {s['bal']:.3f}  top3 {s['top3']:.3f}")


if __name__ == "__main__":
    main()
