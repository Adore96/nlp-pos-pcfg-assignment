from nltk.probability import LidstoneProbDist


def lidstone_estimator(fd, bins):
    """Smoothing estimator for HMM training. Must live in an importable
    module (not a notebook cell) — pickle needs to find it by this exact
    module path when the saved tagger is loaded later."""
    return LidstoneProbDist(fd, 0.1, bins)