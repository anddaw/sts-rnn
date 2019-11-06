from scipy import stats


def pearson_r(predictions, ground_truths):
    r, _ = stats.pearsonr(predictions, ground_truths)
    return r * 100
