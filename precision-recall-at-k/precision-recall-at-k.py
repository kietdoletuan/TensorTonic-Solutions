def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """
    topk_recommend = recommended[:k]
    rec_items = set(topk_recommend)
    relevant_count = sum(item in rec_items for item in relevant)
    precision = relevant_count/k
    recall = relevant_count / len(relevant)

    return [precision, recall]
    pass