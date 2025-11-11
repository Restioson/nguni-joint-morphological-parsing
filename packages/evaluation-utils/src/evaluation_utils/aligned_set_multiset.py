import copy


def multiset_sub(a, b):
    """Compute multiset `a - b` and return the result. This is immutable and modifies neither a nor b."""
    a = copy.deepcopy(a)
    for elt in b:
        if elt in a:
            a.remove(elt)

    return a


def multiset_intersection(a, b):
    """Compute multiset intersection `a ∩ b` and return the result. This is immutable and modifies neither a nor b."""
    b = copy.deepcopy(b)

    intersection = []

    for elt in a:
        if elt in b:
            b.remove(elt)
            intersection.append(elt)

    return intersection


def eval_model_aligned_multiset(target_tags, predicted_tags):
    results_per_tag = dict()
    default_entry = {"false_pos": 0, "true_pos": 0, "false_neg": 0}

    for prediction, target in zip(predicted_tags, target_tags):
        # According to Seker & Tsafarty, 2020:
        # - |true_pos| of token = |pred ∩ gold|
        # - |false_pos| of token = |pred - gold|
        # - |false_neg| of token = |gold - pred|

        true_pos = multiset_intersection(prediction, target)
        false_pos = multiset_sub(prediction, target)
        false_neg = multiset_sub(target, prediction)

        for tag in true_pos:
            results_per_tag.setdefault(tag, copy.deepcopy(default_entry))
            results_per_tag[tag]["true_pos"] += 1

        for tag in false_pos:
            results_per_tag.setdefault(tag, copy.deepcopy(default_entry))
            results_per_tag[tag]["false_pos"] += 1

        for tag in false_neg:
            results_per_tag.setdefault(tag, copy.deepcopy(default_entry))
            results_per_tag[tag]["false_neg"] += 1

    f1_per_tag = dict()
    total_true_pos = 0
    total_false_pos = 0
    total_false_neg = 0
    for tag, results in results_per_tag.items():
        true_pos = results["true_pos"]
        false_pos = results["false_pos"]
        false_neg = results["false_neg"]

        # Add to global stats for micro averaging
        total_true_pos += true_pos
        total_false_pos += false_pos
        total_false_neg += false_neg

        # Calculate per-tag F1
        precision = true_pos / (true_pos + false_pos) if true_pos + false_pos > 0 else 0
        recall = true_pos / (true_pos + false_neg) if true_pos + false_neg > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
        f1_per_tag[tag] = f1

    macro_f1 = sum(f1_per_tag.values()) / len(f1_per_tag)
    micro_precision = total_true_pos / (total_true_pos + total_false_pos)
    micro_recall = total_true_pos / (total_true_pos + total_false_neg)
    micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)

    return micro_f1, macro_f1


def eval_model_aligned_set(target_tags, predicted_tags):
    results_per_tag = dict()
    default_entry = {"tagger_produced": 0, "gold_standard_produced": 0, "tagger_correct": 0}

    for prediction, target in zip(predicted_tags, target_tags):
        for i in range(len(prediction)):
            pred_tag = prediction[i]
            target_tag = target[i] if i < len(target) else None

            results_per_tag.setdefault(pred_tag, copy.deepcopy(default_entry))
            results_per_tag.setdefault(target_tag, copy.deepcopy(default_entry))

            results_per_tag[pred_tag]["tagger_produced"] += 1

            if target_tag is not None:
                results_per_tag[target_tag]["gold_standard_produced"] += 1

            if pred_tag == target_tag:
                results_per_tag[pred_tag]["tagger_correct"] += 1

        for i in range(len(prediction), len(target)):
            target_tag = target[i]
            results_per_tag.setdefault(target_tag, copy.deepcopy(default_entry))
            results_per_tag[target_tag]["gold_standard_produced"] += 1

    f1_per_tag = dict()
    total_gold_standard_produced = 0
    total_tagger_produced = 0
    total_correct = 0
    for tag, results in results_per_tag.items():
        gold_standard_produced = results["gold_standard_produced"]
        tagger_produced = results["tagger_produced"]
        correct = results["tagger_correct"]

        # Add to global stats for micro averaging
        total_gold_standard_produced += gold_standard_produced
        total_tagger_produced += tagger_produced
        total_correct += correct

        # Calculate per-tag F1
        precision = correct / tagger_produced if tagger_produced > 0 else 0
        recall = correct / gold_standard_produced if gold_standard_produced > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
        f1_per_tag[tag] = f1

    macro_f1 = sum(f1_per_tag.values()) / len(f1_per_tag)
    micro_precision = total_correct / total_tagger_produced
    micro_recall = total_correct / total_gold_standard_produced
    micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)

    return micro_f1, macro_f1