import json
import numpy as np
import pandas as pd
import argparse


def validate(sub, ans, k, categories):
    message = 'ok'
    status = 0
    eval_files = set(ans).intersection(set(sub))
    if len(eval_files) == 0:
        message = 'No sample for evaluation.'
        status = 1
        return sub, ans, message, status
    n = {c: 0 for c in categories}
    for eval_file in eval_files:
        gt = ans[eval_file]
        pr = sub[eval_file]
        if not isinstance(gt, dict):
            message = 'Invalid data type found in {} in the answer file. Should be dict.'.format(eval_file)
            status = 1
            return sub, ans, message, status
        if not isinstance(pr, dict):
            message = 'Invalid data type found in {} in the prediction file. Should be dict.'.format(eval_file)
            status = 1
            return sub, ans, message, status
        for c, v in gt.items():
            n[c] += len(v)
        for c, v in pr.items():
            if c not in categories:
                message = 'Invalid category found in {}(invalid category: {}).'.format(eval_file, c)
                status = 1
                return sub, ans, message, status
            if len(v) == 0:
                message = 'No prediction in {}({})'.format(eval_file, c)
                status = 1
            if len(v) > k:
                message = 'The number of predictions of {} exceeded in {}(maximum is {}).'.format(c, eval_file, k)
                status = 1
                return sub, ans, message, status
            for e in v:
                if not isinstance(e, list):
                    message = 'Invalid data type found in {} in the prediction file. Should be list.'.format(eval_file)
                    status = 1
                    return sub, ans, message, status
                if len(e) != 3:
                    message = 'Invalid data type found in {} in the prediction file. The length of the element should be 3.'.format(eval_file)
                    status = 1
                    return sub, ans, message, status

    for c, v in n.items():
        if v == 0:
            message = 'There must be at least one object({}) in the answer file.'.format(c)
            status = 1
            return sub, ans, message, status

    return sub, ans, message, status


def mAP(sub, ans, k, categories, threshold):
    scores = {}
    eval_files = set(ans).intersection(set(sub))
    tps = {category: 0 for category in categories}

    for eval_file in eval_files:
        for c, gt in ans[eval_file].items():
            tps[c] += len(gt)
        results = get_result(sub[eval_file], ans[eval_file], threshold)
        for c, result in results.items():
            if c not in scores:
                scores[c] = result
            else:
                scores[c] += result

    print('\nnumber of samples:', len(eval_files))
    print('number of objects:')
    for c, v in tps.items():
        print('  {}: {}'.format(c, v))
    print('scores:')
    score = 0
    for c, results in scores.items():
        score_per_category = 0
        pred_count = 0
        detected = 0
        max_hit_labels = min(k*len(eval_files), tps[c])
        df_results = pd.DataFrame(results).sort_values(1, ascending=False)
        for data in df_results.iterrows():
            pred_count += 1
            if data[1][0]:
                detected += 1
                score_per_category += detected/pred_count
        score_per_category /= max_hit_labels
        print('  AP for {}: '.format(c), score_per_category)
        score += score_per_category
    return score/len(tps)


def get_result(pr, gt, threshold):
    results = {}
    for c, pred_bbs in pr.items():
        results[c] = []
        if c not in gt:
            gt[c] = []
        df_pred = pd.DataFrame(pred_bbs)
        len_pred = df_pred.columns[-1]
        for i, pred_bb in df_pred.sort_values(len_pred, ascending=False).iterrows():
            if len(gt[c]) > 0:
                pred_true_dist = {j: compute_dist_xy(pred_bb, true_bb) for j, true_bb in enumerate(gt[c])}
                nearest = min(pred_true_dist.items(), key=lambda x: x[1])
                if nearest[1] <= threshold:
                    results[c].append([1, pred_bb[len_pred]])
                    del gt[c][nearest[0]]
                else:
                    results[c].append([0, pred_bb[len_pred]])
            else:
                results[c].append([0, pred_bb[len_pred]])

    return results


def compute_dist_xy(pred_bb, true_bb):
    return np.linalg.norm(np.array(pred_bb[:2]) - np.array(true_bb[:2]))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ground-truth-path', default = 'data/ans.json')
    parser.add_argument('--predictions-path', default = 'data/sub.json')

    args = parser.parse_args()

    return args


def main():
    # parse the arguments
    args = parse_args()
    ground_truth_path = args.ground_truth_path
    predictions_path = args.predictions_path

    # load the files
    with open(ground_truth_path) as f:
        ans = json.load(f)
    with open(predictions_path) as f:
        sub = json.load(f)

    # validation
    k = 50 # The maximum number of predictions per sample
    categories = ['pedestrian', 'vehicle']
    threshold = 1.0
    sub, ans, message, status = validate(sub, ans, k, categories)

    # evaluation
    if status == 0:
        score = mAP(sub, ans, k, categories, threshold)
        print('\nmAP: {}'.format(score))
    else:
        print(message)


if __name__ == '__main__':
    main()
