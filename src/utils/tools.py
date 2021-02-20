import logging
from utils.rank_metrics import *


def create_logger(name='result_logger', path='results.log'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(path)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def evaluation_methods(df, metrics):
    """
    Generate evaluation scores
    :param df:
    :param metrics:
    :return:
    """
    evaluations = []
    data_df = df.copy(deep=True)
    data_df["q*s"] = data_df['q'] * data_df['score']
    for metric in metrics:
        k = int(metric.split('@')[-1])
        tmp_df = data_df.sort_values(by='q*s', ascending=False, ignore_index=True)
        df_group = tmp_df.groupby('uid')
        if metric.startswith('ndcg@'):
            ndcgs = []
            for uid, group in df_group:
                ndcgs.append(ndcg_at_k(group['label'].tolist()[:k], k=k, method=1))
            evaluations.append(np.average(ndcgs))
        elif metric.startswith('hit@'):
            hits = []
            for uid, group in df_group:
                hits.append(int(np.sum(group['label'][:k]) > 0))
            evaluations.append(np.average(hits))
        elif metric.startswith('precision@'):
            precisions = []
            for uid, group in df_group:
                if len(group['label'].tolist()) < k:
                    print(group)
                    print(uid)
                precisions.append(precision_at_k(group['label'].tolist()[:k], k=k))
            evaluations.append(np.average(precisions))
        elif metric.startswith('recall@'):
            recalls = []
            for uid, group in df_group:
                if np.sum(group['label']) == 0:
                    continue
                recalls.append(1.0 * np.sum(group['label'][:k]) / np.sum(group['label']))
            evaluations.append(np.average(recalls))
        elif metric.startswith('f1@'):
            f1 = []
            for uid, group in df_group:
                if np.sum(group['label']) == 0:
                    continue
                f1.append(2 * np.sum(group['label'][:k]) / (np.sum(group['label']) + k))
            evaluations.append(np.average(f1))
    return evaluations

