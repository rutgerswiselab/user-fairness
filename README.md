# user-fairness
This repository includes the implementation for User-oriented Fairness in Recommendation:
*Yunqi Li, Hanxiong Chen, Zuohui Fu, Yingqiang Ge, and Yongfeng Zhang. 2021. User-oriented Fairness in Recommendation. 
In Proceedings of the Web Conference 2021 (WWW'21).*

## Refernece

For inquiries contact Yunqi Li (yunqi.li@rutgers.edu) or Hanxiong Chen (hanxiong.chen@rutgers.edu) or Yongfeng Zhang (yongfeng.zhang@rutgers.edu)

```
@inproceedings{li2021user,
  title={User-oriented Fairness in Recommendation},
  author={Yunqi Li, Hanxiong Chen, Zuohui Fu, Yingqiang Ge, and Yongfeng Zhang},
  booktitle={Proceedings of the the Web Conference 2021},
  year={2021}
}
```

## Environments

Python 3.6.6

Packages:
```
pandas==0.24.2
Gurobi==9.0.2
```

## Datasets

- The processed datasets are in  [`./dataset/`](https://github.com/rutgerswiselab/user-fairness/tree/master/dataset)
- **Amazon Datasets**: The origin dataset can be found [here](http://jmcauley.ucsd.edu/data/amazon/). 
- For each dataset directory contains processed splitted testing datasets for re-ranking. 
    * 0.05_count_\*\_test_ratings.txt: grouping by total number of interactions.
    * sum_0.05_price_\*\_test_ratings.txt: grouping by total consumption.
    * max_0.05_price_\*\_test_ratings.txt: grouping by maximum price.

## Example to run the codes
-   
```
# Neural Collaborative Reasong on ML-100k dataset
> cd NCR/src/
> python main.py --rank 1 --model_name NCR --optimizer Adam --lr 0.001 --dataset ml100k01-1-5 --metric ndcg@5,ndcg@10,hit@5,hit@10 --max_his 5 --test_neg_n 100 --l2 1e-4 --r_weight 0.1 --random_seed 2022 --gpu 0
```
