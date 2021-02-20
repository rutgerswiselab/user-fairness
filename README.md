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
[Gurobi](https://www.gurobi.com/) is a commercial optimization solver. To run our code, please first install Gurobi and purchase a license. Without a license or fail to install properly, our code will not be able to run.

## Datasets

- The processed datasets are in  [`./dataset/`](https://github.com/rutgerswiselab/user-fairness/tree/master/dataset)
- **Amazon Datasets**: The origin dataset can be found [here](http://jmcauley.ucsd.edu/data/amazon/). 
- For each dataset directory contains processed splitted testing datasets for re-ranking. 
    * 0.05_count_\*\_test_ratings.txt: grouping by total number of interactions.
    * sum_0.05_price_\*\_test_ratings.txt: grouping by total consumption.
    * max_0.05_price_\*\_test_ratings.txt: grouping by maximum price.

## Run the codes
###  Prepare input data
- To run the code, please put the ranking file generated by recommendation model under the corresponding dataset folder. For example, to run model with 5Beauty-rand dataset, put the "\*\_rank.csv" file under "dataset/5Beauty-rand/" directory.
- Ranking csv file format: *uid \\t iid \\t score \\t label*
    - uid: user id column
    - iid: item id column
    - score: predicted score column
    - label: 0 or 1 to indicate this is a negative sample or positive sample

### Modify model.py
- Before running the code, please update the info of "src/model.py" in "\_\_main\_\_" section. You need to update **dataset\_name**, **model\_name**, **group\_name\_title**, **group\_1\_file**, **group\_2\_file**. 
    - **group\_name\_title**: select from "0.05" (interaction), "max\_0.05"(max price) and "sum\_0.05"(total consumption)
    - **group\_1\_file**: select from "\_count\_active\_test\_ratings.txt" (interaction), "\_price\_active\_test\_ratings.txt" (max price/total consumption)
    - **group\_2\_file**: select from "\_count\_inactive\_test\_ratings.txt" (interaction), "\_price\_inactive\_test\_ratings.txt" (max price/total consumption)
- You can chage the variable *epsilon* to control the strictness for fairness. 

### Run
```
> cd user-fairness/src/
> python model.py
```

### Result
The result file will be located in ./results/ by default.
