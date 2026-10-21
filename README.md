**#Assignment 2 - Question 1**

Here we implement decision-tree decision tree baseline and bootstrap aggregation on Olivetti faces.The seed used is '242009' obtained from 'M24MA2009'

**# Run**

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python q1.py --out results
```


We download Olivetti Faces through scikit-learn , then we train the Trees on clean images only , and evaluate the final frozen model , chosing the best based on the validation score.


**Question 2**

Here we reuse that exact split in question 1.

```sh
python q2.py --results results
```

Outputs are stored in 'results/q2/'.

**Question 3**

We reuse the saved split and selected settings from Question 1 and 2.

```sh
python q3.py --results results
```


The full assignment run uses 30 resamples and writes its tables and figure to
`results/q3/`.

Tree selection and Bagging selection both uses Validation score only.

Thankyou