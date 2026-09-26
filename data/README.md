# Dataset instructions

The Kaggle Bike Sharing Demand CSV files are intentionally not committed to this repository.

## Expected local files

~~~text
D:\Bike\train.csv
D:\Bike\test.csv
D:\Bike\sampleSubmission.csv
~~~

## Source

[Kaggle — Bike Sharing Demand](https://www.kaggle.com/competitions/bike-sharing-demand/data)

## Training

From the repository root:

~~~powershell
python train_model.py --train "D:\Bike\train.csv"
~~~

## Submission generation

~~~powershell
python generate_submission.py --test "D:\Bike\test.csv" --sample "D:\Bike\sampleSubmission.csv"
~~~

Raw dataset files remain local because they are not required to review the source code, tests, or committed evaluation report.
