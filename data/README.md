# Data directory

Do not commit Kaggle CSV files to this repository.

For local Windows work, the project accepts:

```text
D:\Bike\train.csv
D:\Bike\test.csv
D:\Bike\sampleSubmission.csv
```

For Docker training, mount the directory:

```powershell
docker run --rm -v "D:\Bike:/app/data" IMAGE_NAME ...
```

Inside Docker, the equivalent paths are:

```text
/app/data/train.csv
/app/data/test.csv
/app/data/sampleSubmission.csv
```
