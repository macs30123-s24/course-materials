## Lab 7 - PySpark

### 1. Launch an EMR Cluster
```bash
aws emr create-cluster \
    --name "Spark Cluster" \
    --release-label "emr-6.2.0" \
    --applications Name=Hadoop Name=Hive Name=JupyterEnterpriseGateway Name=JupyterHub Name=Livy Name=Pig Name=Spark Name=Tez \
    --instance-type m5.xlarge \
    --instance-count 3 \
    --use-default-roles \
    --region us-east-1 \
    --ec2-attributes '{"KeyName": "vockey"}' \
    --configurations '[{"Classification": "jupyter-s3-conf", "Properties": {"s3.persistence.enabled": "true", "s3.persistence.bucket": "<YOUR_BUCKET_NAME>"}}]'
```

- While waiting for the EMR Cluster to launch, please complete the exercises below, save as python files, run them on Midway 3.

The sbatch file for submitting the job and running on Midway:
```bash
#!/bin/bash

#SBATCH --job-name=ssd-spark-example
#SBATCH --output=ssd-spark.out
#SBATCH --error=ssd-spark.err
#SBATCH --ntasks=10
#SBATCH --partition=caslake
#SBATCH --account=macs30123

module load python spark

export PYSPARK_DRIVER_PYTHON=/software/python-anaconda-2022.05-el8-x86_64/bin/python3

spark-submit --master local[*] YOUR_FILE.py
```


### 2. PySpark Exercises

- Ex1. Basic Functions

    - Complete the script in [this python file](./lab_wk7_spark.py) that processes and analyzes a dataset containing political speeches to identify and count the occurrences of certain policy-related keywords.
        - Filter the speeches, extract keywords and map, and reduce by key
        - Expected output: [('Politician Name 1', count), ('Politician Name 2', count)]

- Ex2. Spark SQL
    - Complete code in [this notebook](./lab_wk7_sparksql.ipynb)


