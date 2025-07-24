from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
import pandas as pd
import torch

spark = SparkSession.builder.appName("GPUExample").getOrCreate()

@pandas_udf("double")
def gpu_heavy_udf(s: pd.Series) -> pd.Series:
    # Example: run a dummy GPU task on input series
    tensor = torch.tensor(s.values).cuda()
    result = tensor * 2  # dummy operation
    return pd.Series(result.cpu().numpy())

df = spark.createDataFrame([(1.0,), (2.0,), (3.0,)], ["value"])
df = df.withColumn("gpu_result", gpu_heavy_udf(df["value"]))
df.show()
