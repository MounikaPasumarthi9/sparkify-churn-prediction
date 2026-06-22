# Databricks notebook source
# DBTITLE 1,Cell 1
#Building
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
import time
import os
import matplotlib.pyplot as plt
%matplotlib inline

import datetime
import pandas as pd
import pyspark
from pyspark.sql import SparkSession, SQLContext
import pyspark.sql.functions as F
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window

from sklearn.metrics import f1_score, recall_score, precision_score
from pyspark.sql.types import IntegerType, DoubleType, DateType, FloatType
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, GBTClassifier, LinearSVC

# COMMAND ----------

# DBTITLE 1,Cell 2
# Load data from Unity Catalog volume
df = spark.read \
.option("inferSchema", True) \
.option("header", True) \
.json("/Volumes/workspace/default/customer_data/Pyspark_Project/mini_sparkify_event_data.json")
    


# COMMAND ----------

df.display()

# COMMAND ----------

df.printSchema()

# COMMAND ----------

df.select('userId').show()

# COMMAND ----------

#Getting the Unique users

df.select("userId").dropDuplicates().count()

# COMMAND ----------

#Getting unique pages
df.select("page").distinct().collect()

# COMMAND ----------

#finding most visiting times

df.groupby("page").count().sort(F.desc('count')).show(truncate=False)

# COMMAND ----------

cancellation_event =F.udf(lambda x:1 if x=='Cancellation Confirmation' else 0, IntegerType())
df = df.withColumn("label",cancellation_event("page")); df.show()

# COMMAND ----------

#First creating a label & looking at each hour
each_hour = F.udf(lambda x: datetime.datetime.fromtimestamp(x/1000.0).hour,IntegerType())
df = df.withColumn("hour", each_hour(df.ts))
df.show(n=130)




# COMMAND ----------

#count of song per each hour

songs_per_hour = df.groupBy('hour').count().orderBy(F.desc("hour"))
songs_per_hour_pd = songs_per_hour.toPandas()
songs_per_hour_pd.hour = pd.to_numeric(songs_per_hour_pd.hour)

# COMMAND ----------

plt.scatter(songs_per_hour_pd['hour'], songs_per_hour_pd['count'])
plt.xlim(-1,24)
plt.ylim(0,1.2 *songs_per_hour_pd['count'].max())
plt.xlabel('hour')
plt.ylabel('songs per hour')

# COMMAND ----------

# DBTITLE 1,Cell 13
#Data Preprocessing 
# separating churned users and non churned users, Machine learning wants a single row for each users


churned_collect = df.where(df.label==1).select('userId').collect()
churned_users = set([int(float(row.userId))  for row in churned_collect])
churned_df = df.where(F.col('userId').isin (churned_users))


# COMMAND ----------

#Now collect all userids
all_collect = df.where(df.userId!='').select('userId').collect()
all_users = set([int(float(row.userId)) for row in all_collect])
all_df = df.where(F.col('userId').isin (all_users))

# COMMAND ----------

stayed_users = all_users - churned_users
print(stayed_users)

# COMMAND ----------

# DBTITLE 1,Cell 16
# Finding average number songs played by the churned user and non churned user
average_songs_played_bychurned = churned_df.filter(df.page=='NextSong').count()/len(churned_users)
print(average_songs_played_bychurned)




# COMMAND ----------

print(f"Stayed users count: {len(stayed_users)}")

# COMMAND ----------

average_songs_played_byStayed = all_df.filter(all_df.page =='NextSong').count()/len(stayed_users) 
print(average_songs_played_byStayed)

# COMMAND ----------

#dropping duplicates
churned_df = churned_df.filter(churned_df.userId!='')
churned_df.dropDuplicates(subset=['userId']).groupBy('gender').count().show()

# COMMAND ----------

#Creating a Feature Dataframe
feature_df = spark.createDataFrame(all_users, IntegerType()).withColumnRenamed('value','userId')
gender_identity = F.udf(lambda x:1 if x=='M' else 0, IntegerType())
df = df.withColumn("gender_binary", gender_identity('gender'))


# COMMAND ----------

#Joining the gender_binary column with feature dataframe

feature_df = feature_df.join(df.select('userId','gender_binary'),'userId') \
    .dropDuplicates(subset=['userId'])


# COMMAND ----------

#creating a level column and joining to feature dataframe 
subscription_level = F.udf(lambda x:1 if x=="free" else 0, IntegerType())
df = df.withColumn("level_binary",subscription_level("level"))



# COMMAND ----------

#now joining the level binary to feature dataframe
feature_df = feature_df.join(df.select("userId","ts","level_binary"),"userId") \
    .sort(F.desc("ts")) \
    .dropDuplicates(subset=["userId"]) \
        .drop("ts")

# COMMAND ----------

create_churn = F.udf(lambda x:1 if x in churned_users else 0, IntegerType())
feature_df = feature_df.withColumn("label",create_churn("userId"))


# COMMAND ----------

''' Now trying to group the different page activities in to different groups 
Neutral pages: “Cancel”, “Home”, “Logout”, “Save Settings”, “About”, “Settings” \

Negative pages: “Thumbs Down”, “Roll Advert”, “Help”, “Error” \

Positive pages: “Add to Playlist”, “Add Friend”, “NextSong”, “Thumbs Up” \

Downgrade pages: “Submit Downgrade”, “Downgrade” \

Upgrade pages: “Submit Upgrade”, “Upgrade” \ '''

pages = {}
pages['Neutral_pages'] = df.filter((df.page == 'Cancel') |(df.page =="Home")|(df.page =="Logout")|(df.page =="Save Settings")|(df.page =="About")|(df.page =="Settings"))
pages['Negative_pages'] = df.filter((df.page == 'Thumbs Down') |(df.page =="Roll Advert")|(df.page =="Help")|(df.page =="Error"))
pages['Positive_pages'] = df.filter((df.page == 'Add to Playlist')|(df.page =="Add Friend")|(df.page =="NextSong")|(df.page =="Thumbs Up"))
pages['Downgrade_pages'] = df.filter((df.page == 'Submit Downgrade')|(df.page =="Downgrade"))
pages['Upgrade_pages'] = df.filter((df.page == 'Submit Upgrade')|(df.page =="Upgrade"))

for key,value in pages.items():
    value_df = value.select('userId') \
    .groupBy('userId') \
    .agg({'userId':'count'}) \
    .withColumnRenamed('count(userId)', key)
    
    feature_df = feature_df.join(value_df,'userId','left').sort('userId') \
    .fillna({key:'0'})




# COMMAND ----------

#calculating the time spent by users on platform 

def delta_time(x,y):
    val1 = datetime.datetime.fromtimestamp(x/1000.0)
    val2 = datetime.datetime.fromtimestamp(y/1000.0)
    delta = val1-val2
    delta_days = delta.days
    if delta_days==0:
        return 1 
    return delta_days
delta = F.udf(delta_time,IntegerType())


# COMMAND ----------

#calculating the min,max timestamps

min_date_df = df.select('userId','ts') \
    .groupBy('userId') \
        .agg(F.min('ts'))

max_date_df = df.select('userId','ts') \
    .groupBy('userId') \
        .agg(F.max('ts'))

# COMMAND ----------

delta_df = min_date_df.join(max_date_df,'userId') \
    .withColumn('userActivedays',delta(F.col('min(ts)'),F.col('max(ts)'))) \
        .drop("min(ts)","max(ts)")



# COMMAND ----------

feature_df = feature_df.join(delta_df,'userId','left').sort('userId') \
    .fillna({'userActivedays':'0'})


# COMMAND ----------

feature_df.printSchema()
