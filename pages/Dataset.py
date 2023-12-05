import streamlit as st

from collections import defaultdict
import boto3

import pandas as pd

from dotenv import find_dotenv, load_dotenv
import os

load_dotenv(find_dotenv())

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

client = boto3.client('s3', region_name='eu-west-3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_ACCESS_KEY, config=boto3.session.Config(signature_version='s3v4'))
s3 = boto3.resource('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_ACCESS_KEY,)
bucket = s3.Bucket("soundx-audio-dataset")

@st.cache_data
def load_data():
    files = defaultdict(list)
    for obj in bucket.objects.all():
        if obj.key.endswith('.wav'):
            label, file = obj.key.split('/')[0], obj.key.split('/')[1]
            files[label].append(file)
    return files


files = load_data()

summary = pd.DataFrame()
for label in files.keys():
    if label.startswith('AAAAA'):
        continue
    num_files = len(files[label])
    clean_files = len([file for file in files[label] if file.startswith('CLEAN')])
    df = pd.DataFrame({'label': label, 'num_files': num_files, 'clean_files': clean_files, 'raw_files': num_files - clean_files}, index=[0])
    # st.write(df)
    summary = pd.concat([summary, df], axis=0)

summary = summary.sort_values(by=['num_files'])
st.dataframe(summary)