from credit_risk import train_pipeline
from credit_risk import inference
from credit_risk.data_processing import data_management as dm
data1 = dm.load_dataset('test_data.csv')
data = data1.to_json()
cs = inference.predict_single(data,249123)
print(cs)

from aletheia import *

import tensorflow as tf
