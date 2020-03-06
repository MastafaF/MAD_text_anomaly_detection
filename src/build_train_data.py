import os
import sys

# export MAD='/Users/foufamastafa/Documents/master_thesis_KTH/MAD_anomaly_detection'
assert os.environ.get('MAD'), 'Please set the environment variable MAD'
MAD = os.environ['MAD']
DATA_PATH = MAD + "/data/"
sys.path.append(MAD + '/data/')


# Goal = build an NLI-like dataset with
# Normal_Class = Harry Potter in English
# Anomaly_Class = Any book different from Harry Potter
# Expected behaviour: Harry Potter in Russian should be identified as similar to Harry Potter in English

