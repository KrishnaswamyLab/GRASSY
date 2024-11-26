import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PATH = "data/processed/drugbank_12k_processed_with_props.csv"
df = pd.read_csv(PATH)
props = ["molwt","logp","qed","fsp3","tpsa"]

fig = plt.figure(figsize=(15, 10))
n_props = len(props)
for i, prop in enumerate(props):
    fig.add_subplot(2, 3, i+1)
    data = df[prop]
    sns.histplot(data, bins=30, kde=True, stat="probability")
    plt.xlabel(prop)

fig.savefig("drugbank_12k_eda.pdf", dpi=80)