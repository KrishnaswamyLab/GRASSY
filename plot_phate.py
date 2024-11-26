import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import phate
import scprep

PATH = "saved_tensors_12k/"
CHAFFER_PATH = "saved_tensors_12k_chaffer/"
GT_CSV_PATH = "data/processed/drugbank_12k_processed_with_props.csv"
gt_df = pd.read_csv(GT_CSV_PATH)
gt_df_props = gt_df['qed'].values

latents = []
props = []
chaffer_latents = []
chaffer_props = []

for file in os.listdir(PATH):
    if "latents" in file:
        latent_tensor = torch.load(os.path.join(PATH, file))
        print ("latent:", latent_tensor.shape, file)
        latents.append(latent_tensor)
    elif "props" in file:
        prop_tensor = torch.load(os.path.join(PATH, file))
        print ("prop:", prop_tensor.shape, file)
        props.append(prop_tensor)

for file in os.listdir(CHAFFER_PATH):
    if "latents" in file:
        latent_tensor = torch.load(os.path.join(CHAFFER_PATH, file))
        chaffer_latents.append(latent_tensor)
    elif "props" in file:
        prop_tensor = torch.load(os.path.join(CHAFFER_PATH, file))
        chaffer_props.append(prop_tensor)

print (len(latents), len(gt_df_props))

latents.extend(chaffer_latents)
props.extend(chaffer_props)
latents = torch.cat(latents, dim=0).numpy() # N x 128
props = torch.cat(props, dim=0).numpy() # N x 5

print (latents.shape, len(gt_df_props))

phate_operator = phate.PHATE()

phate_operator.set_params(potential_method="log")

# Fit and transform the latent representation
z_phate = phate_operator.fit_transform(latents)

fig = plt.figure(figsize=(15, 6))
fig.add_subplot(1,2,1)
plt.scatter(z_phate[:-3, 0], z_phate[:-3, 1], s=5, c=gt_df_props, cmap='viridis')
plt.scatter(z_phate[-3, 0], z_phate[-3, 1], s=25, color="#ff6b6b", label=f"14n", marker="s", alpha=1)
plt.scatter(z_phate[-2, 0], z_phate[-2, 1], s=25, color="#1dd1a1", label=f"C29", marker="s", alpha=1)
plt.scatter(z_phate[-1, 0], z_phate[-1, 1], s=25, color="#ff9f43", label=f"XCT790", marker="s", alpha=1)
plt.colorbar(label="QED", shrink=0.75)
plt.xlabel('PHATE 1')
plt.ylabel('PHATE 2')
plt.legend()
plt.title('DrugBank-12K Latent Representations colored by QED')

fig.add_subplot(1,2,2)
sns.histplot(gt_df_props, bins=30, kde=True, color="#5f27cd")
plt.xlabel("QED")

fig.savefig("plots/phate_drugbank_12k_qed_chaffer_gt.pdf", dpi=80)
# plt.show()