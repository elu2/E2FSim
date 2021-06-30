#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import os


# In[9]:


colnames = ["L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10", "bistable_count", "rebi_count"]
init_df = pd.DataFrame(columns=colnames)

if os.path.exists("model_rebi_counts.csv"):
    option = input("File exists. Overwrite? [y/n]").lower()
    if option == "y":
        init_df.to_csv("model_rebi_counts.csv", index=False)
        print("New empty file created.")
    else:
        print("Previous file retained.")

else:
    init_df.to_csv("model_rebi_counts.csv", index=False)


# In[ ]:




