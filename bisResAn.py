import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

dir_path = "C:/Users/UAL-Laptop/Downloads/iterationResults/"
csv_names = os.listdir(dir_path)
existing = [int(csv_names[i].split("Rates.csv")[0][1:])
            for i in range(len(csv_names))]
existing.sort()

x = [i for i in range(0, len(existing))]
lab_text = []
rate_y = []
for i in existing:
    rate_df = pd.read_csv(dir_path + "l" + str(i) +
                          "Rates.csv").sort_values("bis_rate")
    rate = rate_df.iloc[-1][1]
    param = rate_df.iloc[-1][0]
    lab_text.append(param)
    rate_y.append(rate)

plt.figure(figsize=(16, 4))
plt.plot(x, rate_y)
plt.xlabel("Iterations")
plt.ylabel("Bistable Rate")
plt.title("Back Iteration Results")
for i in range(len(lab_text)):
    plt.text(x[i], rate_y[i] + 0.03, lab_text[i], fontsize=10)
plt.ylim(0, 1.1)
plt.tight_layout()
plt.savefig("./BackIterationPlot.jpg", transparent=False)
