import os
import networkx as nx
import pandas as pd
data_dir = os.path.expanduser("cora")
edgelist = pd.read_csv(os.path.join(data_dir, "cora.cites"), sep='\t', header=None, names=["target", "source"])
edgelist["label"] = "cites"