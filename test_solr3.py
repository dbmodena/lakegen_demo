import sys
from src.client_solr import LocalSolrClient

client = LocalSolrClient(core="bologna")
res = client.select(["temperatura"], q_op="OR", rows=1)
doc = res.get("response", {}).get("docs", [])[0]
print(doc.keys())
print(doc.get("columns"))
