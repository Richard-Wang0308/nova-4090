# hops for your example query against role-B synthons
.venv/bin/python scaffold_hop.py query \
  --smiles "Nc1cn[nH]c(=O)c1" --role 2 --top 20 --out hops.csv

# hops for a DB molecule (e.g. your smi2 = mol_id 138879, role C)
.venv/bin/python scaffold_hop.py query --mol-id 138879 --role 2 --top 20

# pairwise within a capped subset
.venv/bin/python scaffold_hop.py pairwise --role 2 --limit 3000 --top 50



# 1) same Murcko scaffold
.venv/bin/python scaffold_same_murcko.py \
  --smiles "Nc1cn[nH]c(=O)c1" --role 4 --out same_murcko_hits.csv

# 2) generic Murcko (topology only)
.venv/bin/python scaffold_generic_murcko.py \
  --mol-id 138879 --role 4 --out generic_murcko_hits.csv

# 3) scaffold substructure (contains query core)
.venv/bin/python scaffold_substructure.py \
  --smiles "Nc1cn[nH]c(=O)c1" --role 4 --out scaffold_substruct_hits.csv