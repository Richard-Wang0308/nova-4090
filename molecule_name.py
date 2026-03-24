import sqlite3
import os
from rdkit import Chem
from rdkit.Chem import AllChem
import json
from typing import List, Dict, Optional
from tqdm import tqdm  # For progress bar

from combinatorial_db.reactions import react_molecules

def get_reaction_name_from_smiles(target_smiles: str, db_path: str, 
                                   search_2_component: bool = True, 
                                   search_3_component: bool = True,
                                   max_results: int = 1) -> Optional[str]:
    """
    Find the first reaction name that produces the given SMILES.
    
    Args:
        target_smiles: The product SMILES to search for
        db_path: Path to molecules.sqlite database
        search_2_component: Whether to search 2-component reactions
        search_3_component: Whether to search 3-component reactions
        max_results: Maximum number of results (1 = return first match)
    
    Returns:
        Reaction name like 'rxn:2:84799:75255' or None if not found
    """
    try:
        # Canonicalize target SMILES for comparison
        target_mol = Chem.MolFromSmiles(target_smiles)
        if not target_mol:
            print(f"Invalid SMILES: {target_smiles}")
            return None
        
        canonical_target = Chem.MolToSmiles(target_mol)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all reactions
        cursor.execute("SELECT rxn_id, roleA, roleB, roleC FROM reactions")
        reactions = cursor.fetchall()
        
        for rxn_id, roleA, roleB, roleC in reactions:
            # Search 2-component reactions
            if search_2_component:
                # Get molecules matching roleA
                cursor.execute("""
                    SELECT mol_id FROM molecules 
                    WHERE (role_mask & ?) = ?
                """, (roleA, roleA))
                roleA_ids = [row[0] for row in cursor.fetchall()]
                
                # Get molecules matching roleB
                cursor.execute("""
                    SELECT mol_id FROM molecules 
                    WHERE (role_mask & ?) = ?
                """, (roleB, roleB))
                roleB_ids = [row[0] for row in cursor.fetchall()]
                
                # Try all combinations
                for mol1_id in roleA_ids:
                    for mol2_id in roleB_ids:
                        product = react_molecules(rxn_id, mol1_id, mol2_id, db_path)
                        if product:
                            product_mol = Chem.MolFromSmiles(product)
                            if product_mol and Chem.MolToSmiles(product_mol) == canonical_target:
                                conn.close()
                                return f"rxn:{rxn_id}:{mol1_id}:{mol2_id}"
            
            # Search 3-component reactions
            if search_3_component and roleC is not None:
                cursor.execute("""
                    SELECT mol_id FROM molecules 
                    WHERE (role_mask & ?) = ?
                """, (roleA, roleA))
                roleA_ids = [row[0] for row in cursor.fetchall()]
                
                cursor.execute("""
                    SELECT mol_id FROM molecules 
                    WHERE (role_mask & ?) = ?
                """, (roleB, roleB))
                roleB_ids = [row[0] for row in cursor.fetchall()]
                
                cursor.execute("""
                    SELECT mol_id FROM molecules 
                    WHERE (role_mask & ?) = ?
                """, (roleC, roleC))
                roleC_ids = [row[0] for row in cursor.fetchall()]
                
                # Try all combinations
                for mol1_id in roleA_ids:
                    for mol2_id in roleB_ids:
                        for mol3_id in roleC_ids:
                            product = react_three_components(rxn_id, mol1_id, mol2_id, mol3_id, db_path)
                            if product:
                                product_mol = Chem.MolFromSmiles(product)
                                if product_mol and Chem.MolToSmiles(product_mol) == canonical_target:
                                    conn.close()
                                    return f"rxn:{rxn_id}:{mol1_id}:{mol2_id}:{mol3_id}"
        
        conn.close()
        return None
        
    except Exception as e:
        print(f"Error finding reaction from SMILES {target_smiles}: {e}")
        return None


def batch_convert_smiles_to_reactions(smiles_list: List[str], db_path: str, 
                                       output_file: str = "reaction_names.txt",
                                       use_cache: bool = True,
                                       cache_file: str = "product_cache.json") -> Dict[str, Optional[str]]:
    """
    Convert a list of SMILES to their reaction names.
    
    Args:
        smiles_list: List of SMILES strings
        db_path: Path to molecules.sqlite database
        output_file: Output file to save results
        use_cache: Whether to build/use cache for faster lookup
        cache_file: Cache file path
    
    Returns:
        Dictionary mapping SMILES -> reaction_name
    """
    results = {}
    
    if use_cache:
        print("Building product cache (this may take a while)...")
        cache = get_or_build_product_cache(db_path, cache_file)
        print(f"Cache built with {len(cache)} products")
        
        print("\nConverting SMILES to reaction names...")
        for smiles in tqdm(smiles_list):
            reaction_names = get_reaction_name_from_smiles_cached(smiles, cache)
            results[smiles] = reaction_names[0] if reaction_names else None
    else:
        print("Converting SMILES to reaction names (without cache)...")
        for smiles in tqdm(smiles_list):
            reaction_name = get_reaction_name_from_smiles(smiles, db_path)
            results[smiles] = reaction_name
    
    # Save results
    with open(output_file, 'w') as f:
        for smiles, reaction_name in results.items():
            if reaction_name:
                f.write(f"{reaction_name}\t{smiles}\n")
            else:
                f.write(f"NOT_FOUND\t{smiles}\n")
    
    print(f"\nResults saved to {output_file}")
    
    # Print summary
    found = sum(1 for v in results.values() if v is not None)
    print(f"\nSummary:")
    print(f"  Total: {len(smiles_list)}")
    print(f"  Found: {found}")
    print(f"  Not found: {len(smiles_list) - found}")
    
    return results


def get_or_build_product_cache(db_path: str, cache_file: str = "product_cache.json") -> dict:
    """
    Build or load a cache mapping SMILES -> reaction names.
    This only needs to be done once, then can be reused.
    """
    import os
    
    # Check if cache exists
    if os.path.exists(cache_file):
        print(f"Loading existing cache from {cache_file}...")
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    # Build cache
    print("Building new cache...")
    cache = {}
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT rxn_id, roleA, roleB, roleC FROM reactions")
    reactions = cursor.fetchall()
    
    for rxn_id, roleA, roleB, roleC in tqdm(reactions, desc="Processing reactions"):
        # 2-component reactions
        cursor.execute("SELECT mol_id FROM molecules WHERE (role_mask & ?) = ?", (roleA, roleA))
        roleA_ids = [row[0] for row in cursor.fetchall()]
        
        cursor.execute("SELECT mol_id FROM molecules WHERE (role_mask & ?) = ?", (roleB, roleB))
        roleB_ids = [row[0] for row in cursor.fetchall()]
        
        for mol1_id in roleA_ids:
            for mol2_id in roleB_ids:
                product = react_molecules(rxn_id, mol1_id, mol2_id, db_path)
                if product:
                    mol = Chem.MolFromSmiles(product)
                    if mol:
                        canonical = Chem.MolToSmiles(mol)
                        reaction_name = f"rxn:{rxn_id}:{mol1_id}:{mol2_id}"
                        
                        if canonical not in cache:
                            cache[canonical] = []
                        cache[canonical].append(reaction_name)
        
        # 3-component reactions
        if roleC is not None:
            cursor.execute("SELECT mol_id FROM molecules WHERE (role_mask & ?) = ?", (roleC, roleC))
            roleC_ids = [row[0] for row in cursor.fetchall()]
            
            for mol1_id in roleA_ids:
                for mol2_id in roleB_ids:
                    for mol3_id in roleC_ids:
                        product = react_three_components(rxn_id, mol1_id, mol2_id, mol3_id, db_path)
                        if product:
                            mol = Chem.MolFromSmiles(product)
                            if mol:
                                canonical = Chem.MolToSmiles(mol)
                                reaction_name = f"rxn:{rxn_id}:{mol1_id}:{mol2_id}:{mol3_id}"
                                
                                if canonical not in cache:
                                    cache[canonical] = []
                                cache[canonical].append(reaction_name)
    
    conn.close()
    
    # Save cache
    print(f"Saving cache to {cache_file}...")
    with open(cache_file, 'w') as f:
        json.dump(cache, f)
    
    return cache


def get_reaction_name_from_smiles_cached(target_smiles: str, cache: dict) -> List[str]:
    """Fast lookup using pre-built cache."""
    mol = Chem.MolFromSmiles(target_smiles)
    if not mol:
        return []
    
    canonical = Chem.MolToSmiles(mol)
    return cache.get(canonical, [])


# Main execution
if __name__ == "__main__":
    # Your SMILES list
    smiles_list = [
        "COc1ccc2cc(NCC=N)ccc2c1", 
        "N=CCNCC#Cc1ccc2ccsc2c1", 
        "CNC1CC(C#Cc2cccs2)C1", 
        "CNC1CCC(c2ccc(-c3ccc(C)s3)cc2)CC1", 
        "CNC(C)c1cc2cc(-c3cccc(C)c3)ccc2o1", 
        "CNC(C)c1cc2ccccc2s1", 
        "[Na]c1cn(CCCCCI)nn1", 
        "CNC1CC1c1ccc(-c2cn(I)nn2)cc1", 
        "CNC(C)c1ccc2c(c1)-c1ccccc1C2", 
        "CNC1CCC(c2c[nH]c3ccccc23)CC1", 
        "CNC[C@@H](c1ccc2ccccc2c1)C1CC1", 
        "N=CCNCC#Cc1csc2ccccc12", 
        "CNC1CC[C@@H](c2ccc(C)cc2)C1", 
        "CNC1CC2(Cc3ccccc3C2)C1", 
        "CNC1CC(c2ccc3ccccc3c2)C1", 
        "CCNC(Cc1cc(C)ccc1C)CC1CC1", 
        "CNC1CCC(c2ccc(C)cc2)CC1", 
        "CNC1CCc2cc3ccccc3cc2C1", 
        "CNC1CCN(c2ccc(C)cc2C)C1", 
        "CNC1CC[C@H](c2ccc(C)cc2)C1", 
        "Cc1ccc2sc(CNCC=N)cc2c1", 
        "C[C@H](N)c1ccc(-c2cn(-c3ccc(F)c(Br)c3)nn2)cc1Cl", 
        "NCC=Cc1ccc(-c2cn(I)nn2)cc1", 
        "[Na]c1cn(CCCCCBr)nn1", 
        "ClCC=Cc1ccc(-c2cn(C3(CI)CCNC3)nn2)cc1",
        "Cc1csc(-c2ccc(CCC(C)NCCc3cnc4nc[nH]c4c3)cc2)c1", 
        "C[C@H](N)c1cn(-c2ccc(I)cc2Br)nn1", 
        "C[C@@H](N)c1cn(-c2ccc(I)cc2Br)nn1", 
        "CNC(C)c1ccc2sc3ccccc3c2c1", 
        "CC(N)c1cn(-c2ccc(I)cc2Br)nn1", 
        "CNC(C)c1cc2cc(-c3ccc(C)o3)ccc2o1", 
        "NCc1cc(F)ccc1CCCn1cc(-c2cc(Br)cs2)nn1", 
        "CNC(C)c1ccc2ccsc2c1", 
        "Cc1cc(Br)c(-c2cn(CCC3CC3c3cnc(Cl)nc3N)nn2)s1", 
        "CNC(C)c1cc2ccccc2o1", 
        "CNC(C)c1cc2cc(C)ccc2s1"
    ]
    
    # Set database path
    db_path = os.path.join(os.path.dirname(__file__), "combinatorial_db","molecules.sqlite")
    
    # Convert with cache (recommended for large lists)
    results = batch_convert_smiles_to_reactions(
        smiles_list, 
        db_path, 
        output_file="reaction_names.txt",
        use_cache=True,  # Set to False if you don't want to build cache
        cache_file="product_cache.json"
    )
    
    # Print first few results
    print("\nFirst 5 results:")
    for i, (smiles, rxn_name) in enumerate(list(results.items())[:5]):
        print(f"{i+1}. {rxn_name} -> {smiles}")
