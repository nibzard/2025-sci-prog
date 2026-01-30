import json
import os

notebook_path = 'ads_feature_extraction/ads_feature_extraction.ipynb'
output_path = 'ads_feature_extraction/extracted_logic.txt'

if os.path.exists(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    found_cells = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'visual_impact' in source or 'dominant_colors' in source:
                found_cells.append(source)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n\n' + '='*50 + '\n\n'.join(found_cells))
    print(f"Extracted {len(found_cells)} cells to {output_path}")
else:
    print(f"Notebook not found at {notebook_path}")
