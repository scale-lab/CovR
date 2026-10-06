import re 

def get_top_module(design):
    top_module = None 
 
    module_pattern = re.compile(r'^\s*module\s+([a-zA-Z_][a-zA-Z0-9_]*)')
    with open(design, 'r') as file:
        for line in file:
            match = module_pattern.search(line)
            if match:
                top_module = match.group(1)
                break 

    if top_module is None: 
        print("[Error] Failed at extracting the top module name!!")
        
    print("Top Module: ", top_module)
    
    return top_module 
