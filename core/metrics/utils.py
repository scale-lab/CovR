# MIT License
#
# Copyright (c) 2026, CovR, SCALE Lab, Brown University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
