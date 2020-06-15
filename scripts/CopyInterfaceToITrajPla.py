#!/usr/bin/env python
import shutil
import re
from pathlib import Path
import os
import jinja2
import codecs

regexp_virtual_deconstructor = r"^((\s)*virtual(\s)*(~(\w)+\(\s*\))(\s)*=(\s)*0(\s*);)"
regexp_construnctor = r"^(\s*(\w+\([\w\s,&*]*\)));"
def copyInterfaceAndModifyDestructor():
    p = Path.cwd()
    #shutil.copy()
    files = os.listdir(str(p))
    
    filesToCopy = [f for f in files if f[0] == 'I' and  Path.joinpath(p, f).suffix == '.h' and f != 'Interfaces.h' ]
    IincludePath = Path.joinpath(p, '../../../ITraPla/include/TraPla')
    print(IincludePath)
    if IincludePath.exists():
        shutil.rmtree(str(IincludePath))
    os.makedirs(str(IincludePath))
    
    for f in filesToCopy:
        newpath = Path.joinpath(IincludePath, f)
        with codecs.open(newpath, 'w', encoding='ascii', errors='ignore') as WriteToFile:
            with codecs.open(str(f), 'r', encoding='ascii', errors='ignore') as DataFile:
                for line in DataFile:
                    m = re.search(regexp_virtual_deconstructor, line)
                    if m:
                        line = '\tvirtual %s {};' %(m.group(4))
                    m1 = re.search(regexp_construnctor, line)
                    if m1:
                        line = '\t%s {};' % (m1.group(2))
                    WriteToFile.write(line.expandtabs(tabsize=4))

                    

    

                
    

    

if __name__ == "__main__":

    print("Copy Interface to ITraPla")
    copyInterfaceAndModifyDestructor()


