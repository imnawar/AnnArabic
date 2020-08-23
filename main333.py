import re

# -*- coding: utf-8 -*-

def stetmentAnalyisi():
    s = u" ل ماعندي  "
    returnValue = True
    if re.match("^((?![/U+0644]).)*$", s):
        print("yessss in if ")
        returnValue = False
    return returnValue


stetmentAnalyisi()


