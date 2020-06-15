import os
import re
def getinitpath():
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "__init__.py"))


def load_element(var, pattern='([0-9a-z.-]+)'):
    '''Loads a file content'''
    filename = getinitpath()
    with open(filename, "rt") as init_file:
        travis_init = init_file.read()
        elem = re.search("%s = '%s'" % (var, pattern), travis_init).group(1)
    return elem


def load_version():
    return load_element("__version__")


def load_readme():
    path_to_readme = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "..", "README.md"))
    with open(path_to_readme, "rt") as fh:
        long_description = fh.read()
    return long_description
