import os
import shutil
import json
import errno
import subprocess
import codecs


class PackageNotFoundError(Exception):
    pass

def getPackageId(package_name, version, user, channel):
    return "{name}/{version}@{user}/{channel}".format(name=package_name, version=version, user=user, channel=channel)


def call_process(cmd, env = None):
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        env=env
    )
    try:
        outs, errs = proc.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        raise TimeoutError(errno.ENOENT, os.strerror(errno.ENOENT), "vcvarsall doesn't respond")
    return outs


def conan_info(package_name="apsw_lib_trapla_series", version="0.5.0", user="apsw", channel="testing"):
    package_id = getPackageId(package_name, version, user, channel)
    cmd = "conan info {package_id} --paths -j".format(package_id=package_id)
    out = call_process(cmd)
    try:
        result = json.loads(out)
    except json.JSONDecodeError:
        raise PackageNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), "Package {package_id} not found.".format(package_id=package_id))
    return result, package_id

def getsummary():
    cpp_types = {}
    result, package_id = conan_info(package_name="rpp_architecture", version="0.5.0", channel="stable")
    package_folder = [ref for ref in result if ref["reference"] == package_id][0]["package_folder"]
    conan_link = os.path.join(package_folder, '.conan_link')
    if os.path.exists(conan_link):
        with open(conan_link, 'r') as File:
            package_folder = File.read()
    
    with open(os.path.join(package_folder, 'cpp', 'cpp', 'summery.json'), 'r') as File:
        cpp_types.update(json.load(File)["structs"])


    result, package_id = conan_info(package_name="apsw_if_roadgraph", version="0.3.0", channel="stable")
    package_folder = [ref for ref in result if ref["reference"] == package_id][0]["package_folder"]
    conan_link = os.path.join(package_folder, '.conan_link')
    if os.path.exists(conan_link):
        with open(conan_link, 'r') as File:
            package_folder = File.read()
    
    with open(os.path.join(package_folder, 'include', 'cpp', 'summery.json'), 'r') as File:
        cpp_types.update(json.load(File)["structs"])

    return cpp_types

def recursive_find_files(path, suffix, files):
	onlydirs = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
	#print(onlydirs)

	for subdir_short in onlydirs:
		subdir = os.path.join(path, subdir_short)
		recursive_find_files(subdir, suffix, files)

	onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
	
	for file in onlyfiles:
		filename, file_extension = os.path.splitext(file)
		if file_extension in suffix:
			file_path = os.path.join(path, file)
			files.append(file_path)

def check_and_correct(files, structs):
    changed_files = []
    for f in files:
        print(os.path.abspath(f))
        textcpph = ""
        #with codecs.open(os.path.abspath(f), 'r', encoding='ascii', errors='ignore') as DataFile:
        with open(os.path.abspath(f), 'r') as DataFile:
            textcpph = DataFile.read()
            max_indx = -1
            for k in structs.keys():
                
                indx = textcpph.find(k)
                
                textcpph = textcpph.replace(k, structs[k])
                if indx > max_indx:
                    max_indx = indx
        if max_indx < 0:
            continue
        
        shutil.move(os.path.abspath(f), os.path.abspath(f) + '.old')
        with open(os.path.abspath(f), 'w') as DataFile:
            DataFile.write(textcpph)
        changed_files.append(f)

    print(changed_files)

        





        

    

if __name__ == "__main__":
    structs = getsummary()
    files = []
    recursive_find_files("../libraries", [".cpp", ".h"], files)
    check_and_correct(files, structs)