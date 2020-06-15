# -*- coding: utf-8 -*-
#!/usr/bin/env python
import os
import re

from pathlib import Path, PurePath
import codecs

import shutil

pattern 	     = r"^((\s)*(#\s?include)\s+([<\"]((\w+[\\\/]?\w+)+(\.h)?(\.hpp)?)[\">]))"
#pattern 	     = r"(#\s?include)\s+([<\"]((\w+[\\\/]?\w+)+(\.h)?(\.hpp)?)[\">])"
pattern_rel_path = r"(\w+[._]?\w+)\/(include)\/(.+)\/(\w+\.h)" 


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

def writeHeadersToFile(path, headers):
	indx = 0
	outputpath = path + '.tmp'
	with codecs.open(path, 'r', encoding='ascii', errors='ignore') as InputFile:
		print("Output File: %s" %(outputpath))
		with open(outputpath, 'w') as OutputFile:
			for line in InputFile:
				onlyline = line.replace('\r', '').replace('\n', '')
				m = re.search(pattern, onlyline)
				if m:
					outputstring = str(headers[indx])
					outputstring = outputstring.replace('\\', '/') + '\n'
					OutputFile.write(outputstring)
					indx = indx + 1
				else:
					outputstring = onlyline + '\n'
					OutputFile.write(outputstring)

	oldpath = path + '.old'
	shutil.copyfile(path, oldpath)
	os.remove(path)
	shutil.copyfile(outputpath, path)
	os.remove(outputpath)



def getHeaderFromFile(path):
	header_in_file = []

	with codecs.open(path, 'r', encoding='ascii', errors='ignore') as DataFile:
		for line in DataFile:
			onlyline = line.replace('\r', '').replace('\n', '')
			m = re.search(pattern, onlyline)
			if m:
				header_in_file.append(m.group(5))
	return header_in_file

if __name__ == '__main__':
	folderName   = os.getcwd()


	path_IF_Pre2Rpp  = Path('C:/work/libraries/IF_Pre2Rpp')
	path_IF_PiloPa  = Path('C:/work/libraries/IF_PiloPa')
	files_IF_PiloPa_header = []
	files_IF_Pre2Rpp_header = []
	files_IF_Series_header = []
	files_IF_Series_cpp = []

	recursive_find_files(str(path_IF_PiloPa), ['.h', '.hpp'], files_IF_PiloPa_header)
	recursive_find_files(str(path_IF_Pre2Rpp), ['.h', '.hpp'], files_IF_Pre2Rpp_header)
	recursive_find_files(str(folderName), ['.h', '.hpp'], files_IF_Series_header)
	recursive_find_files(str(folderName), ['.c', '.cpp'], files_IF_Series_cpp)


	for file in files_IF_Series_cpp:
		print("File: %s" %(file) )
		IF_PiloPa_header = ([Path(x).name for x in files_IF_PiloPa_header])
		IF_Pre2Rpp_header = ([Path(x).name for x in files_IF_Pre2Rpp_header])


		headersInFile = getHeaderFromFile(file)

		headers = []
		for header in headersInFile:

			if header in IF_PiloPa_header:
				headers.append('#include <IF_PiloPa/%s>' %(header))
			elif header in IF_Pre2Rpp_header:
				headers.append('#include <IF_Pre2Rpp/%s>' %(header))
			else:
				headers.append('#include "%s"' % (header))

		writeHeadersToFile(file, headers)


	








