import sys
import csv
from collections import defaultdict
import re

all_data_file = open("../Data/AllSSAData/all_ssa_data.csv","r")

allData = csv.reader(all_data_file)
cleanAllData = []
countryList = []
varList = []
yearList = [1960,1961,1962,1963,1964,1965,1966,1967,1968,1969,1970,1972,1973,1973,1974,1975,1976,1977,1978,1979,1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014]
for line in allData:
	cleanLine=[]
	for l in line:
		try:
			l=l.decode("utf8").encode('ascii','ignore')
			l = l.lower()
			cleanLine.append(l)
	 	except:
	 		try:
	 			l=l.encode('utf8')
	 			l = l.lower()
	 			cleanLine.append(l)
	 		except:
	 			l = l.lower()
	 			cleanLine.append(l)
	cleanAllData.append(cleanLine)
	if cleanLine[0]=="":
		break

header = cleanAllData.pop(0)

for line in cleanAllData:
	countryList.append(line[0])
	varList.append(line[2])


countryYearVarValDict = defaultdict(lambda:defaultdict(lambda: defaultdict(float)))

for line in cleanAllData:
	for yrIndex in range(4,len(header)):
		yrStr= header[yrIndex]
		yr= int(yrStr.strip().split(" ")[0])
		country = line[0]
		var = line[2]
		try:
			val = float(line[yrIndex])
		except:
			val = "NA"
		if country=="":
			continue

		countryYearVarValDict[country][yr][var] = val

countryList = set(countryList)
varList = set(varList)
varList.remove("")
countryList.remove("")

#removing countries based on insufficient data as observed from counts below
# countryList.remove("south sudan")
# countryList.remove("sao tome and principe")
# countryList.remove("eritrea")
# countryList.remove("tanzania")
# countryList.remove("guinea")
# countryList.remove("angola")
# countryList.remove("somalia")

varList.remove("literacy rate, adult total (% of people ages 15 and above)")
varList.remove("gini index (world bank estimate)")
varList.remove("government expenditure on education, total (% of government expenditure)")
varList.remove("central government debt, total (% of gdp)")
varList.remove("government expenditure on education, total (% of gdp)")
varList.remove("gross fixed capital formation, private sector (% of gdp)")
varList.remove("final consumption expenditure, etc. (annual % growth)")
varList.remove("gross capital formation (annual % growth)")
varList.remove("general government final consumption expenditure (annual % growth)")
varList.remove("exports of goods and services (annual % growth)")
varList.remove("gdp per capita growth (annual %)")
varList.remove("gdp per capita (constant 2005 us$)")
varList.remove("gdp (constant 2005 us$)")
varList.remove("gdp per capita (current us$)")
varList.remove("gdp (current us$)")


#determine availability of data for different parameters for different countries

# for v in varList:	
# 	sorted_data_size = []
# 	print "==========" + v + "=========="
# 	for c in countryList:
# 		countryNACount = 0
# 		for y in yearList:
# 			if countryYearVarValDict[c][y][v]=="NA":
# 				countryNACount+=1.0
# 		sorted_data_size.append((c, countryNACount))
# 	sorted_data_size = sorted(sorted_data_size, key = lambda x:x[1])
# 	for k in sorted_data_size:
# 		print k
# 	print "\n\n"

# keep track of countries for which we have a reasonable amount of data for all parameters
valid_countries = []
for c in countryList:
	skip = False
	for v in varList:
		countryNACount = 0
		for y in yearList:
			if countryYearVarValDict[c][y][v]=="NA":
				countryNACount+=1.0
		if countryNACount > 35:
			skip = True
	if not skip:
		valid_countries.append(c)


# print len(valid_countries)
# for x in valid_countries:
# 	print x

#look at data for pruned country list and parameter list to make sure data is avilable for the same years


# for v in varList:	
# 	sorted_data_size = []
# 	print "==========" + v + "=========="
# 	for c in valid_countries:
# 		print "::" + c + "::"
# 		values = []
# 		for y in yearList:
# 			if not countryYearVarValDict[c][y][v] == "NA":
# 				values.append((y, countryYearVarValDict[c][y][v]))
# 		for x in values:
# 			print x[0], x[1]

pruned_all_data_file = open("../Data/pruned_all_data.txt","r")
pruned_all_data = pruned_all_data_file.readlines()
pruned_all_data_file.close()
varMinDict = {}
varMaxDict = {}


for l in range(len(pruned_all_data)):
	line = pruned_all_data[l]
	line = line.strip()
	lineList = line.split(" ")
	if line[0]=="=":
		variable = re.sub('[=]','',line)
		varMinDict[variable]=(float("-inf"),'dummy')
		varMaxDict[variable] = (float("inf"),'dummy')
	else:
		if line[0]==":":
			countryMin = re.sub('[::]','',line)
			lineListMin =pruned_all_data[l+1].strip().split(" ")
			if int(lineListMin[0])>varMinDict[variable][0] and not (countryMin=="congo, dem. rep." or countryMin=="tanzania" or countryMin=="south africa"):
				varMinDict[variable] = (int(lineListMin[0]),countryMin)
			if not pruned_all_data[l-1][0]=="=":
				x=1
				while not pruned_all_data[l-x][0]==":":
					x+=1
				countryMax = re.sub('[::]','',pruned_all_data[l-x].strip())
				lineListMax = pruned_all_data[l-1].strip().split(" ")
				if int(lineListMax[0])<varMaxDict[variable][0] and not (countryMax=="chad" or countryMax=="sudan" or countryMax=="gabon"):
					varMaxDict[variable] = (int(lineListMax[0]),countryMax)		
mins = []
maxs = []
# for k in varMinDict:
# 	print "==========="+str(k)+"===========" + "\n" + str(varMinDict[k]) + ", " + str(varMaxDict[k]) + "\n"+ "Number of years of data available: " + str(varMaxDict[k][0]-varMinDict[k][0])
# 	print '\n'
# 	mins.append(varMinDict[k][0])
# 	maxs.append(varMaxDict[k][0])

# print max(mins),min(maxs), min(maxs)-max(mins)+1

valid_countries.remove("south africa")
valid_countries.remove("gabon")
valid_countries.remove("tanzania")
valid_countries.remove("congo, dem. rep.")
valid_countries.remove("sudan")
valid_countries.remove("chad")

# for v in varList:	
# 	sorted_data_size = []
# 	print "==========" + v + "=========="
# 	for c in valid_countries:
# 		print "::" + c + "::"
# 		values = []
# 		for y in yearList:
# 			if not countryYearVarValDict[c][y][v] == "NA":
# 				values.append((y, countryYearVarValDict[c][y][v]))
# 		for x in values:
# 			print x[0], x[1]

countryYearVarValDict['uganda'][1986]["foreign direct investment, net inflows (% of gdp)"] = 0
countryYearVarValDict['uganda'][1987]["foreign direct investment, net inflows (% of gdp)"] = 0
countryYearVarValDict['togo'][2008]["general government final consumption expenditure (% of gdp)"] = 10.1

cleanCountryYearVarValDict = defaultdict(lambda:defaultdict(lambda: defaultdict(float)))

for v in varList:	
	sorted_data_size = []
	# print "==========" + v + "=========="
	for c in valid_countries:
		# print "::" + c + "::"
		values = []
		for y in range(1986,2012):
			if not countryYearVarValDict[c][y][v] == "NA":
				cleanCountryYearVarValDict[c][y][v]= countryYearVarValDict[c][y][v]
				# print y,countryYearVarValDict[c][y][v]
			if countryYearVarValDict[c][y][v]== "NA":
				# print "HOLE!"
				pass

for i in cleanCountryYearVarValDict["cameroon"][1986].keys():
	print i

# for y in yearList:
# 	print str(countryYearVarValDict["uganda"][y]["gdp growth (annual %)"]) +", " + str(y)







	


