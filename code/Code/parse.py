import sys
import csv
from collections import defaultdict
import re
from random import randint
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, svm
from sklearn.decomposition import PCA
from pylab import *
from mpl_toolkits.mplot3d.axes3d import Axes3D
import math
import matplotlib.patches as mpatches

#aggregate data file
all_data_file = open("../Data/AllSSAData/all_ssa_data.csv","r")

###############################PARSING###############################
allData = csv.reader(all_data_file)
cleanAllData = []
countryList = []
varList = []
yearList = range(1960, 2015)
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
def removeInvalidCountries():
  countryList.remove("south sudan")
  countryList.remove("sao tome and principe")
  countryList.remove("eritrea")
  countryList.remove("tanzania")
  countryList.remove("guinea")
  countryList.remove("angola")
  countryList.remove("somalia")

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
def detAvail():
  for v in varList: 
    sorted_data_size = []
    print "==========" + v + "=========="
    for c in countryList:
        countryNACount = 0
        for y in yearList:
            if countryYearVarValDict[c][y][v]=="NA":
                countryNACount+=1.0
        sorted_data_size.append((c, countryNACount))
    sorted_data_size = sorted(sorted_data_size, key = lambda x:x[1])
    for k in sorted_data_size:
        print k
    print "\n\n"

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

def printCtrys():
  print len(valid_countries)
  for x in valid_countries:
    print x

#look at data for pruned country list and parameter list to make sure data is avilable for the same years

def printPruned():

  for v in varList: 
    sorted_data_size = []
    print "==========" + v + "=========="
    for c in valid_countries:
        print "::" + c + "::"
        values = []
        for y in yearList:
            if not countryYearVarValDict[c][y][v] == "NA":
                values.append((y, countryYearVarValDict[c][y][v]))
        for x in values:
            print x[0], x[1]


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
  for k in varMinDict:
    print "==========="+str(k)+"===========" + "\n" + str(varMinDict[k]) + ", " + str(varMaxDict[k]) + "\n"+ "Number of years of data available: " + str(varMaxDict[k][0]-varMinDict[k][0])
    print '\n'
    mins.append(varMinDict[k][0])
    maxs.append(varMaxDict[k][0])

  print max(mins),min(maxs), min(maxs)-max(mins)+1

valid_countries.remove("south africa")
valid_countries.remove("gabon")
valid_countries.remove("tanzania")
valid_countries.remove("congo, dem. rep.")
valid_countries.remove("sudan")
valid_countries.remove("chad")

def printPruned_2():

  for v in varList: 
    sorted_data_size = []
    print "==========" + v + "=========="
    for c in valid_countries:
        print "::" + c + "::"
        values = []
        for y in yearList:
            if not countryYearVarValDict[c][y][v] == "NA":
                values.append((y, countryYearVarValDict[c][y][v]))
        for x in values:
            print x[0], x[1]

#update year list after looking at valid years
yearList = range(1986, 2014) 

countryYearVarValDict['uganda'][1986]["foreign direct investment, net inflows (% of gdp)"] = countryYearVarValDict['uganda'][1985]["foreign direct investment, net inflows (% of gdp)"] + 1.0/3 * (countryYearVarValDict['uganda'][1988]["foreign direct investment, net inflows (% of gdp)"] - countryYearVarValDict['uganda'][1985]["foreign direct investment, net inflows (% of gdp)"]) 
countryYearVarValDict['uganda'][1987]["foreign direct investment, net inflows (% of gdp)"] = countryYearVarValDict['uganda'][1985]["foreign direct investment, net inflows (% of gdp)"] + 2.0/3 * (countryYearVarValDict['uganda'][1988]["foreign direct investment, net inflows (% of gdp)"] - countryYearVarValDict['uganda'][1985]["foreign direct investment, net inflows (% of gdp)"]) 
countryYearVarValDict['togo'][2008]["general government final consumption expenditure (% of gdp)"] = 1.0/2 * (countryYearVarValDict['togo'][2007]["general government final consumption expenditure (% of gdp)"] + countryYearVarValDict['togo'][2009]["general government final consumption expenditure (% of gdp)"])

cleanCountryYearVarValDict = defaultdict(lambda:defaultdict(lambda: defaultdict(float)))

for v in varList:   
    sorted_data_size = []
    # print "==========" + v + "=========="
    for c in valid_countries:
        # print "::" + c + "::"
        values = []
        for y in yearList:
            if not countryYearVarValDict[c][y][v] == "NA":
                cleanCountryYearVarValDict[c][y][v]= countryYearVarValDict[c][y][v]
                #convert to % of gdp - different USD values don't matter
                if v == "net official development assistance and official aid received (constant 2012us$)":
                    cleanCountryYearVarValDict[c][y]["net official development assistance and official aid received (% of gdp)"] = 100.0 * float(countryYearVarValDict[c][y][v])/float(countryYearVarValDict[c][y]["gdp (constant 2005 us$)"])
                # print y,countryYearVarValDict[c][y][v]

varList = cleanCountryYearVarValDict["cameroon"][1986].keys() #get updated variable list
varList.remove("net official development assistance and official aid received (constant 2012us$)")

def printFinalPruned():
  for v in varList: 
    sorted_data_size = []
    print "==========" + v + "=========="
    for c in valid_countries:
        print "::" + c + "::"
        for y in yearList:
            print y, cleanCountryYearVarValDict[c][y][v]

  for y in yearList:
    print str(countryYearVarValDict["uganda"][y]["gdp growth (annual %)"]) +", " + str(y)

# # # # # # # # # # # # # # # # # # # # # # # # # # REGRESSION # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #



varList.remove("gdp growth (annual %)")


regr_independent_vars = [
'final consumption expenditure, etc. (% of gdp)', 
'exports of goods and services (% of gdp)', 
# 'general government final consumption expenditure (constant 2005 us$)', 
'population growth (annual %)', 
'population, total', 
#'gross capital formation (% of gdp)', 
# 'exports of goods and services (constant 2005 us$)', 
#'final consumption expenditure, etc. (constant 2005 us$)', 
# 'gdp growth (annual %)', 
# 'net official development assistance and official aid received (constant 2012us$)', 
'net official development assistance and official aid received (% of gdp)', 
# 'gross capital formation (constant 2005 us$)', 
# 'gross national expenditure (constant 2005 us$)', 
'foreign direct investment, net inflows (% of gdp)', 
# 'gross national expenditure (% of gdp)', 
# 'general government final consumption expenditure (% of gdp)'
]

regr_independent_vars_2 = [
'final consumption expenditure, etc. (% of gdp)', 
'exports of goods and services (% of gdp)', 
# 'general government final consumption expenditure (constant 2005 us$)', 
'population growth (annual %)', 
#'population, total', 
#'gross capital formation (% of gdp)', 
# 'exports of goods and services (constant 2005 us$)', 
#'final consumption expenditure, etc. (constant 2005 us$)', 
# 'gdp growth (annual %)', 
# 'net official development assistance and official aid received (constant 2012us$)', 
'net official development assistance and official aid received (% of gdp)', 
# 'gross capital formation (constant 2005 us$)', 
# 'gross national expenditure (constant 2005 us$)', 
'foreign direct investment, net inflows (% of gdp)', 
# 'gross national expenditure (% of gdp)', 
# 'general government final consumption expenditure (% of gdp)'
]

all_vars = regr_independent_vars + ["gdp growth (annual %)"]
dataGDP = []
mingrowth = float("inf")
maxgrowth = float("-inf")
for c in valid_countries:
    cAdd = c
    if c=="congo, rep.":
        cAdd="congo"
    tup = (cAdd,)
    for y in range(1986,2014):
        tup += (cleanCountryYearVarValDict[c][y]["gdp growth (annual %)"],)
        if mingrowth>cleanCountryYearVarValDict[c][y]["gdp growth (annual %)"]:
            mingrowth= cleanCountryYearVarValDict[c][y]["gdp growth (annual %)"]
        if maxgrowth<cleanCountryYearVarValDict[c][y]["gdp growth (annual %)"]:
            maxgrowth = cleanCountryYearVarValDict[c][y]["gdp growth (annual %)"]
    dataGDP.append(tup)


# #Extra countries just for GDP growth
def gdpExtraCtrys():
  for c in ["south africa","sudan","congo, dem. rep.","central african republic","angola","zimbabwe","zambia","burundi","chad","burundi","malawi","gabon","tanzania","south sudan","ethiopia","ghana","djibouti","eritrea","niger","mali","sierra leone","cote d'ivoire","liberia","guinea bissau","guinea"]:
    cAdd = c
    if c=="congo, dem. rep.":
        cAdd = "dem. rep. congo"
    if c=="central african republic":
        cAdd = "central african rep."
    if c=="south sudan":
        cAdd = "s. sudan"
    tup = (cAdd,)
    for y in range(1986,2014):
        tup+= (countryYearVarValDict[c][y]["gdp growth (annual %)"],)
    dataGDP.append(tup)

def writeCSVandViz():
  gdpExtraCtrys()
  with open('../Data/gdpGrowth.csv','w') as output:
    csv_out = csv.writer(output)
    csv_out.writerow(['country','1986','1987','1988','1989','1990','1991','1992','1993','1994','1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013'])
    for row in dataGDP:
        csv_out.writerow(row)



#### Simple Regression against one independent variable mapped to one dependent variable####
def simpleRegr():
    indeVarDict = defaultdict(lambda: defaultdict(float))

    odaCofDict = defaultdict(float)
    fdiCofDict = defaultdict(float)

    for c in valid_countries:
      regr_independent_data = []
      for v in regr_independent_vars:
          indeVarDict[c][v]=[]

      regr_dependent_data = []
      lag2_regr_dependent_data = []
      lag2_regr_independent_data = []
      for y in range(1987, 2013): #one year lag
          regr_dependent_data.append(cleanCountryYearVarValDict[c][y]["gdp growth (annual %)"])
      for y in range(1988, 2014): #two year lag
          lag2_regr_dependent_data.append(cleanCountryYearVarValDict[c][y]["gdp growth (annual %)"])
      for y in range(1986, 2012):
          values = []
          for v in regr_independent_vars:
              values.append(cleanCountryYearVarValDict[c][y][v])
              indeVarDict[c][v].append(cleanCountryYearVarValDict[c][y][v])
          regr_independent_data.append(tuple(values))
      for y in range(1986, 2012):
          values = []
          for v in regr_independent_vars:
              values.append(cleanCountryYearVarValDict[c][y][v])
          lag2_regr_independent_data.append(tuple(values))        
      regr = linear_model.LinearRegression()
      lag2_regr = linear_model.LinearRegression()

      lag2_regr.fit(lag2_regr_independent_data,lag2_regr_dependent_data)

      regr.fit(regr_independent_data, regr_dependent_data)

      odaCofDict[c] = regr.coef_[3]
      fdiCofDict[c] = regr.coef_[4]

      print "===================Coefficients for " + c + "==================="
      print regr_independent_vars
      print regr.coef_
      print "\n"

      print "===================Coefficients for lag2 " + c + "==================="
      print regr_independent_vars
      print lag2_regr.coef_
      print "\n\n"

    ### VISUALIZATION ###

    # for v in regr_independent_vars:
    #   for c in valid_countries:
    #       fig = plt.figure()
    #       fig.suptitle(c + ' ' + v + " vs GDP growth %", fontsize=14, fontweight='bold')
    #       ax = fig.add_subplot(111)
    #       ax.set_xlabel(v)
    #       ax.set_ylabel('GDP growth %')
    #       x = np.array(indeVarDict[c][v])
    #       y = []
    #       for yr in range(1987, 2013):    #one year lag
    #           y.append(cleanCountryYearVarValDict[c][yr]["gdp growth (annual %)"])
    #       y = np.array(y) 
    #       fit = np.polyfit(x,y,deg=1)
    #       color = "#%06x" % (randint(0,0xFFFFFF))
    #       plt.plot(x, fit[0] * x + fit[1], color=color)
    #       plt.scatter(x, y)
    #       #plt.savefig("../Visualizations/SimpleRegr/"+c+'_'+v+'_'+'vs gdp_growth.png')
    #       plt.show()

#### Regression against multiple independent variables mapped to one dependent variable (with population in numbers) ####
def multipleRegr_1():
    indeVarDict = defaultdict(lambda: defaultdict(float))

    for c in valid_countries:
        regr_independent_data = []
        for v in regr_independent_vars:
            indeVarDict[c][v]=[]

        regr_dependent_data = []
        pred_dependent_data = []
        pred_independent_data = []
        lag2_regr_dependent_data = []
        lag2_regr_independent_data = []
        for y in range(1987, 2013): #one year lag
            regr_dependent_data.append(cleanCountryYearVarValDict[c][y]["gdp growth (annual %)"])
        for y in range(1987, 2012):
            pred_dependent_data.append(cleanCountryYearVarValDict[c][y]["gdp growth (annual %)"])
        for y in range(1988, 2014): #two year lag
            lag2_regr_dependent_data.append(cleanCountryYearVarValDict[c][y]["gdp growth (annual %)"])
        for y in range(1986, 2012):
            values = []
            for v in regr_independent_vars:
                values.append(cleanCountryYearVarValDict[c][y][v])
                indeVarDict[c][v].append(cleanCountryYearVarValDict[c][y][v])
            regr_independent_data.append(tuple(values))
        for y in range(1986, 2011):
            values = []
            for v in regr_independent_vars:
                values.append(cleanCountryYearVarValDict[c][y][v])
            pred_independent_data.append(tuple(values))
        for y in range(1986, 2012):
            values = []
            for v in regr_independent_vars:
                values.append(cleanCountryYearVarValDict[c][y][v])
            lag2_regr_independent_data.append(tuple(values))
            
        regr = linear_model.LinearRegression()
        pred_regr = linear_model.LinearRegression()
        lag2_regr = linear_model.LinearRegression()
        regu_regr = linear_model.LinearRegression()

        pca1 = PCA(n_components=2)
        regr_independent_data = np.array(regr_independent_data)
        pca1.fit(regr_independent_data)
        dimensionalityReducedInput = np.array(pca1.transform(regr_independent_data))

        pca2 = PCA(n_components=2)
        lag2_regr_independent_data = np.array(lag2_regr_independent_data)
        pca2.fit(lag2_regr_independent_data)
        lag2_dimensionalityReducedInput = np.array(pca2.transform(lag2_regr_independent_data))

        pcapred = PCA(n_components=2)
        pred_independent_data = np.array(pred_independent_data)
        pcapred.fit(pred_independent_data)
        pred_dimensionalityReducedInput = np.array(pcapred.transform(pred_independent_data))

        lag2_regr.fit(lag2_dimensionalityReducedInput,lag2_regr_dependent_data)

        pred_regr.fit(pred_dimensionalityReducedInput,pred_dependent_data)

        regr.fit(dimensionalityReducedInput, regr_dependent_data)

        regu_regr.fit(pred_independent_data, pred_dependent_data)

        print "===================Principal Component Coefficients for " + c + "==================="
        print regr_independent_vars
        print regr.coef_
        print dimensionalityReducedInput

        values = []
        for v in regr_independent_vars:
          values.append(cleanCountryYearVarValDict[c][2011][v])
        valTuple = [tuple(values)]
        val = [values,[0,0,0,0,0,0]]
        pcaTest = PCA(n_components=2)
        val = np.array(val)
        test_dimensionalityReducedInput = np.array(pcaTest.fit_transform(val))
        print test_dimensionalityReducedInput
        print val
        print "Predicted Growth for "+ c.title()+ " in 2011 is " + str(regu_regr.predict(valTuple)[0]) +"%"
        print "Predicted Growth for "+ c.title()+ " in 2011 is " + str(pred_regr.predict(test_dimensionalityReducedInput)[0]) +"%"
        print "Actual Growth for "+ c.title()+ " in 2011 is " + str(cleanCountryYearVarValDict[c][2011]["gdp growth (annual %)"]) +"%"
        print "\n"

        ### VISUALIZATION ###
        comps = pca1.components_
 
        # I've omitted the code to create ind; a list of the indexes of the
        # loadings ordered by distance from origin.
        ind = []
        for i in comps:
          for a in i:
            ind.append(a)
        sorted(ind)
        plt.scatter(*comps, alpha=0.3, label="Components");
        plt.scatter(*comps[:, ind[:3]], c='r', marker='o',
                     s=80, linewidths=1, facecolors="none", 
                     edgecolors='r', 
                     label="Most Variance Caused");

        print cleanCountryYearVarValDict.keys()
         
        plt.title("Components plot");
        plt.xlabel("Components: Principal Component 1");
        plt.ylabel("Components: Principal Compnent 2");
        plt.grid();
        plt.legend(loc='lower right');
        plt.show()

        X = []
        Y = []
        Z = regr_dependent_data

        for i in dimensionalityReducedInput:
            X.append(i[0])
            Y.append(i[1])

        fig = plt.figure(figsize=plt.figaspect(0.5))

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
        ax.set_title("Regression for one-year lag: " + c.title())
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_zlabel("GDP growth (%)")

        # plt.savefig("../Visualizations/MultipleRegrWithoutPopulationInNumber/"+c+'_independent_variables_one_year_lag'+'vs_gdp_growth.png')
        plt.show()

        print "===================Principal Component Coefficients for lag2 " + c + "==================="
        print regr_independent_vars
        print lag2_regr.coef_
        print lag2_dimensionalityReducedInput
        print "\n\n"

        ### VISUALIZATION ###


        X = []
        Y = []
        Z = lag2_regr_dependent_data

        for i in lag2_dimensionalityReducedInput:
            X.append(i[0])
            Y.append(i[1])

        fig = plt.figure(figsize=plt.figaspect(0.5))

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
        ax.set_title("Regression for two-year lag: " + c.title())
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_zlabel("GDP growth (%)")

        # plt.savefig("../Visualizations/MultipleRegrWithoutPopulationInNumber/"+c+'_independent_variables_two_year_lag'+'vs_gdp_growth.png')
        plt.show()

#### Regression against multiple independent variables mapped to one dependent variable (without population in numbers) ####
def multipleRegr_2():
    indeVarDict = defaultdict(lambda: defaultdict(float))

    for c in valid_countries:
        regr_independent_data = []
        for v in regr_independent_vars_2:
            indeVarDict[c][v]=[]
        regr_dependent_data = []
        lag2_regr_dependent_data = []
        lag2_regr_independent_data = []
        for y in range(1987, 2013): #one year lag
            regr_dependent_data.append(cleanCountryYearVarValDict[c][y]["gdp growth (annual %)"])
        for y in range(1988, 2014): #two year lag
            lag2_regr_dependent_data.append(cleanCountryYearVarValDict[c][y]["gdp growth (annual %)"])
        for y in range(1986, 2012):
            values = []
            for v in regr_independent_vars_2:
                values.append(cleanCountryYearVarValDict[c][y][v])
                indeVarDict[c][v].append(cleanCountryYearVarValDict[c][y][v])
            regr_independent_data.append(tuple(values))
        for y in range(1986, 2012):
            values = []
            for v in regr_independent_vars_2:
                values.append(cleanCountryYearVarValDict[c][y][v])
            lag2_regr_independent_data.append(tuple(values))

        regr = linear_model.LinearRegression()
        lag2_regr = linear_model.LinearRegression()

        pca1 = PCA(n_components=2)
        regr_independent_data = np.array(regr_independent_data)
        pca1.fit(regr_independent_data)
        dimensionalityReducedInput = np.array(pca1.transform(regr_independent_data))

        pca2 = PCA(n_components=2)
        lag2_regr_independent_data = np.array(lag2_regr_independent_data)
        pca2.fit(lag2_regr_independent_data)
        lag2_dimensionalityReducedInput = np.array(pca2.transform(lag2_regr_independent_data))

        lag2_regr.fit(lag2_dimensionalityReducedInput,lag2_regr_dependent_data)

        regr.fit(dimensionalityReducedInput, regr_dependent_data)

        print "===================Principal Component Coefficients for " + c + "==================="
        print regr_independent_vars
        print regr.coef_
        print dimensionalityReducedInput
        print "\n"

        ### VISUALIZATION ###


        X = []
        Y = []
        Z = regr_dependent_data

        for i in dimensionalityReducedInput:
            X.append(i[0])
            Y.append(i[1])

        fig = plt.figure(figsize=plt.figaspect(0.5))

        ax = fig.add_subplot(1, 2, 1, projection='3d')
        X, Y = np.meshgrid(X, Y)
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)
        ax.set_title("Regression for one-year lag: " + c.title())
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_zlabel("GDP growth (%)")

        fig.colorbar(surf, shrink=0.5, aspect=10)

        # plt.savefig("../Visualizations/MultipleRegrWithPopulationInNumber/"+c+'_independent_variables_one_year_lag'+'vs_gdp_growth.png')
        plt.show()

        print "===================Principal Component Coefficients for lag2 " + c + "==================="
        print regr_independent_vars
        print lag2_regr.coef_
        print lag2_dimensionalityReducedInput
        print "\n\n"

        ### VISUALIZATION ###


        X = []
        Y = []
        Z = lag2_regr_dependent_data

        for i in lag2_dimensionalityReducedInput:
            X.append(i[0])
            Y.append(i[1])

        fig = plt.figure(figsize=plt.figaspect(0.5))
        
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        X, Y = np.meshgrid(X, Y)
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)
        ax.set_title("Regression for two-year lag: " + c.title())
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_zlabel("GDP growth (%)")

        # plt.savefig("../Visualizations/MultipleRegrWithPopulationInNumber/"+c+'_independent_variables_two_year_lag'+'vs_gdp_growth.png')
        fig.colorbar(surf, shrink=0.5, aspect=10)

        plt.show()



# # # # # # # # # # # # # # # # # # # # # # # # SUPPORT VECTOR MACHINE # # # # # # # # # # # # # # # # # # # # # # # # 
svm_input_vars = [
'gdp growth (annual %)',
'final consumption expenditure, etc. (% of gdp)', 
'exports of goods and services (% of gdp)', 
# 'general government final consumption expenditure (constant 2005 us$)', 
'population growth (annual %)', 
#'population, total', 
# 'gross capital formation (% of gdp)' 
#'exports of goods and services (constant 2005 us$)', 
# 'final consumption expenditure, etc. (constant 2005 us$)',
#'gross capital formation (constant 2005 us$)' 
# 'gross national expenditure (constant 2005 us$)',
'gross national expenditure (% of gdp)',
#'gdp (constant 2005 us$)'
# 'general government final consumption expenditure (% of gdp)'  
]

#####FUNCTION DID NOT WORK BECAUSE DATA WAS TOO BIASED FOR THE SVM TO LEARN THE PARAMETERS TO LABEL THE COUNTRIES FDI OR ODA SPECIFICALLY PER YEAR. MOST COUNTRIES ARE FDI IN ALL YEARS#####
def oneVarSVM():
    inputVarDict = defaultdict(lambda: defaultdict(float))
    svcDict = defaultdict(float)
    for c in valid_countries:
      svm_input_data = []
      svm_class_data = []
      svm_oda_input = []
      svm_fdi_input = []
      for v in svm_input_vars:
          inputVarDict[c][v]=[]
      for y in range(1986, 2011): #one year lag
          classs = 0 #oda
          if odaCofDict[c]<fdiCofDict[c]:
              classs = 1 #fdi
          svm_class_data.append(classs)
      for y in range(1986, 2011):
          values = []
          for v in svm_input_vars:
              values.append(countryYearVarValDict[c][y][v])
              inputVarDict[c][v].append(countryYearVarValDict[c][y][v])
          svm_input_data.append(tuple(values))
          if odaCofDict[c]<fdiCofDict[c]:
              svm_fdi_input.append(tuple(values))
          else:
              svm_oda_input.append(tuple(values))
      if not 0 in set(svm_class_data):
          continue
      elif not 1 in set(svm_class_data):
          continue

      pca = PCA(n_components=2)
      pca.fit(svm_input_data)
      print pca.explained_variance_ratio_
      pca.fit(svm_fdi_input)
      print pca.explained_variance_ratio_
      print "########## " + c + "#######"


    dimensionalityReducedInput = pca.transform(svm_input_data) 

    C = 1.0 #regularization parameter for the SVM
    svm_input_data = np.array(svm_input_data)
    svm_class_data = np.array(svm_class_data)
    dimensionalityReducedInput = np.array(dimensionalityReducedInput)
    svc = svm.SVC().fit(dimensionalityReducedInput, svm_class_data)
    svcDict[c] = svc
    

    fig = plt.figure()
    x1_samples = np.array(svm_oda_input)
    x2_samples = np.array(svm_fdi_input)

    plt.scatter(x1_samples[:,0],x1_samples[:,1], marker='+')
    plt.scatter(x2_samples[:,0],x2_samples[:,1], c= 'green', marker='o')

    X = np.concatenate((x1_samples,x2_samples), axis = 0)
    Y = np.array(svm_class_data)
    Y = np.array([0]*13 + [1]*12)

    C = 1.0  # SVM regularization parameter
    clf = svm.SVC(kernel = 'rbf',  gamma=0.7, C=C )
    clf.fit(X, Y)

    h = .02  # step size in the mesh
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))


    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
    plt.show()
    ########################################################################################################

############################### FUNCTION FOR MULTICLASS SVM, "FDI WILL GROW", "ODA WILL GROW", "BOTH WILL GROW", and "NONE WILL GROW" depending on patterns in economy per year #################
def highDimSVM():
    inputVarDict = defaultdict(lambda: defaultdict(float))
    classDict = {}
    classDict[0] = "Neither should grow"
    classDict[1] = "FDI should grow"
    classDict[2] = "ODA should grow"
    classDict[3] = "Both should grow"
    svcDict = defaultdict(float)
    valid_countries.pop(0)
    valid_countries.remove("equatorial guinea")
    for c in valid_countries:
        svm_input_data = []
        svm_class_data = []
        for v in svm_input_vars:
            inputVarDict[c][v]=[]
        for y in range(1986, 2011): #one year lag
            classs = 0 #oda
            if cleanCountryYearVarValDict[c][y+1]['net official development assistance and official aid received (% of gdp)']-cleanCountryYearVarValDict[c][y]['net official development assistance and official aid received (% of gdp)']>0:
                if cleanCountryYearVarValDict[c][y+1]['foreign direct investment, net inflows (% of gdp)']-cleanCountryYearVarValDict[c][y]['foreign direct investment, net inflows (% of gdp)']>0:
                    classs = 3 #both grow
                else:
                    classs = 2 #oda grows
            elif cleanCountryYearVarValDict[c][y+1]['foreign direct investment, net inflows (% of gdp)']-cleanCountryYearVarValDict[c][y]['foreign direct investment, net inflows (% of gdp)']>0:
                    classs = 1 #fdi grows
            else:
                classs = 0 #none grow
            svm_class_data.append(classs)
        for y in range(1986, 2011):
            values = []
            for v in svm_input_vars:
                values.append(countryYearVarValDict[c][y][v])
                inputVarDict[c][v].append(countryYearVarValDict[c][y][v])
            svm_input_data.append(tuple(values))

        pca3 = PCA(n_components=2)
        svm_input_data = np.array(svm_input_data)
        pca3.fit(svm_input_data)
        dimensionalityReducedInput = np.array(pca3.transform(svm_input_data))
        classLabels = np.array(svm_class_data) 

        print dimensionalityReducedInput
        print classLabels
        print "########## " + c + "#######"

        X = dimensionalityReducedInput
        y = classLabels

        h = .02  # step size in the mesh



        # plt.scatter(dimensionalityReducedInput[:,0],dimensionalityReducedInput[:,1], marker='+')
        C = 1.0  # SVM regularization parameter


        clf_rbf = svm.SVC(kernel = 'rbf',  gamma=0.7, C=C).fit(X, y)
        clf_linearonevsone = svm.SVC(kernel='linear', C=C).fit(X, y)
        clf_poly = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
        clf_linearonevsall = svm.LinearSVC(C=C).fit(X, y)

        values = []
        for v in svm_input_vars:
          values.append(cleanCountryYearVarValDict[c][2011][v])
        valTuple = [tuple(values)]
        val = [values,[0,0,0,0,0]]
        pcaTest = PCA(n_components=2)
        val = np.array(val)
        test_dimensionalityReducedInput = np.array(pcaTest.fit_transform(val))

        print "RBF SVM's prediction for 2011 is " + classDict[clf_rbf.predict(test_dimensionalityReducedInput)[0]]
        print "Linear One Vs One SVM's prediction for 2011 is " + classDict[clf_linearonevsone.predict(test_dimensionalityReducedInput)[0]]
        print "Linear One Vs All SVM's prediction for 2011 is " + classDict[clf_linearonevsall.predict(test_dimensionalityReducedInput)[0]]
        print "3-degree Polynomial SVM's prediction for 2011 is " + classDict[clf_poly.predict(test_dimensionalityReducedInput)[0]]
        print "actual oda change " + str(cleanCountryYearVarValDict[c][2011]['net official development assistance and official aid received (% of gdp)']-cleanCountryYearVarValDict[c][2010]['net official development assistance and official aid received (% of gdp)'])
        print "actual fdi change " + str(cleanCountryYearVarValDict[c][2011]['foreign direct investment, net inflows (% of gdp)']-cleanCountryYearVarValDict[c][2010]['foreign direct investment, net inflows (% of gdp)'])

        # create a mesh to plot in
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h),copy=False)

            # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        Z = clf_rbf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(c+": SVM RBF")
        p1 = Rectangle((0, 0), 1, 1, fc="b")
        p2 = Rectangle((0, 0), 1, 1, fc="g")
        p3 = Rectangle((0, 0), 1, 1, fc="r")
        p4 = Rectangle((0, 0), 1, 1, fc="y")
        fig.legend(handles=[p1,p2,p3,p4],labels=["Neither expected to grow","FDI expected to grow","ODA expected to grow","Both expected to grow"],loc='upper right')
        #plt.savefig("../Visualizations/HighDimSVM/RBG/"+c+'.png')
        plt.show()

        Z = clf_linearonevsone.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title(c+": SVM Linear-One vs One")
        plt.xticks(())
        plt.yticks(())
        p1 = Rectangle((0, 0), 1, 1, fc="b")
        p2 = Rectangle((0, 0), 1, 1, fc="g")
        p3 = Rectangle((0, 0), 1, 1, fc="r")
        p4 = Rectangle((0, 0), 1, 1, fc="y")
        fig.legend(handles=[p1,p2,p3,p4],labels=["Neither expected to grow","FDI expected to grow","ODA expected to grow","Both expected to grow"],loc='upper right')
        #plt.savefig("../Visualizations/HighDimSVM/LinearOneVsOne/"+c+'.png')
        plt.show()

        Z = clf_linearonevsall.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title(c+": SVM Linear-One vs All")
        plt.xticks(())
        plt.yticks(())
        p1 = Rectangle((0, 0), 1, 1, fc="b")
        p2 = Rectangle((0, 0), 1, 1, fc="g")
        p3 = Rectangle((0, 0), 1, 1, fc="r")
        p4 = Rectangle((0, 0), 1, 1, fc="y")
        fig.legend(handles=[p1,p2,p3,p4],labels=["Neither expected to grow","FDI expected to grow","ODA expected to grow","Both expected to grow"],loc='upper right')
        #plt.savefig("../Visualizations/HighDimSVM/LinearOneVsAll/"+c+'.png')
        plt.show()

        Z = clf_poly.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title(c+": SVM 3-Degree Polynomial")
        plt.xticks(())
        plt.yticks(())
        p1 = Rectangle((0, 0), 1, 1, fc="b")
        p2 = Rectangle((0, 0), 1, 1, fc="g")
        p3 = Rectangle((0, 0), 1, 1, fc="r")
        p4 = Rectangle((0, 0), 1, 1, fc="y")
        fig.legend(handles=[p1,p2,p3,p4],labels=["Neither expected to grow","FDI expected to grow","ODA expected to grow","Both expected to grow"],loc='upper right')
        #plt.savefig("../Visualizations/HighDimSVM/Poly/"+c+'.png')
        plt.show()
simpleRegr()















    


