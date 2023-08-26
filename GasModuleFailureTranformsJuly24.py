import matplotlib.pyplot as plt
import random
import pandas as pd
from datetime import datetime as dt
import numpy as np
import seaborn as sns
import os
import statsmodels.api as sm
import sklearn.model_selection import train_test_split
pd.set_option('display.max_rows',500)


class GasModuleFailureModel():
    """ 
        Processing steps for the Gas Module Failure Model


    """

    def __init__(self):
        pass

    def LoadAndRemoveDupes(self):
        '''
            Load Dataset from csv files
            Remove dupes

        '''
        BVDF = pd.DataFrame(np.load('BVDF.npy'))
        BVDFCols = pd.read_csv('OWCEGasModuleBatteryHistoryAllJuly17_2023.rpt',nrows = 1)
        BVDF.columns = BVDFCols.columns
        #BVDF['Year'] = np.floor(np.floor(BVDF.UpdateTime / 1000) / 365 + 1970).astype('int32')
        BVDF.drop_duplicates(subset = ['ModuleId','SeriesId'],keep = 'last',inplace = True)
        #BVDF[['CurrentVoltage','LastReadVoltage','HistoricVoltage1','HistoricVoltage2' ]] = \
        #     BVDF[['CurrentVoltage','LastReadVoltage','HistoricVoltage1','HistoricVoltage2' ]] / 100000

        self.BVDF = BVDF
        np.save('BVDFNoDupes.npy',BVDF.to_numpy('int32'))

    def ConvertToNumpyTensor(self):
        self.BVDF = pd.DataFrame(np.load('BVDFNoDupes.npy'))
        self.BVDF.columns = pd.read_csv('OWCEGasModuleBatteryHistoryAllJuly17_2023.rpt',nrows = 1).columns
        self.BVDF.set_index(['ModuleId','SeriesId'], inplace = True)
        TempCols = ['CurrentTemperature', 'LastReadTemperature','HistoricTemperature1', 'HistoricTemperature2']
        VoltCols = ['CurrentVoltage', 'LastReadVoltage','HistoricVoltage1', 'HistoricVoltage2']
        self.BVDF = self.BVDF[TempCols + VoltCols]

        TempColList = [('Temp',str(Cnt)) for Cnt in range(1,5)]
        VoltColList = [('Volt',str(Cnt)) for Cnt in range(1,5)]
        # create multi-index for temp and voltage
        #
        self.BVDF.columns = pd.MultiIndex.from_tuples(TempColList + VoltColList) 

#    def Load[(str(Cnt),'Temp') for Cnt in range(1,5)]

    
    def OLSTest(DF):
        xM = DF[['CurrentTemperature','SeriesId']]
        xM = sm.add_constant(xM)
        yM = DF.CurrentVoltage
        model = sm.OLS(yM,xM) 
        fitted = model.fit()
        return fitted.params
 
    def ComputeOLSCoef():
        ModList = Volt.index.to_list()
        ResultsDict = {} 
        LastTime = dt.now()
        for ModCnt,ModNum in enumerate(ModList):
            if ((ModCnt % 10000) == 0):
                print(str(ModCnt) + '.........' +  str(dt.now()) + '.....' + str(dt.now() - LastTime))
                Tmp = pd.DataFrame.from_dict(ResultsDict)
                Tmp.T.to_csv( 'SimOut/Coef_' + str(ModCnt) + '.csv',header = None)
                ResultsDict = {}
                LastTime = dt.now()
        
            Tmp = BVDF[BVDF.ModuleId == ModNum]
            ResultsDict[ModNum] = OLSTest(Tmp)


    def UnivariateTransformations(self,X):
        return [X.min(axis = 1),X.max(axis = 1)]


    def GetTrainTestSplit(self):
        X_train, X_test, y_train, y_test = train_test_split(

        X_train.shape, y_train.shape
        X_test.shape, y_test.shape

        clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
        clf.score(X_test, y_test)


            
    def ComputeOLSCoef_2(self,SampleSize = 1000):
        """
        Models for predicting battery failure
        basic regression model - uses multiple factors in order to 'explain' data. this means that
        the isample points in the model 'fit'. Variations in temerature contribute to variaions in voltage.
        We want to use voltage as a proxy for health.

        Forecasting voltage is a slighlty different problem:  we want to minimize
        consuming battery voltage at a greater rate than forecast - indicates a failure
        need to gather performance/diagnotic statist
        failure analysis - deviations from expected value inducate a failure condition is at work
         idicate a fialure condition 
         lar understanding of  it helps: truck rolls
         run competing models and evaluate forecasting error
         useful for characterizing 

        LSTM?
        arima

        * run many models and compare results
            - volt only
            - temp and volt



        Parameters
        ----------



        Returns
        -------


        """
        self.ModuleList = self.BVDF.index.get_level_values(level = 0).unique().to_list() 
        self.TestList = random.sample(self.ModuleList,SampleSize)
        #ModList = index.to_list()
        #ResultsDict = {} 
        #LastTime = dt.now()
        #for ModCnt,ModNum in enumerate(ModList):
        #    if ((ModCnt % 10000) == 0):
        #        print(str(ModCnt) + '.........' +  str(dt.now()) + '.....' + str(dt.now() - LastTime))
        #        Tmp = pd.DataFrame.from_dict(ResultsDict)
        #        Tmp.T.to_csv( 'SimOut/Coef_' + str(ModCnt) + '.csv',header = None)
        #        ResultsDict = {}
        #        LastTime = dt.now()
        # 
        #     Tmp = BVDF[BVDF.ModuleId == ModNum]
        #     ResultsDict[ModNum] = OLSTest(Tmp)



print('Create GF object')
GF = GasModuleFailureModel()
GF.ConvertToNumpyTensor()
GF.ComputeOLSCoef_2()
print('Remove Dupes and Create  ')
#GF.LoadAndRemoveDupes()
#GF.ConvertToNumpyTensor()
#GF.ProcessVoltage_1()
