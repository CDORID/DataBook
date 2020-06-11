import pandas as pd
import numpy as np
from sklearn import metrics


class ModelKPI():
    def __init__(self):
        pass

    def compute_AUC(self,model,X_test,Y_test):
        yroc = Y_test
        predroc = model.predict_proba(X_test)[:,1]
        AUC = metrics.roc_auc_score(yroc, predroc)
        return AUC

    def compute(self,model,X_test,Y_test,loop = 1):

        model_name = str(str(loop) +"-"+ model.__class__.__name__)

        threshold_list = np.arange(0.0, 1.0, 0.005).tolist()
        accuracy_list = []
        PPV_list = []
        NPV_list = []
        TNR_list = []
        TPR_list = []

        pred_proba_df = pd.DataFrame(model.predict_proba(X_test))

        point_equilibrium = 0
        self.delta_tpr_tnr_equilibrium = 1


        for i in threshold_list:
                ## confusion matrix computation
                Y_test_pred = pred_proba_df.applymap(lambda x: 1 if x > i else 0)

                confusion = metrics.confusion_matrix(Y_test,Y_test_pred.iloc[:,1])

                # Confusion matrix identification
                TP = confusion[1,1]
                FP = confusion[0,1]
                TN = confusion[0,0]
                FN = confusion[1,0]

                # ratio computation
                NPV         = TN / (TN + FN)
                PPV         = TP / (TP + FP)
                accuracy    = (TP + TN) / (TP + FP + TN + FN)
                TNR         = TN / (FP + TN)
                TPR         = TP / (TP + FN)

                ### find point of equilibrium
                if abs(TPR-TNR) < self.delta_tpr_tnr_equilibrium:
                    self.point_equilibrium = np.mean([TPR,TNR])
                    self.delta_tpr_tnr_equilibrium = abs(TPR-TNR)

                ### AUC

                ### list appending to add the individual risk result
                NPV_list.append(NPV)
                PPV_list.append(PPV)
                accuracy_list.append(accuracy)
                TNR_list.append(TNR)
                TPR_list.append(TPR)


        ### Df for all the risks for the model
        self.NPV = pd.DataFrame({'risk':threshold_list, model_name : NPV_list})
        self.PPV = pd.DataFrame({'risk':threshold_list, model_name : PPV_list})
        self.accuracy = pd.DataFrame({'risk':threshold_list, model_name : accuracy_list})
        self.TNR = pd.DataFrame({'risk':threshold_list, model_name : TNR_list})
        self.TPR = pd.DataFrame({'risk':threshold_list, model_name : TPR_list})
        self.AUC = self.compute_AUC(model,X_test,Y_test)
        print(self.AUC)


class MultiModelKPI():
    def __init__(self):
        self.accuracy_df = pd.DataFrame({"risk":np.arange(0.0, 1.0, 0.005).tolist()})
        self.PPV_df = pd.DataFrame({"risk":np.arange(0.0, 1.0, 0.005).tolist()})
        self.NPV_df = pd.DataFrame({"risk":np.arange(0.0, 1.0, 0.005).tolist()})
        self.TNR_df = pd.DataFrame({"risk":np.arange(0.0, 1.0, 0.005).tolist()})
        self.TPR_df = pd.DataFrame({"risk":np.arange(0.0, 1.0, 0.005).tolist()})

        ### initializing list for individual kpis
        self.AUC_list = []
        self.fitted_models = []

    def multi_compute(self,models,X_train,X_test,Y_train,Y_test):
        i = 0
        for model in models:
            ## report models
            i = i+1
            model_name = str(str(i) +"-"+ model.__class__.__name__)
            print("Test : ", model_name)

            ## Train the models and create the prediction vector
            fitted_model = model.fit(X_train,Y_train)
            prediction_train = model.predict(X_train)

            ## compute KPIs
            modelmetrics = ModelKPI()
            modelmetrics.compute(fitted_model,X_test,Y_test,i)

            #Creating DF with results for all models
            self.accuracy_df     = pd.concat([self.accuracy_df, modelmetrics.accuracy],axis = 1)
            self.PPV_df          = pd.concat([self.PPV_df,      modelmetrics.PPV],axis = 1)
            self.NPV_df          = pd.concat([self.NPV_df,      modelmetrics.NPV],axis = 1)
            self.TNR_df          = pd.concat([self.TNR_df,      modelmetrics.TNR],axis = 1)
            self.TPR_df          = pd.concat([self.TPR_df,      modelmetrics.TPR],axis = 1)

            # Creating list for indiv KPIs /model
            self.AUC_list.append(modelmetrics.AUC)
            self.fitted_models.append(fitted_model)

        ## removing redundant columns of indexes
        self.accuracy_df    = self.accuracy_df.loc[:,~self.accuracy_df.columns.duplicated()]
        self.PPV_df         = self.PPV_df.loc[:,~self.PPV_df.columns.duplicated()]
        self.NPV_df         = self.NPV_df.loc[:,~self.NPV_df.columns.duplicated()]
        self.TNR_df         = self.TNR_df.loc[:,~self.TNR_df.columns.duplicated()]
        self.TPR_df         = self.TPR_df .loc[:,~self.TPR_df .columns.duplicated()]

        print("\n",i, ' Model Tested')

    def plot(self):

        import matplotlib.pyplot as plt
        plt.rc('font', family='serif')
        plt.rc('font', serif='Times New Roman')
        plt.rcParams.update({'font.size': 12})
        fig = plt.figure(figsize=(16,8))
        fig = self.accuracy_df.plot('risk')
        plt.ylabel('Percentage')
        plt.xlabel('Risk acceptance')
        plt.title('Accuracy measurements for the models tested')
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
