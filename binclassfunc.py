import pandas as pd
import numpy as np
from sklearn import metrics


class Model_KPI():

    def __init__(self):
        print('init')



    def compute_AUC(model,X_test,Y_test):
        yroc = Y_test.as_matrix().reshape(Y_test.as_matrix().size,1)
        predroc = model.predict_proba(X_test)[:,1]
        AUC = metrics.roc_auc_score(yroc, predroc)
        return AUC


    def compute(self,model,X_test,Y_test):

        model_name = str(str(i) +"-"+ model.__class__.__name__)

        threshold_list = np.arange(0.0, 1.0, 0.005).tolist()
        accuracy_list = []
        PPV_list = []
        NPV_list = []
        TNR_list = []
        TPR_list = []

        pred_proba_df = pd.DataFrame(model.predict_proba(X_test))

        point_equilibrium = 0
        delta_tpr_tnr_equilibrium = 1


        for i in threshold_list:
                ## confusion matrix computation
                Y_test_pred = pred_proba_df.applymap(lambda x: 1 if x > i else 0)
                test_accuracy = metrics.accuracy_score(Y_test.as_matrix().reshape(Y_test.as_matrix().size,1),
                                                       Y_test_pred.iloc[:,1].as_matrix().reshape(Y_test_pred.iloc[:,1].as_matrix().size,1))

                confusion = metrics.confusion_matrix(Y_test.as_matrix().reshape(Y_test.as_matrix().size,1),
                                       Y_test_pred.iloc[:,1].as_matrix().reshape(Y_test_pred.iloc[:,1].as_matrix().size,1))

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
        self.AUC = compute_AUC(model,X_test,Y_test)
