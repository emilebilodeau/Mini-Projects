from flask import Flask, jsonify, request
from flask_restful import Resource, Api, reqparse
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
api = Api(app)



class RawFeats:
    def __init__(self, feats):
        self.feats = feats
    
    def fit(self, X, y=None):
        pass
    
    def transform(self, X, y=None):
        return X[self.feats]
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
    
class ToDenseTransformer():      
    def transform(self, X, y=None, **fit_params):
        return X.todense()
    
    def fit(self, X, y=None, **fit_params):
        return self
        
    
class FeatEngineer:
    def __init__(self, feats):
        self.feats = feats
        
    def fit(self, X, y=None):
        pass
    
    def transform(self, X, y=None):
        temp_df = pd.DataFrame(X)
        num_feats = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
        temp_df.columns = num_feats
        temp_df['total_income'] = temp_df['ApplicantIncome'] + temp_df['CoapplicantIncome']
        temp_df['loanAmountTerm_ratio'] = temp_df['LoanAmount'] / temp_df['Loan_Amount_Term']
        temp_df['loanAmountIncome_ratio'] = temp_df['LoanAmount'] / temp_df['total_income']
        return np.array(temp_df)
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

model = pickle.load(open('projectIV_model.sav', 'rb'))

class Scoring(Resource):
    def post(self):
        json_data = request.get_json()
        df = pd.DataFrame(json_data.values(), index=json_data.keys()).transpose()
        res = model.predict_proba(df)
        return res.tolist()
    
api.add_resource(Scoring, '/scoring')

if __name__=='__main__':
    app.run(debug=True)