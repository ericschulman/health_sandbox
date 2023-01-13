import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import random

from xgboost import plot_importance
from xgboost import XGBRegressor
import statsmodels.api as sm

from sklearn.model_selection import train_test_split



##################################
#### misc data utilities#####
##############################

def clean_x_data(df,exclude_cols):
    
    x_cols = []
    x_cols_float = []
    for col in df.columns:
        right_type = (df[col].dtype == int) or (df[col].dtype == float)
        if right_type and not (col in exclude_cols): 
            col_var = df[col].var() 
            if col_var > 0:
                x_cols.append(col)
                if df[col].max() > 10:
                    x_cols_float.append(col)
                
    df_no_nas = df.copy()
    df_no_nas = df_no_nas.replace([np.inf, -np.inf], np.nan)
    df_no_nas = df_no_nas.replace(np.nan,0)
    X = df_no_nas
    X[x_cols_float] = X[x_cols_float].copy()/X[x_cols_float].copy().var()
    return X[x_cols].astype(float)

def train_test_pfold(X,y, num_trials = 5):
    training_test = []
    for i in range(num_trials):
        test_size = 0.33
        np.random.seed()
        X_train, X_test, y_train, y_test = train_test_split(sm.add_constant(X), y, test_size=test_size)
        training_test.append( (X_train, X_test, y_train, y_test) )
    return training_test

##########################################
####### bid data setup ##############
##########################################
def read_bid_data(year):
    name = '../data/BPT%s_data/ma_2.txt'%(year)
    data_merge = pd.read_csv(name, on_bad_lines='skip',encoding = "ISO-8859-1",delimiter='\t')
    for i in [2,6]:#[1,2,3,5,6]:
        #5 is benchmk... to much about actual stuff...
        name = '../data/BPT%s_data/ma_%s.txt'%(year,i)
        data = pd.read_csv(name, on_bad_lines='skip',encoding = "ISO-8859-1",delimiter='\t')
        data_merge = data_merge.merge(data)
    return data_merge




def clean_bid_data(df, 
    keys = ['bid_id','contract_year','allow_p0013','allow_q0013','version'],
    y_cols = ['allow_o0013']):
    
    X = clean_x_data(df,keys+y_cols)
    y = df[y_cols]#[~na_rows]
    
    #rename the columns:
    data_dict =pd.read_excel('../data/BPT2018_data/BPT2018_dictionary.xlsx',)
    data_dict = data_dict.rename(columns={'NAME':'features'})
    data_dict['features']

    rename_names = {}
    for col in list(X.columns):
        rename_names[col] = col + '//' + list(data_dict[data_dict['features'] == col]['FIELD_TITLE'])[0]

    X = X.rename(columns = rename_names)
    return y,X


##########################################
####### enroll data setup ##############
##########################################

def read_enroll_files(year,month):
    name = '../data/Monthly_Report_By_Plan_%s_%s/Monthly_Report_By_Plan_%s_%s.csv'%(year,month,year,month)
    file = open(name,'r',encoding = "ISO-8859-1")
    lines = file.readlines()
    
    count = 0
    end_loop = 0
    
    #iterate until we find the header
    while end_loop == 0:    
        if 'Contract' in lines[count] and 'Enrollment' in  lines[count]:
            end_loop = 1
        count += 1
    file.close()
    data = pd.read_csv(name,skiprows=count-1, on_bad_lines='skip',encoding = "ISO-8859-1")
    return data


def merge_enrollment_data(year):
    year_file = read_enroll_files(year,'01')
    for month in range(2,13):
        #pad month name
        month_name = str(month).zfill(2)
        month_file = read_enroll_files(year,month_name)
        year_file = pd.concat([year_file,month_file])
    
    #group by plan/bid
    year_file['Year'] = year
    year_file = year_file [~year_file['Plan ID'].isna()]
    year_file = year_file [year_file['Enrollment']!='*']
    year_file['Enrollment'] = year_file['Enrollment'].astype('int')
    year_file['Plan ID'] = year_file['Plan ID'].apply(lambda x : str(int(x)).zfill(3) )
    year_file_group = year_file.groupby(
        ['Plan ID','Contract Number'],as_index=False).agg({'Enrollment':'sum','Organization Type':'first',
                                                          'Plan Type':'first','Offers Part D':'first',
                                                           'Organization Name':'first',
                                         'Organization Marketing Name':'first','Plan Name':'first',
                                         'Parent Organization':'first'})
    
    return year_file_group

#########################################################
############### ml stuff #########################3
#######################################################


def get_predictions(model,X_test):
    y_pred = model.predict(X_test)
    y_pred[y_pred < 0] = 0
    return np.array(y_pred).reshape(X_test.shape[0],1)


def get_mse(y_pred,y_test):
    y_test = np.array(y_test).reshape(y_pred.shape)
    #ever enrolled count is column 0... want to weight mse by mkt size
    return (y_pred - y_test)**2


def plot_prediction(split,y_pred,y_test):
    #TODO what is this for?
    #setup the data
    y_pred = y_pred.flatten()
 
    plt.hist(y_pred[y_pred < split],label='predictions',alpha=.5,density=True)
    plt.hist(y_test[y_pred < split],label='true',alpha=.5,density=True)
    plt.legend()
    plt.show()

    plt.hist(y_pred[y_pred > split],label='predictions',alpha=.5,density=True)
    plt.hist(y_test[y_pred > split],label='true',alpha=.5,density=True)
    plt.legend()
    plt.show()

    
def plot_importance(X_test,model,v=False):
    #use similar code for producing rankings of features according to LASSO
    cols = np.array(X_test.columns)
    importance_raw = model.get_booster().get_score(importance_type='weight')
    importance = []

    for key in importance_raw.keys():
        importance.append([key,importance_raw[key]])

    importance  = pd.DataFrame( importance, columns=['features','score'])
    importance = importance.sort_values('score',ascending=False)

    plt.barh(importance.head(20)['features'].iloc[::-1], importance.head(20)['score'].iloc[::-1])
    plt.show()
    if v:
        print(importance['features'].head(20))


######### running the models ####################
def run_lasso(a,training_test):
    mses = []
    r2s = []
    num_trials = len(training_test)
    for j in range(num_trials):
        X_train, X_test, y_train, y_test = training_test[j]
        lasso = sm.OLS(1000*y_train, X_train).fit_regularized(method='elastic_net', alpha=a, L1_wt=1.0)
        y_pred = get_predictions(lasso,X_test/1000)
        mse = float( get_mse(y_pred,y_test).mean() )
        r2 = float( 1 - mse/y_test.var() ) 
        #print( 1 - r2 )
        mses.append(mse)
        r2s.append(r2)
        
        if j == num_trials -1:
            #plot the difference between true and predicted
            split = 5000

            #also plot important feature
            param_df = pd.DataFrame(np.abs(lasso.params/1000),
                                    columns=['score']).sort_values(by=['score'], ascending=False).head(20)
            param_df['features'] = param_df.index.astype('str')
            plt.barh(param_df['features'].iloc[::-1], param_df['score'].iloc[::-1])
            plt.show()
            print(param_df['score'].iloc[::-1])
        
    print('alpha_value:', a)
    print( np.array(mses).mean() , np.array(r2s).mean()  )
    print( np.median(mses) , np.median(r2s)  )
    print('==========================================')
    print('==========================================')



def run_tree(a,training_test):
    mses = []
    r2s = []
    param = a
    num_trials = len(training_test)
    for i in range(num_trials):
        X_train, X_test, y_train, y_test = training_test[i]
        model = XGBRegressor(n_estimators=X_train.shape[1], max_depth=param[0],
                             eta=param[1], subsample=param[2], colsample_bytree=param[3])
        model.fit(X_train, y_train)

        y_pred = get_predictions(model,X_test)

        mse = float( get_mse(y_pred,y_test).mean() )
        r2 = float( 1 - mse/y_test.var() ) 
        
        mses.append(mse)
        r2s.append(r2)
        
        if i == num_trials -1:
            plot_importance(X_test,model,v=True)
            
    print( np.array(mses).mean() , np.array(r2s).mean()  )
    print( np.median(mses) , np.median(r2s)  )