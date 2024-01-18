import pandas as pd
import glob
from tabulate import tabulate
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

def PreProcessing():
    # read the data into a DataFrame
    directory = './training_data/order_data/'

    file_list = glob.glob(directory+'*')

    df_list = []

    # This Piece of Code Reads the Data from the file_list and appends into a CSV
    for file in file_list:
        df = pd.read_csv(file, delimiter="\t",names=["order_id","driver_id","passenger_id","start_region_hash","dest_region_hash","Price","Time"])
        df_list.append(df)

    merged_df = pd.concat(df_list)
    ORDERS=merged_df
    #cleaning the data by cleaining the time column
    ORDERS["Time"]=ORDERS["Time"].apply(lambda x: str(x)[:16])


    directory = './training_data/weather_data/'

    file_list = glob.glob(directory+'*')

    df_list = []

    for file in file_list:
        # read the file into a DataFrame and append it to the list
        df = pd.read_csv(file, delimiter="\t",names=["Time","Weather","Temperature","PM2.5"])
        df_list.append(df)

    merged_df = pd.concat(df_list)
    WEATHER=merged_df
    #cleaning the data by cleaining the time column
    WEATHER["Time"]=WEATHER["Time"].apply(lambda x: str(x)[:16])

    #combining the two dataframes ORDERS and WEATHER on the basis of time
    ORDER_WEATHER_DF =  pd.merge(ORDERS,WEATHER, on="Time",how="left")
    #dropping the duplicate rows
    ORDER_WEATHER_DF = ORDER_WEATHER_DF.drop_duplicates()

    directory = './training_data/cluster_map/'

    file_list = glob.glob(directory+'*')

    df_list = []

    for file in file_list:
        # read the file into a DataFrame and append it to the list
        df = pd.read_csv(file, delimiter="\t",names=["region hash","region id"])
        df_list.append(df)

    merged_df = pd.concat(df_list)
    REGIONS=merged_df

    REGIONSMapped = {}
    for i,n in enumerate(REGIONS["region hash"]):
        REGIONSMapped[n] = REGIONS["region id"][i] 

    ORDER_WEATHER_DF['start_region_hash'] = ORDER_WEATHER_DF['start_region_hash'].map(REGIONSMapped)
    ORDER_WEATHER_DF['dest_region_hash'] = ORDER_WEATHER_DF['dest_region_hash'].map(REGIONSMapped)

    #filling the missing values with 0
    ORDER_WEATHER_DF.fillna(0,inplace=True)

    ORDERS["start_region_hash"]=ORDERS["start_region_hash"].map(REGIONSMapped)

    #creating a dataframe with the supply and demand droping the null values
    SUPPLY_DATAFRAME = ORDERS[["driver_id","start_region_hash","Time","dest_region_hash"]].dropna()
    supply = SUPPLY_DATAFRAME.groupby(["Time","start_region_hash"]).size().reset_index(name='Supply')
    DEMAND_DATAFRAME = ORDERS[["passenger_id","start_region_hash","Time","dest_region_hash"]].dropna()
    demand = DEMAND_DATAFRAME.groupby(["Time","start_region_hash"]).size().reset_index(name='Demand')

    #merging the supply and demand dataframes
    SUPPLY_DEMAND = pd.merge(demand,supply,on=["Time","start_region_hash"],how="inner")
    SUPPLY_DEMAND = pd.merge(SUPPLY_DEMAND,ORDER_WEATHER_DF[["start_region_hash","Time","Weather","Temperature","PM2.5"]],on=["Time","start_region_hash"], how = "left").drop_duplicates().reset_index(drop=True)
        
    SUPPLY_DEMAND["year"]=SUPPLY_DEMAND["Time"].apply(lambda x:int(x[:4]))
    SUPPLY_DEMAND["month"]=SUPPLY_DEMAND["Time"].apply(lambda x:int(x[5:7]))
    SUPPLY_DEMAND["day"]=SUPPLY_DEMAND["Time"].apply(lambda x:int(x[8:10]))
    SUPPLY_DEMAND["hour"]=SUPPLY_DEMAND["Time"].apply(lambda x:int(x[11:13]))
    SUPPLY_DEMAND["min"]=SUPPLY_DEMAND["Time"].apply(lambda x:int(x[14:16]))
    SUPPLY_DEMAND = SUPPLY_DEMAND.drop("Time",axis=1)
    SUPPLY_DEMAND["Gap"] = SUPPLY_DEMAND["Demand"]-SUPPLY_DEMAND["Supply"]
    GAP_DF = SUPPLY_DEMAND[["year","month","day","hour","min","start_region_hash","Weather","Temperature","PM2.5","Gap"]]
    
    # print(final)
    # print(tabulate(final, headers=["year","month","day","hour","min","start_region_hash","Weather","Temperature","PM2.5","Gap"))
    print(tabulate(GAP_DF.head(), headers='keys', tablefmt='psql'))
    return GAP_DF
        
        
PreprocessedData=PreProcessing()
    
      
def RegressionModels(preprocessedData):
       # apply regression models to the data and predict the gap
        # Split the data into training and testing sets
        df=preprocessedData
        X_train, X_test, y_train, y_test = train_test_split(df.drop('Gap', axis=1), df['Gap'], test_size=0.2, random_state=42)

        # Fit and evaluate a linear regression model the value of score varies from 0 to 1 and the closer it is to 1 the better the model is
        linearRegression = LinearRegression()
        linearRegression.fit(X_train, y_train)
        lr_prediction = linearRegression.predict(X_test)
        lr_MAE = mean_absolute_error(y_test, lr_prediction)
        lr_Score = r2_score(y_test, lr_prediction)
        print('Linear Regression - Mean Absolute Error:', lr_MAE, 'Score:', lr_Score)

        #plot the data using matlibplot
        #actual and predicted values
        plt.scatter(y_test, lr_prediction)
        plt.xlabel('Actual Gap')
        plt.ylabel('Predicted Gap')
        plt.title('Actual vs. Predicted Gap')
        plt.show()
        
        # Fit and evaluate a random forest regression model the value of score varies from 0 to 1 and the closer it is to 1 the better the model is
        randomForest = RandomForestRegressor()
        randomForest.fit(X_train, y_train)
        rf_prediction = randomForest.predict(X_test)
        rf_MAE = mean_absolute_error(y_test, rf_prediction)
        rf_Score = r2_score(y_test, rf_prediction)
        print('Random Forest Regression - Mean Absolute Error:', rf_MAE, 'Score:', rf_Score)

        plt.scatter(y_test, rf_prediction)
        plt.xlabel('Actual Gap')
        plt.ylabel('Predicted Gap')
        plt.title('Actual vs. Predicted Gap')
        plt.show()
  
    

RegressionModels(PreprocessedData)

