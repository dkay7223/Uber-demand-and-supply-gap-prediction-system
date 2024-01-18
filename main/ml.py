import os
import glob
import pandas as pd
import seaborn as sns
import gc
import rarfile
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor


class UberAISystem:
    
    #path is the absolute path of this system. 
    
    def __init__(self, path):
        
        
        
        ##Getting Dataset
        rar_path = os.path.join(path, "AI-Project Dataset.rar")
        
        #Extraction
        with rarfile.RarFile(rar_path) as rf:
            rf.extractall()
        
        #After Extraction, the extracted zip files
        file_list = [os.path.join(path, "training_data.zip"), os.path.join(path, "test_set.zip")]
        
        
        #Unzipping the Zipped files (training data and test_set)
        for file in file_list:
            with ZipFile(file, 'r') as zip_ref:
                zip_ref.extractall()
                
                

        
        ##Reading Order DATA
        order_data_path = os.path.join(path, "training_data", "order_data")
        csv_files = glob.glob(os.path.join(order_data_path, '*'))
        
        #stores individual dataframes read from the CSV files
        data_frames = []
        
        for csv_file in csv_files:
            data_frame = pd.read_csv(csv_file, delimiter="\t",
                                     names=["order_id", "driver_id", "passenger_id", "start_region_hash",
                                            "dest_region_hash", "Price", "Time"])
            data_frames.append(data_frame)
        order_data = pd.concat(data_frames, ignore_index=True)
        order_data["Time"] = order_data["Time"].apply(lambda x: str(x)[:16])
        gc.collect()



        ##Reading weather DATA
        weather_data_path = os.path.join(path, "training_data", "weather_data")
        csv_files = glob.glob(os.path.join(weather_data_path, '*'))
        
        #stores individual dataframes read from the CSV files
        data_frames = []
        
        for csv_file in csv_files:
            data_frame = pd.read_csv(csv_file, encoding="utf-8", delimiter='\t',
                                     names=["Time", "Weather", "Temperature", "PM2.5"])
            data_frames.append(data_frame)
        weather_data = pd.concat(data_frames, ignore_index=True)
        weather_data["Time"] = weather_data["Time"].apply(lambda x: str(x)[:16])
        gc.collect()


        #Merged_data contains all the dataframes concatenated based on Time, duplicates removed
        merged_data = pd.merge(order_data, weather_data, on="Time", how="left")
        merged_data = merged_data.drop_duplicates()

        cluster_map_path = os.path.join(path, "training_data", "cluster_map", "cluster_map")
        
        
        #reading the region data containing region hash and id
        
        region_data = pd.read_csv(cluster_map_path, delimiter="\t", names=["region hash", "region id"])
        gc.collect()




        #region_mapping is the dictionary which stores key-value pairs
        region_mapping = {}
        #Region hash column assigned as key, value is region id
        #based on region table provided
        
        for index, value in enumerate(region_data["region hash"]):
            region_mapping[value] = region_data["region id"][index]

        merged_data['start_region_hash'] = merged_data['start_region_hash'].map(region_mapping)
        merged_data['dest_region_hash'] = merged_data['dest_region_hash'].map(region_mapping)

        test_data = merged_data.drop(merged_data.columns[:3], axis=1)
        sns.heatmap(test_data.corr())
        gc.collect()
        merged_data.fillna(0, inplace=True)
        order_data["start_region_hash"] = order_data["start_region_hash"].map(region_mapping)


        #creating supply and demand dfs
        #supply using driver_id
        supply_df = order_data[["driver_id", "start_region_hash", "Time", "dest_region_hash"]].dropna()
        supply = supply_df.groupby(["Time", "start_region_hash"]).size().reset_index(name='Supply')
        gc.collect()
        
        #demand using passenger_id
        demand_df = order_data[["passenger_id", "start_region_hash", "Time", "dest_region_hash"]].dropna()
        demand = demand_df.groupby(["Time", "start_region_hash"]).size().reset_index(name='Demand')
        gc.collect()


        
        final_data = pd.merge(demand, supply, on=["Time", "start_region_hash"], how="inner")
        final_data = pd.merge(final_data, merged_data[["start_region_hash", "Time", "Weather", "Temperature", "PM2.5"]],
                              on=["Time", "start_region_hash"], how="left").drop_duplicates().reset_index(drop=True)
        final_data["year"] = final_data["Time"].apply(lambda x: int(x[:4]))
        final_data["month"] = final_data["Time"].apply(lambda x: int(x[5:7]))
        final_data["day"] = final_data["Time"].apply(lambda x: int(x[8:10]))
        final_data["hour"] = final_data["Time"].apply(lambda x: int(x[11:13]))
        final_data["min"] = final_data["Time"].apply(lambda x: int(x[14:16]))
        final_data = final_data.drop("Time", axis=1)
        final_data["Gap"] = final_data["Demand"] - final_data["Supply"]
        self.final_data = final_data[
            ["year", "month", "day", "hour", "min", "start_region_hash", "Weather", "Temperature", "PM2.5", "Gap"]]



    def train_models(self):
        scores = []
        model_names = []
        trained_models = []
        gc.collect()
        
        X_train, X_test, y_train, y_test = train_test_split(self.final_data.drop("Gap", axis=1), self.final_data["Gap"],
                                                            test_size=0.1, random_state=52)

        model_list = [
            XGBRegressor(),
            RandomForestRegressor(),
            GradientBoostingRegressor(),
            AdaBoostRegressor(),
            BaggingRegressor(),
            LinearRegression(),
            MLPRegressor()
        ]

        model_labels = [
            "XGB Regressor",
            "Random Forest Regressor",
            "Gradient Boosting Regressor",
            "AdaBoost Regressor",
            "Bagging Regressor",
            "Linear Regression",
            "MLP Regressor"
        ]

        for idx, reg in enumerate(model_list):
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)
            print(f"{model_labels[idx]} MAE : ", mean_absolute_error(y_pred, y_test))
            scores.append(mean_absolute_error(y_pred, y_test))
            model_names.append(model_labels[idx])
            trained_models.append(reg)
            gc.collect()

        self.models = trained_models
        self.score = scores
        self.names = model_names

    def make_predictions(self, input_data):
        for index, model in enumerate(self.models):
            print("Prediction : ", self.names[index])
            predictions = model.predict(input_data)
            yield (predictions)

if __name__ == "__main__":
    path = "./"
    uber_ai_system = UberAISystem(path)
    uber_ai_system.train_models()

    # Input some data for prediction
    input_data = ...
    predictions = list(uber_ai_system.make_predictions(input_data))