import numpy as npz
import pandas as pd
import seaborn as sn
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def transformation():
    fname = r"C:\Users\HP\Desktop\MSCPS\OPER-5151EL_Business_Analytics\PROJECT\ObesityDataSet_raw_and_data_sinthetic" \
            r".csv "
    data = pd.read_csv(fname)

    # ---- Data cleaning ----

    # Check missing values
    print(data.isnull().sum())

    # Check data types
    print(data.dtypes)

    # ------Attribute Transformation
    # Gender attribute
    mapping_dict_gender = {"Male": 0, "Female": 1}
    data['Gender'] = data['Gender'].map(mapping_dict_gender)
    data['Gender'] = data['Gender'].astype('float64')

    # change some attribute to binary value 0 or 1
    mapping_dict = {"no": 0, "yes": 1}
    data['family_history_with_overweight'] = data['family_history_with_overweight'].map(mapping_dict)
    data['family_history_with_overweight'] = data['family_history_with_overweight'].astype('float64')
    data['Frequent consumption of high caloric food'] = data['Frequent consumption of high caloric food'].map(mapping_dict)
    data['Frequent consumption of high caloric food'] = data['Frequent consumption of high caloric food'].astype('float64')
    data['SMOKE'] = data['SMOKE'].map(mapping_dict)
    data['SMOKE'] = data['SMOKE'].astype('float64')
    data['Calories consumption monitoring'] = data['Calories consumption monitoring'].map(mapping_dict)
    data['Calories consumption monitoring'] = data['Calories consumption monitoring'].astype('float64')

    # change some attribute to ordinary value 0 to 3
    mapping_dict_ord = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    data['Consumption of food between meals'] = data['Consumption of food between meals'].map(mapping_dict_ord)
    data['Consumption of food between meals'] = data['Consumption of food between meals'].astype('float64')

    data['Consumption of alcohol'] = data['Consumption of alcohol'].map(mapping_dict_ord)
    data['Consumption of alcohol'] = data['Consumption of alcohol'].astype('float64')

    # ---- Create new dataframe with BMI then discard height and weight
    data_BMI = data.iloc[:, 0:-2]
    data_BMI['BMI'] = data['Weight'] / (data['Height'] / 100) ** 2
    data_BMI = data_BMI.drop('Height', axis=1)
    data_BMI = data_BMI.drop('Weight', axis=1)
    # ------ Feature selection ---
    # correlation matrix
    sn.heatmap(data.corr(), annot=True, fmt=".2f")
    plt.show()

    # Based on the correction matrix let's select few features
    new_df = data_BMI[['Age', 'family_history_with_overweight', 'Frequent consumption of high caloric food',
                       'Frequency of consumption of vegetables', 'Consumption of water daily',
                       'Consumption of alcohol', 'BMI']].copy()

    return new_df

