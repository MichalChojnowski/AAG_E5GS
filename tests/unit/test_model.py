import pytest
import pandas as pd
from model import Model

# Make a new Model object without running __init__
test_model = object.__new__(Model)

data_file_location = "/opt/app/data/housing.csv"

@pytest.fixture
def simple_dataframe():
    data = [{
        "total_rooms": 5823,
        "total_bedrooms": 2201,
        "population": 3155,
        "households": 100,
        "median_house_value": 400910,
        "housing_median_age": 22.4,
        "ocean_proximity": "<1H"
    },{
        "total_rooms": 7823,
        "total_bedrooms": 4402,
        "population": 2781,
        "households": 150,
        "median_house_value": 750500,
        "housing_median_age": 5.0,
        "ocean_proximity": "NEAR BAY"
    },
    {
        "total_rooms": 10228,
        # Purposefully missing total_bedrooms
        "population": 6394,
        "households": 320,
        "median_house_value": 320000,
        "housing_median_age": 6.7,
        "ocean_proximity": "NEAR BAY"
    }]
    return pd.DataFrame(data, columns=['total_rooms','total_bedrooms','population','households',
                                        'median_house_value','housing_median_age','ocean_proximity'])

@pytest.fixture
def small_real_dataframe():
    dataframe = pd.read_csv(data_file_location)
    return dataframe.head(500)  # 500 row sample of the real dataset



@pytest.mark.summarize_tests
def test_summarize_data_simple(simple_dataframe):
    summary = test_model.summarize_data(simple_dataframe)
    assert summary["young"] == [535250.0] # should be average of 750500 and 320000
    assert summary["medium"] == [400910.0] # should be only element 400910
    assert summary["old"] == [] # no element are old (age > 30) in the simple_dataframe


@pytest.mark.summarize_tests
def test_summarize_data_real(small_real_dataframe):
    summary = test_model.summarize_data(small_real_dataframe)
    assert summary["young"] == [127500.0]
    assert summary["medium"] == [211282.05]
    assert summary["old"] == [189147.71]


@pytest.mark.summarize_tests
@pytest.mark.parametrize("drop_column", ['housing_median_age', 'ocean_proximity'])
def test_summarize_data_missing_column(small_real_dataframe, drop_column):
    missing_column_df1 = small_real_dataframe.drop(drop_column, axis=1)
    summary = test_model.summarize_data(missing_column_df1)
    assert summary["young"] == []
    assert summary["medium"] == []
    assert summary["old"] == []


@pytest.mark.prepare_tests
def test_data_load_pipeline():
    # When given a file which doesn't exist, load should throw exception
    try:
        test_model.load_data("./missing_data_file.csv")
        assert False
    except FileNotFoundError:
        assert True
    # When given a file which does exist should load without issue
    try:
        test_model.load_data(data_file_location)
        assert True
    except:
        assert False

@pytest.mark.prepare_tests
def test_prepare_pipeline(small_real_dataframe):
    # Before any modification there are 2 rows with NaN values in the dataset
    non_number_values = pd.isna(small_real_dataframe['total_bedrooms'])
    assert len(small_real_dataframe[non_number_values]) == 2

    full_df, values_df, features_df = test_model.prepare_data(small_real_dataframe)
    # Check that we have the newly computed columns
    added_columns = ["rooms_per_household", "bedrooms_per_room", "population_per_household"]
    assert set(added_columns).issubset(full_df.columns)
    assert set(added_columns).issubset(features_df.columns)
    # Check that there are the same number of rows in features_df and values_df
    assert len(features_df.index) == len(values_df.index)
    # After processing there are no NaN values for this field
    non_number_values = pd.isna(full_df['total_bedrooms'])
    assert len(full_df[non_number_values]) == 0


required_prepare_columns = ['total_bedrooms', 'total_rooms', 'households',
                            'total_bedrooms', 'population', 'median_house_value']
@pytest.mark.prepare_tests
@pytest.mark.parametrize("drop_column", required_prepare_columns)
def test_prepare_missing_column(small_real_dataframe, drop_column):
    with pytest.raises(Exception) as excinfo:
        missing_column_df = small_real_dataframe.drop(drop_column, axis=1)
        test_model.prepare_data(missing_column_df)
    assert drop_column in str(excinfo.value)  # check that the missing column is listed


@pytest.mark.fitting_tests
@pytest.mark.parametrize("algorithm", ['Linear Regression', 'Decision Tree', 'Random Forest'])
def test_fit_model(simple_dataframe, algorithm):
    # Load the real data because we need it to fit realistic models
    test_model.__init__("../data/housing.csv")
    learn_features = ['total_bedrooms', 'total_rooms', 'households', 'total_bedrooms', 'population']
    values, predictions, r2 = test_model.fit_model(algorithm, learn_features)
    assert r2 <= 1.0
    assert len(values) == len(predictions)
    assert test_model.model is not None

    # Predict a few known example values from simple dataframe to verify model is working
    full_df, simple_values, simple_features = test_model.prepare_data(simple_dataframe)
    features_encoded = pd.get_dummies(simple_features[learn_features])
    simple_predictions = test_model.model.predict(features_encoded)
    if algorithm == "Linear Regression":
        assert round(simple_predictions[0], 2) == -280814.3
    elif algorithm == "Decision Tree":
        assert round(simple_predictions[0], 2) == 216567.88
    elif algorithm == "Random Forest":
        # different seed for random forest -> slightly different predictions on different runs
        assert simple_predictions[0] - 225950.07 < 15000
