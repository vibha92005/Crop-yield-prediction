import pandas as pd


def predict_crop_yield(model, columns):

    sample = pd.DataFrame([[
        2,
        25,
        700,
        6.5,
        60,
        45,
        50
    ]], columns=columns)

    prediction = model.predict(sample)

    print(f"\nPredicted Yield: {prediction[0]:.2f} tons/ha")
