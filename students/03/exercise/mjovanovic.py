import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    import os
    import matplotlib.pyplot as plt

    # Get the absolute path to the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the absolute path to the dataset file
    data_file_path = os.path.join(script_dir, '..', 'data', '02_exam_scores.csv')


    # Load the dataset
    try:
        data = pd.read_csv(data_file_path)
    except FileNotFoundError:
        print("Error: The dataset file was not found.")
        print(f"Please make sure the file '02_exam_scores.csv' is in the '{os.path.join(script_dir, '..', 'data')}' directory.")
        exit()


    # Prepare the data
    X = data[['study_hours']]
    y = data['exam_score']

    # Create and train the model
    model = LinearRegression()
    model.fit(X, y)

    # Print the coefficients
    print(f"Coefficient: {model.coef_[0]}")
    print(f"Intercept: {model.intercept_}")

    # Predict for a new value
    study_hours_to_predict = 5
    predicted_score = model.predict(pd.DataFrame({'study_hours': [study_hours_to_predict]}))
    print(f"Predicted exam score for {study_hours_to_predict} hours of study: {predicted_score[0]}")

    # Create a scatter plot of the data
    plt.scatter(X, y, color='blue', label='Data Points')

    # Plot the regression line
    plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')

    # Add labels and title
    plt.title('Study Hours vs Exam Score')
    plt.xlabel('Study Hours')
    plt.ylabel('Exam Score')
    plt.legend()
    return


if __name__ == "__main__":
    app.run()