# Import all relevant libraries and modules
import pyspark
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, MultilayerPerceptronClassifier
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import IntegerType
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


# Create instance of pyspark library
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Read in the csv, inferring schema, telling it to use headers from the csv
df = spark.read.csv("Task3 - dataset - HIV RVG.csv",inferSchema=True,header=True)

# Section 3.1
def Task1(df):
    # Take the columns of the data set and remove the last one
    columnsList = df.columns
    columnsList.pop(-1)

    # Create dataframe for values
    dfSummary = pd.DataFrame({'Summary': ["Minimum","Maximum","Mean","Median","Mode","Standard Deviation"]})

    # For each column in the data set
    for c in columnsList:
        # Select normal/abnormal column
        dfCalculations = df.select(c).summary().toPandas()

        # Get minimum, maximum, mean, median, mode, and standard deviation values from the summary table
        # Convert them all to floats and round to 4 decimal places
        minValue = round(float(dfCalculations.iloc[3][c]),4)
        maxValue = round(float(dfCalculations.iloc[7][c]),4)
        meanValue = round(float(dfCalculations.iloc[1][c]),4)
        medianValue = round(float(dfCalculations.iloc[5][c]),4)
        modeValue = round(float(df.groupby(c).count().orderBy("count", ascending=False).first()[0]),4)
        stdValue = round(float(dfCalculations.iloc[5][c]),4)

        # Make new dataframe to join onto the previous one
        dfAppend = pd.DataFrame({c: [minValue,maxValue,meanValue,medianValue,modeValue,stdValue]})
        dfSummary = pd.concat([dfSummary,dfAppend], axis=1)

    # Create table to add to matplotlib window
    cell_text = []
    for row in range(len(dfSummary)):
        cell_text.append(dfSummary.iloc[row])

    plt.figure(figsize=(15,3))
    plt.table(cellText=cell_text, colLabels=dfSummary.columns, loc='center')
    plt.axis('off')
        
    # Create box plot
    plt.figure()
    sn.boxplot(x="Participant Condition", y="Alpha", data=df.toPandas())
        
    # Create density plot
    sn.displot(data=df.toPandas(), x="Beta", col="Participant Condition")

    # Show the matplotlib windows
    plt.show()

# Section 3.2
# Train-Test split
def TTS(df):
    # Replace Patient with 0 and Control with 1
    df = df.replace("Patient","0").replace("Control","1")
    #df = df.withColumn("Participant Condition",df("Participant Condition").cast('int'))
    df = df.withColumn("Participant Condition", df["Participant Condition"].cast(IntegerType()))

    # Randomly shuffle the dataset
    # Split the dataset 90% train, 10% test
    (trainingData, testData) = df.randomSplit([0.9, 0.1])

    """ # Print the number of examples
    print("Patient = 0, Control = 1")
    print("Training data")
    trainingData.groupBy("Participant Condition").count().show()
    print("Test data")
    testData.groupBy("Participant Condition").count().show() """

    return trainingData, testData

def ANN(trainingData, testData, df, epochs):
    # Code adapted from https://spark.apache.org/docs/latest/ml-classification-regression.html#multilayer-perceptron-classifier
    # Take the columns of the data set and remove the last one
    columnsList = df.columns
    columnsList.pop(-1)

    # Index labels, adding metadata to the label column
    # Fit on whole dataset to include all labels in index
    labelIndexer = StringIndexer(inputCol="Participant Condition", outputCol="indexedCondition")
    # Automatically identify categorical features, and index them
    featureAssembler = VectorAssembler(inputCols=columnsList, outputCol="indexedFeatures")

    # specify layers for the neural network:
    # input layer of size 12 (features), two intermediate of size 5 and 4
    # and output of size 1 (classes)
    layers = [8, 5, 4, 2]

    # create the trainer and set its parameters
    trainer = MultilayerPerceptronClassifier(labelCol="indexedCondition", featuresCol="indexedFeatures", maxIter=epochs, layers=layers, blockSize=128, seed=1234)

    # Chain indexers and SVM in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureAssembler, trainer])

    # Train model.  This also runs the indexers
    model = pipeline.fit(trainingData)

    # Make predictions based on the test data
    predictions = model.transform(testData)

    # obtain evaluator
    evaluator = MulticlassClassificationEvaluator(labelCol="indexedCondition", predictionCol="prediction", metricName="accuracy")

    return predictions, evaluator, model

def RandomForest(trainingData, testData, df, trees, leaf):
    # Code adapted from https://spark.apache.org/docs/latest/ml-classification-regression.html#random-forest-classifier
    # Take the columns of the data set and remove the last one
    columnsList = df.columns
    columnsList.pop(-1)

    # Index labels, adding metadata to the label column
    # Fit on whole dataset to include all labels in index
    labelIndexer = StringIndexer(inputCol="Participant Condition", outputCol="indexedCondition")
    # Automatically identify categorical features, and index them
    featureAssembler = VectorAssembler(inputCols=columnsList, outputCol="indexedFeatures")

    # Train a RandomForest model using n trees
    dt = RandomForestClassifier(labelCol="indexedCondition", featuresCol="indexedFeatures", numTrees=trees, maxDepth=leaf)

    # Chain indexers and forest in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureAssembler, dt])

    # Train model.  This also runs the indexers
    model = pipeline.fit(trainingData)

    # Make predictions based on the test data
    predictions = model.transform(testData)

    evaluator = MulticlassClassificationEvaluator(labelCol="indexedCondition", predictionCol="prediction", metricName="accuracy")

    rfModel = model.stages[2]
    print(rfModel)  # summary only  

    return predictions, evaluator, model

def Task2(df):
    # Get training and test data
    trainingData, testData = TTS(df)

    # Create dataframe for values
    ANN_epochs = []
    ANN_accuracy = []

    # Run the ANN 10 times, increasing the number of iterations(epochs) by 100 each time
    for i in range(100,1100,100):
        # Train Artificial Neural Network
        predictions, evaluator, model = ANN(trainingData, testData, df, i)
        # Calculate the accuracy rate
        accuracy = evaluator.evaluate(predictions)
        # Append to lists
        ANN_epochs.append(i)
        ANN_accuracy.append(accuracy)
    
    # Plot epochs against accuracy
    plt.plot(ANN_epochs, ANN_accuracy, 'b')
    plt.xlabel('Epochs (Iterations)') 
    plt.ylabel('Accuracy')
    plt.show()

    # Train Random Forest with 1000 trees and 5/10 leaf nodes
    predictions, evaluator, model = RandomForest(trainingData, testData, df, 1000, 10)
    # Calculate the accuracy rate
    accuracy = evaluator.evaluate(predictions)
    print("Random Forest Accuracy:",accuracy)

def Task3(df):
    # Get training and test data
    trainingData, testData = TTS(df)

    # Lists for different parameters
    ANN_epochs = [50,500,1000]
    RF_trees = [50,500,10000]

    print("ANN: ")
    # For each epoch
    for n in ANN_epochs:
        # Train the model
        predictions, evaluator, model = ANN(trainingData, testData, df, n)
        print("Epochs:", n)
        # Get the cross validation score
        print(cross_val_score(evaluator, np.array(trainingData), np.array(testData), scoring='r2', cv = 10).mean())

    print("RF: ")
    # For each tree
    for n in RF_trees:
        # Train the model
        predictions, evaluator, model = RandomForest(trainingData, testData, df, n, 10)
        print("Trees:", n)
        # Get the cross validation score
        print(cross_val_score(model, np.array(trainingData), np.array(testData), scoring='r2', cv = 10).mean())


#Task1(df)
#Task2(df)
Task3(df)