from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# define a Gaussain NB classifier
clf = GaussianNB()
# define a RandomForest classifier
rfc = RandomForestClassifier(criterion='gini',max_depth=5, n_estimators=10, max_features='auto')
#
dtc= SVC(kernel='poly', degree=3, max_iter=300000)
# Train until max iterations

# define the class encodings and reverse encodings
classes = {0: "Iris Setosa", 1: "Iris Versicolour", 2: "Iris Virginica",3:"invalid data"}
r_classes = {y: x for x, y in classes.items()}

# function to train and load the model during startup
def load_model():
    # load the dataset from the official sklearn datasets
    X, y = datasets.load_iris(return_X_y=True)

    # do the test-train split and train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf.fit(X_train, y_train)
    rfc.fit(X_train,y_train)
    dtc.fit(X_train,y_train)
    

    # calculate the print the accuracy score
    acc = accuracy_score(y_test, clf.predict(X_test))
    acc_rfc = accuracy_score(y_test, rfc.predict(X_test))
    acc_dtc = accuracy_score(y_test, dtc.predict(X_test))
    
    print(f"Model trained using Guassian NB with accuracy: {round(acc, 3)}")
    print(f"Model trained using Random Forest with accuracy: {round(acc_rfc, 3)}")
    #JUST TO CHECK ACCURACY
    print(f"Model trained using SVC with accuracy: {round(acc_dtc, 3)}")

# function to predict the flower using the model
def predict(query_data,classifier):
    x = list(query_data.dict().values())
    if(classifier == 'RandomForest'):
        print('predicted using randomForest')
        prediction = rfc.predict([x])[0]  
    elif(classifier == 'NaiveBayes'):
        prediction = clf.predict([x])[0]
    else:
        prediction=3
        print("incorrect classifier")
    print(f"Model prediction: {classes[prediction]}")
    return classes[prediction]

# function to retrain the model as part of the feedback loop
def retrain(data):
    # pull out the relevant X and y from the FeedbackIn object
    X = [list(d.dict().values())[:-1] for d in data]
    y = [r_classes[d.flower_class] for d in data]

    # fit the classifier again based on the new data obtained
    clf.fit(X, y)
    rfc.fit(X,y)
