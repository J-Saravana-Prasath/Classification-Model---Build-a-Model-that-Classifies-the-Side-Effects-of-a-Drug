import numpy as np
from collections import Counter
import seaborn as sb
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# Creating of dataset
data = pd.read_csv("Final.csv", low_memory=False)
lendata = len(data)
sex = []
age = []
PatientID = []
rate = []
choose = ['male', 'female']
for i in range(len(data)):
    rand = random.choice(choose)
    ag = random.randint(15, 80)
    Id = random.randint(1000, 2000)
    ra = random.randint(7, 10)
    sex.append(rand)
    age.append(ag)
    PatientID.append(Id)
    rate.append(ra)
data.insert(loc=0, column='Patient ID', value=PatientID)
data.insert(loc=1, column='Gender', value=sex)
data.insert(loc=2, column='Age', value=age)
data.insert(loc=4, column='Ratings', value=rate)
# Preprocessing
# Missing values Removed dataset
for column in data.columns:
    test1 = data[column]
    for element in range(len(data)):
        if str(test1[element]).isspace():
            test1[element] = np.nan
    data[column] = test1
data.fillna(np.nan, inplace=True)
data = data.dropna(axis=0, how="any")
data.to_csv("new_drugs.csv", index=False)
tl = len(data)
data = pd.read_csv("new_drugs.csv")
print("Old Data frame:", lendata, "New data frame:", len(data))
# Removing Duplicates and Storing in the list
DrugName = data["urlDrugName"]
DrugName = DrugName.to_numpy()
DupremDrug = []
for i in DrugName:
    if i not in DupremDrug:
        lo = i.lower()
        DupremDrug.append(lo)
# sorted drug list
DupremDrug.sort()
print("Total Number of Drugs Available: ", len(DupremDrug))
print("The Drugs are: ")
for i in range(len(DupremDrug)):
    print("(", 1+i, ")", DupremDrug[i])
countDrug = data["urlDrugName"].value_counts()
InDrug = str(input("Enter the Drug name to be search:")).lower()
if InDrug in DupremDrug:
    print("Details of the", InDrug, "Drug is available")
    new_data = []
    for i in range(0, len(DrugName)):
        if InDrug in DrugName[i] and len(InDrug) == len(DrugName[i]):
            pos = i
            new_data.append(data.iloc[pos])
    new_data = pd.DataFrame(new_data).sort_values(by='Age')
    new_data.to_csv("new_drug1.csv", index=False)
    new_data = pd.read_csv('new_drug1.csv')
    age = new_data['Age']
    i = 0
    for i in range(0, len(age)):
        if age[i] < 15:
            age[i] = 'below 15'
        elif (age[i] >= 15) and (age[i] <= 20):
            age[i] = "15-20"
        elif (age[i] >= 21) and (age[i] <= 30):
            age[i] = "21-30"
        elif (age[i] >= 31) and (age[i] <= 40):
            age[i] = "31-40"
        elif (age[i] >= 41) and age[i] <= 50:
            age[i] = "41-50"
        elif (age[i] >= 51) and (age[i] <= 60):
            age[i] = "51-60"
        elif (age[i] >= 61) and (age[i] <= 70):
            age[i] = "61-70"
        else:
            age[i] = "71 and above"
    new_data['Age'] = age

    def classify(cs1, tit):

        ct = 0
        cs2 = []
        lt = len(cs1)
        for i1 in cs1:
            if i1 not in cs2:
                cs2.append(i1)
        class1 = {cs: ct for cs in cs2}
        for k in class1:
            ct = 0
            for j in range(0, lt):
                if k in cs1[j]:
                    if (len(k)) == len(cs1[j]):
                        ct = ct + 1
            class1[k] = ct
        plt.title(tit)
        plt.pie(class1.values(), labels=class1.keys(), radius=1, autopct='%0.2f%%', shadow=True,
                wedgeprops={'edgecolor': 'white', 'linewidth': 1})
        plt.show()
        return class1

    Effectiveness = new_data['Effectiveness'].to_numpy()
    effect = classify(Effectiveness, ("Effectiveness of the Drug " + InDrug))
    print(effect)
    SideEffects = new_data['Side Effects'].to_numpy()
    sideEffects = classify(SideEffects, ("Side Effects of the Drug " + InDrug))
    print(sideEffects)
    rating = new_data['Ratings']
    rating = rating.to_numpy()
    rate = 0
    avg = 0
    for i in range(0, len(new_data)):
        rate = rating[i] + rate
        avg = avg + 1
    print("The Overall Rating of the Drug ", InDrug, "is", format(rate / avg, '0.2f'))
    sb.countplot(y='Condition', data=new_data)
    plt.title("The Drugs Prescribed for the Condition")
    plt.show()
    data1 = new_data['Effectiveness']
    sb.countplot(x='Effectiveness', data=new_data, hue='Gender', hue_order=['male', 'female'], palette='bright6')
    plt.ylabel('Gender')
    plt.title('Effectiveness of the Drug ' + InDrug.capitalize() + ' based on Gender')
    plt.show()
    sb.countplot(x='Side Effects', data=new_data, hue='Gender', hue_order=['male', 'female'], palette='bright6')
    plt.ylabel('Gender')
    plt.title('Side effects of the Drug ' + InDrug.capitalize() + ' based on Gender')
    plt.show()
    sb.countplot(x='Age', data=new_data.head(len(new_data)), hue='Effectiveness')
    plt.ylabel('Effectiveness')
    plt.title('Effectiveness of the Drug ' + InDrug.capitalize() + ' based on Age')
    plt.show()
    sb.countplot(x='Age', data=new_data.head(len(new_data)), hue='Side Effects')
    plt.ylabel('Side Effects')
    plt.title('Side effects of the Drug ' + InDrug.capitalize() + ' based on Age')
    plt.show()

    count = Counter(new_data['urlDrugName'])
    count = count[InDrug]
    if int(count) > 150:
        data1 = new_data['Effectiveness']
        data2 = []
        for i in data1:
            if i == 'Ineffective' or i == 'Marginally Effective':
                data2.append(0)
            elif i == "Moderately Effective" or i == 'Considerably Effective' or i == 'Highly Effective:
                data2.append(1)
        new_data['Effectiveness'] = data2
        data2 = []
        for i in new_data['Gender']:
            if i == 'male':
                data2.append(0)
            else:
                data2.append(1)
        new_data['Gender'] = data2
        data2 = []
        data1 = new_data['Side Effects']
        for i in data1:
            if i == 'No Side Effects' or i == 'Mild Side Effects':
                data2.append(0)
            elif i == "Moderate Side Effects" or i == 'Severe Side Effects' or i == 'Extremely Severe Side Effects':
                data2.append(1)
        new_data['Side Effects'] = data2
        target = new_data['Effectiveness']
        cols = ['urlDrugName', 'Condition', 'Sides', 'Age']
        for x in cols:
            new_data[x] = pd.factorize(new_data[x])[0]
        new_data.to_csv("new_data.csv", index=False)
        new_data = pd.read_csv('new_data.csv')
        # SVM Classification
        scaler = StandardScaler()
        df = scaler.fit_transform(new_data)
        x_train, x_test, y_train, y_test = train_test_split(df, target, test_size=0.4, random_state=42)
        x_test_scaled = scale(x_test)
        svm_clf = SVC(random_state=42, class_weight='balanced')
        svm_clf.fit(x_train, y_train)
        svm_pred = svm_clf.predict(x_test)
        print(classification_report(y_test, svm_pred))
        cm = confusion_matrix(y_test, svm_pred, labels=svm_clf.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ineffective', 'Moderately', 'Highly'])
        disp.plot()
        plt.show()
        [TP, FP, FN, TN] = cm
        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP / (TP + FN)
        # Precision or positive predictive value
        PPV = TP / (TP + FP)
        # Overall accuracy for each class
        ACC = (TP + TN) / (TP + FP + FN + TN)
        F1 = (2*PPV*TPR)/(PPV+TPR)
        print(TPR, ACC, PPV, F1)
        # Random Forest Classifier
        target = new_data['Side Effects']
        scaler = StandardScaler()
        df = scaler.fit_transform(new_data)
        x_train, x_test, y_train, y_test = train_test_split(df, target, test_size=0.4, random_state=42)
        rf_clf = RandomForestClassifier(random_state=42, class_weight='balanced')
        rf_clf.fit(x_train, y_train)
        rf_pred = rf_clf.predict(x_test)
        print(classification_report(y_test, rf_pred))
        cm = confusion_matrix(y_test, rf_pred, labels=rf_clf.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Side Effects', 'Moderate Side Effects',
                                                                           'Severe Side Effects'])
        disp.plot()
        plt.show()
else:
    print("The details of the Drug is not available")
