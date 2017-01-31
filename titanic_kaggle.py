# titanic_panda.py is Panda tutorial from https://www.kaggle.com/c/titanic/details/getting-started-with-python-ii
# Assumes python vers. 2.7
import csv as csv
import numpy as np
import pandas as pd
import pylab as plt
from fancyimpute import MICE
import random
from sklearn.model_selection import cross_val_score
# from sklearn.ensemble import RandomForestClassifier



class TitanicPanda(object):
    def __init__(self):
        self.df = TitanicPanda.df
        self.df_test = TitanicPanda.df_test


    ''' Numpy Arrays for dataFrame'''
    def data_frame_with_numpy(self):
        csv_file_object = csv.reader(open('/home/user/Documents/Kaggle/Titanic/train.csv', 'rb'))
        header = csv_file_object.next()
        data = []

        for row in csv_file_object:
            data.append(row) # appends in row index
        data = np.array(data)
        print np.shape(data)
        print data

        print data[0:15, 5]
        print type(data[0::, 5])

        # Mean of passenger ages
        # ages_onboard = data[0::,5].astype(np.float) # makes trouble with the empty string in position 6. We need Pandas to fix this.
        # print ages_onboard

    ''' Pandas DataFrame '''
    # For .read_csv, always use header=0 when you know row 0 is the header row
    df = pd.read_csv('/home/user/Documents/Kaggle/Titanic/train.csv', header=0)
    df_test = pd.read_csv('/home/user/Documents/Kaggle/Titanic/test.csv', header=0)


    ''' Data Munging '''
    # print df['Age'].mean()
    # print df['Age'].median()
    # print df[df['Age'] > 60][['Sex', 'Pclass', 'Age', 'Survived']]
    # print df[df['Age'].isnull()][['Sex', 'Pclass', 'Age']]
    # for i in range(1, 4):
    #     print i, len(df[(df['Sex'] == 'male') & (df['Pclass'] == i)])

    # df['Age'].dropna().hist(bins=16, range=(0, 80), alpha=.5)
    # print type(df['Sex'])
    # print df.describe()
    # df[df['Sex'] == 'male'].dropna().hist(bins=16, alpha=.5)
    # plt.show()

    ''' Cleaning the Data '''
    # df['Gender'] = df['Sex'].map(lambda x: x[0].upper())
    def clean_data(self, df):
        df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)  # male=1, female=0
        # self.estimate_by_gender_pclass(df, df.Fare, 'Fare', 'FareFill', 'FareIsNull')
        # self.embarked_estimate(df, df.Embarked, 'Embarked', 'EmbarkedFill', 'EmbarkedIsNull')
        # estimate_by_age_pclass(df, df.Fare, 'Fare', 'FareFill', 'FareIsNull')

    def embarked_estimate(self, df, df_estimated_var, estimated_var, estimated_var_fill, estimated_var_is_null):
        # remove null row in new clone
        df_clone = self.drop_null_rows_Embarked(df)

        # make the statistics using clone
        # Let  0, 1 or 2 (Cherbourg, Southamption and Queenstown) OR (C, S, Q)
        embark_estimate_on_single_max_val = 0
        if embark_estimate_on_single_max_val:
            df_clone[estimated_var_fill] = df_clone[estimated_var].map({'C': 0, 'S': 1, 'Q': 2}).astype(int)
            hist_ar = np.histogram(df_clone[estimated_var_fill].values, bins=[0, 1, 2, 3])
            # get index of max in histogram
            max_val_index = np.argmax(hist_ar[0])
            # Switch to map index value to correct string {C, S, Q}
            condition_list = [max_val_index == 0, max_val_index == 1, max_val_index == 2]
            choice_list = ['C', 'S', 'Q']
            max_val_index = np.select(condition_list, choice_list)
            # print 'max_val_index-----------------------------------------------------------:'
            # print max_val_index
            df[estimated_var_fill] = df[estimated_var]
            df.loc[df_estimated_var.isnull(), estimated_var_fill] = max_val_index
            df[estimated_var_fill] = df[estimated_var_fill].map({'C': 0, 'S': 1, 'Q': 2}).astype(int)
        else:
            # Embarkment based on fare price
            df[estimated_var_fill] = df[estimated_var]
            df.loc[(df_estimated_var.isnull()) & (df.FareFill > 80.0), estimated_var_fill] = 'C'
            df.loc[(df_estimated_var.isnull()) & (df.FareFill <= 80.0), estimated_var_fill] = 'S'
            df[estimated_var_fill] = df[estimated_var_fill].map({'C': 0, 'S': 1, 'Q': 2}).astype(int)

        df[estimated_var_is_null] = pd.isnull(df_estimated_var).astype(int)

    def estimate_by_gender_pclass(self, df, df_estimated_var, estimated_var, estimated_var_fill, estimated_var_is_null):
        median_var = np.zeros((2, 3))
        for i in range(0, 2):
            for j in range(0, 3):
                isMedian = 1
                if isMedian:
                    median_var[i, j] = df[(df['Gender'] == i) & (df['Pclass'] == j + 1)][estimated_var].dropna().median()
                else:
                    # Uses mean value
                    median_var[i, j] = df[(df['Gender'] == i) & (df['Pclass'] == j + 1)][estimated_var].dropna().mean()
        df[estimated_var_fill] = df[estimated_var]

        for i in range(0, 2):
            for j in range(0, 3):
                df.loc[(df_estimated_var.isnull()) & (df.Gender == i) & (df.Pclass == j + 1), estimated_var_fill] = median_var[i, j]

        df[estimated_var_is_null] = pd.isnull(df_estimated_var).astype(int)

    def estimate_by_gender_pclass_title(self, df, df_estimated_var, estimated_var, estimated_var_fill, estimated_var_is_null):
        median_var = np.zeros((2, 3, 5))
        for i in range(0, 2):
            for j in range(0, 3):
                for l in range(0, 5):
                    isMedian = 1
                    if isMedian:
                        median_var[i, j, l] = df[(df['Gender'] == i) & (df['Pclass'] == j + 1) & (df['Title'] == l)][estimated_var].dropna().median()
                    else:
                        # Uses mean value
                        median_var[i, j, l] = df[(df['Gender'] == i) & (df['Pclass'] == j + 1) & (df['Title'] == l)][estimated_var].dropna().mean()
        df[estimated_var_fill] = df[estimated_var]

        for i in range(0, 2):
            for j in range(0, 3):
                for l in range(0, 5):
                    df.loc[(df_estimated_var.isnull()) & (df.Gender == i) & (df.Pclass == j + 1) & (df['Title'] == l), estimated_var_fill] = median_var[i, j, l]

        df[estimated_var_is_null] = pd.isnull(df_estimated_var).astype(int)


    def estimate_by_gender_pclass_title_random(self, df, df_estimated_var, estimated_var, estimated_var_fill, estimated_var_is_null):
        df[estimated_var_fill] = df[estimated_var]
        for i in range(0, 2):
            for j in range(0, 3):
                for l in range(0, 5):
                    age_distr = np.array(df[(df['Gender'] == i) & (df['Pclass'] == j + 1) & (df.Title == l)][estimated_var].dropna())
                    sumOfNan = sum(np.isnan(df[(df['Gender'] == i) & (df['Pclass'] == j + 1) & (df.Title == l)][estimated_var]))
                    if len(age_distr) > 0:
                        df.loc[(df_estimated_var.isnull()) & (df.Gender == i) & (df.Pclass == j + 1) & (df.Title == l), estimated_var_fill] = np.random.choice(age_distr, sumOfNan, replace=True)

        df[estimated_var_is_null] = pd.isnull(df_estimated_var).astype(int)

    # Todo: we get negative age estimates with mice
    def estimate_by_gender_pclass_title_mice(self, df, df_estimated_var, estimated_var, estimated_var_fill, estimated_var_is_null):
        df[estimated_var_fill] = df[estimated_var]
        random.seed(129)
        mice = MICE()  #model=RandomForestClassifier(n_estimators=100))
        if any('Survived' == df.columns.values):
            inputFeatures = ['Pclass','Gender','FamilySize','Survived', 'Parch', 'SibSp', 'Age']
            res = mice.complete(df[inputFeatures].values)
            df[estimated_var_fill] = res[:, 6]
        else:
            inputFeatures = ['Pclass', 'Gender', 'FamilySize', 'Parch', 'SibSp', 'Age']
            res = mice.complete(df[inputFeatures].values)
            df[estimated_var_fill] = res[:, 5]

        df[estimated_var_is_null] = pd.isnull(df_estimated_var).astype(int)


    def estimate_by_gender_title(self, df, df_estimated_var, estimated_var, estimated_var_fill, estimated_var_is_null):
        median_var = np.zeros((2, 5))
        for i in range(0, 2):
            for j in range(0, 5):
                isMedian = 1
                if isMedian:
                    median_var[i, j] = df[(df['Gender'] == i) & (df['Title'] == j)][estimated_var].dropna().median()
                else:
                    # Uses mean value
                    median_var[i, j] = df[(df['Gender'] == i) & (df['Title'] == j)][estimated_var].dropna().mean()
        df[estimated_var_fill] = df[estimated_var]

        for i in range(0, 2):
            for j in range(0, 5):
                df.loc[(df_estimated_var.isnull()) & (df.Gender == i) & (df.Title == j), estimated_var_fill] = median_var[i, j]

        df[estimated_var_is_null] = pd.isnull(df_estimated_var).astype(int)


    def estimate_by_age_pclass(self, df, df_estimated_var, estimated_var, estimated_var_fill, estimated_var_is_null):
        # Must depend on class and age. We are expecting a big matrix since age is a parameter
        median_var = np.zeros((len(df.AgeFill), 3))
        for i in range(0, len(df.AgeFill)):
            for j in range(0, 3):
                median_var[i, j] = df[(df['AgeFill'] == i) & (df['Pclass'] == j+1)][estimated_var].dropna().median()

        df[estimated_var_fill] = df[estimated_var]

        for i in range(0, len(df.AgeFill)):
            for j in range(0, 3):
                df.loc[(df_estimated_var.isnull()) & (df.AgeFill == i) & (df.Pclass == j+1), estimated_var_fill] = median_var[i,j]

        df[estimated_var_is_null] = pd.isnull(df_estimated_var).astype(int)



    ''' Feature Engineering '''
    def feature_engineering(self, df):
        df['FamilySize'] = df['SibSp'] + df['Parch']  # Adding siblings and parents
        isGroupsSize = 4
        if isGroupsSize == 3:
            # Group familysize into three groups (big, medium, singles) = (5:end, 2:4, 1) members (2, 1, 0)
            df.loc[df.FamilySize == 0, 'FamilySize'] = 0
            df.loc[(df.FamilySize > 0) & (df.FamilySize <= 4), 'FamilySize'] = 1
            df.loc[df.FamilySize > 4, 'FamilySize'] = 2
        elif isGroupsSize == 4:
            oneSplitsUntil3 = 0
            if oneSplitsUntil3:
                # Group familysize into three groups (big, medium, small, singles) = (3:end, 2, 1, 0) members (3, 2, 1, 0)
                df.loc[df.FamilySize == 0, 'FamilySize'] = 0
                df.loc[(df.FamilySize == 1), 'FamilySize'] = 1
                df.loc[(df.FamilySize == 2), 'FamilySize'] = 2
                df.loc[df.FamilySize >= 3, 'FamilySize'] = 3
            else:
                # Is the optimal categorization of familysize
                # Group familysize into three groups (big, medium, small, singles) = (5:end, 3:4, 1:2, 0) members (3, 2, 1, 0)
                df.loc[df.FamilySize == 0, 'FamilySize'] = 0
                df.loc[(df.FamilySize > 0) & (df.FamilySize <= 2), 'FamilySize'] = 1
                df.loc[(df.FamilySize > 2) & (df.FamilySize <= 4), 'FamilySize'] = 2
                df.loc[df.FamilySize > 4, 'FamilySize'] = 3
        else:
            pass

        # Passenger title feature.
        titleRegex = '(.*, )|(\\..*)'
        df['Title'] = df.Name
        df.Title = df.Title.str.replace(titleRegex, '').astype('str')
        rare_title = ['Dona', 'Lady', 'the Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']
        for ite in rare_title:
            df.loc[df.Title == ite, 'Title'] = 'Rare Title'

        # df = df.drop(df.index[df.Title.isnull()])
        # print df[df.Title.isnull()]

        # Hack
        # checkVars = ['Master', 'Miss', 'Mr', 'Mrs', 'Rare Title']
        # print 'Number of None:---------------------------------------------------------------------- ' # 1 in testdata and 4 in training data
        # print len(df.loc[(df.Title != 'Master') & (df.Title != 'Miss') & (df.Title != 'Mr') & (df.Title != 'Mrs') & (df.Title != 'Rare Title'), 'Title'])
        # countNones = 0 # we showed earlier that only 5 places occur
        # df.loc[(df.Title != 'Master') & (df.Title != 'Miss') & (df.Title != 'Mr') & (df.Title != 'Mrs') & (df.Title != 'Rare Title'), 'Title'] = 'Miss'
        df.loc[(df.Title == 'Mlle'), 'Title'] = 'Miss'
        df.loc[(df.Title == 'Ms'), 'Title'] = 'Miss'
        df.loc[(df.Title == 'Mme'), 'Title'] = 'Mrs'
        df['Title'] = df['Title'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Rare Title': 4}).astype(int)


        # self.estimate_by_gender_title(df, df.Age, 'Age', 'AgeFill', 'AgeIsNull')
        # self.estimate_by_gender_pclass_title(df, df.Age, 'Age', 'AgeFill', 'AgeIsNull')
        # self.estimate_by_gender_pclass_title_random(df, df.Age, 'Age', 'AgeFill', 'AgeIsNull')
        self.estimate_by_gender_pclass_title_mice(df, df.Age, 'Age', 'AgeFill', 'AgeIsNull')
        # self.estimate_by_gender_pclass(df, df.Age, 'Age', 'AgeFill', 'AgeIsNull')
        self.estimate_by_gender_pclass_title_random(df, df.Fare, 'Fare', 'FareFill', 'FareIsNull')
        # self.estimate_by_gender_pclass_title(df, df.Fare, 'Fare', 'FareFill', 'FareIsNull')
        self.embarked_estimate(df, df.Embarked, 'Embarked', 'EmbarkedFill', 'EmbarkedIsNull')


        df['Age*Class'] = df.AgeFill * df.Pclass
        df['Fare*Class'] = df.FareFill * df.Pclass
        df['Fare*Age'] = df.FareFill * df.AgeFill

        # Categorize the people in children and grown ups by gender
        # let (0,1,2) denote (woman, man, child)
        df['ChildOrAdult'] = df.Gender
        df.loc[df.AgeFill < 16, 'ChildOrAdult'] = 2  # < 16 gives the best score when uploaded to kaggle better than 17 and 18
        df.loc[df.AgeFill >= 64, 'ChildOrAdult'] = 3  # Old
        # df.loc[(df.AgeFill >= 16) & (df.AgeFill < 24), 'ChildOrAdult'] = 3
        # todo: look at the convergence for the age of a child (children must always be accompanied, check on siblings)
        # todo: could there be any correlations on gender for a child to survive
        # Answ.: gender is already in the calculation
        # todo: a feature could examine whether the passengers in cabins a upper compared to lower deck were more likely to survive.


        # family or not in family
        # implies a score of: 0.75598. Thus the feature is not good.
        # We need to drop siblings and Parch
        df['FamilyOrNot'] = df.Gender
        df.loc[df.FamilySize == 0, 'FamilyOrNot'] = 0
        df.loc[df.FamilySize != 0, 'FamilyOrNot'] = 1

        # Obs. from data: The majority of adult men died. Hence a model is not easily able to predict if a man survived and would possible estimate a man as dead
        # Consequence: The real structure in the data is shown in the women and children that survived and assuming all men dead.
        if any('Survived' == df.columns.values):
            # Killing men
            df.loc[(df.ChildOrAdult == 1) & (df.Survived == 1), 'Survived'] = 0
            # df.loc[(df.ChildOrAdult == 1) & (df.Survived == 1) & (df.FamilySize > .5), 'Survived'] = 0
            # Killing old people
            # df.loc[(df.ChildOrAdult == 3) & (df.Survived == 1), 'Survived'] = 0
            # Killing women
            # df.loc[(df.ChildOrAdult == 0) & (df.Survived == 1) & (df.FamilySize > 3.5), 'Survived'] = 0
            # df.loc[(df.ChildOrAdult == 0) & (df.Survived == 1) & (df['Age*Class'] > 74.0), 'Survived'] = 0
            # Killing children
            # df.loc[(df.ChildOrAdult == 2) & (df.Survived == 1) & (df.FamilySize > 3.5), 'Survived'] = 0
            # df.loc[(df.ChildOrAdult == 2) & (df.Survived == 1) & (df.Pclass == 3), 'Survived'] = 0
            # Killing people by class
            # df.loc[(df.Survived == 1) & (df.Pclass > 2), 'Survived'] = 0
            pass

    ''' Final preparation '''
    def drop_variable(self, df):
        # print df.dtypes[df.dtypes.map(lambda x: x=='object')]
        # Hence the columns are dropped, since they are of type object
        df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
        df = df.drop(['Age'], axis=1)
        df = df.drop(['PassengerId'], axis=1)
        df = df.drop(['Fare'], axis=1)

        df = df.drop(['Fare*Age'], axis=1)
        # df = df.drop(['Age*Class'], axis=1)
        # df = df.drop(['FamilySize'], axis=1)
        # df = df.drop(['ChildOrAdult'], axis=1)
        # df = df.drop(['Gender'], axis=1)
        # df = df.drop(['Pclass'], axis=1)
        # df = df.drop(['Title'], axis=1)
        df = df.drop(['FareIsNull'], axis=1)
        # df = df.drop(['FareFill'], axis=1)
        # df = df.drop(['AgeFill'], axis=1)
        df = df.drop(['AgeIsNull'], axis=1)
        # df = df.drop(['EmbarkedFill'], axis=1)
        df = df.drop(['EmbarkedIsNull'], axis=1)
        df = df.drop(['FamilyOrNot'], axis=1)
        # df = df.drop(['Parch'], axis=1)
        # df = df.drop(['SibSp'], axis=1)
        # df = df.drop(['SibSp', 'Parch'], axis=1)  # made model worse!
        return df

    def drop_null_rows_Embarked(self, df):
        # print df.index[df.Embarked.isnull()]
        return df.drop(df.index[df.Embarked.isnull()])

    def drop_null_rows_Fare(self, df):
        # print df.index[df.Embarked.isnull()]
        return df.drop(df.index[df.Fare.isnull()])

    def prepare_data_random_forest(self, df):
        self.clean_data(df)
        self.feature_engineering(df)
        df = self.drop_variable(df)
        self.feature_scaling(df)
        return df

    def decision_boundary(self, x, theta):
        return np.dot(theta.T, x)


    def feature_2D_plot(self, df, correlated_column_1, correlatedColumn2, h):
        xDimRed = df[[correlated_column_1, correlatedColumn2]]
        X = xDimRed.values
        y = df['Survived'].values
        # create a mesh to plot in
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        return X, y, xx, yy

    def compute_score_crossval(self, clf, X, y, scoring='accuracy'):
        # 5-fold cross validation score with accuracy metric
        crossVal = cross_val_score(clf, X, y, cv=5, scoring=scoring)
        return np.mean(crossVal)

    def feature_scaling(self, df):
        # Scales all features to be values in [0,1]
        features = list(df.columns)
        df[features] = df[features].apply(lambda x: x/x.max(), axis=0)


def main():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_selection import SelectFromModel
    from sklearn.naive_bayes import GaussianNB
    from sklearn import svm
    from collections import OrderedDict
    from sklearn.ensemble import IsolationForest
    import seaborn as sns
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import GridSearchCV

    ''' Prepare data: '''
    # Training data

    titanic_panda_inst = TitanicPanda()
    # Make copy and with new pointer ref. Avoid writing to class variable titanic_panda_inst.df, titanic_panda_inst.df_test
    df_publ = titanic_panda_inst.df[:][:]
    df_test_publ = titanic_panda_inst.df_test[:][:]

    df = titanic_panda_inst.prepare_data_random_forest(df_publ)
    print '\n TRAINING DATA:----------------------------------------------- \n'
    print df.head(3)
    print '\n'
    print df.info()
    print '\n'
    print df.describe()
    train_data = df.values
    # Test data
    passengerId_df_test = titanic_panda_inst.df_test['PassengerId']  # Submission column
    df_test = titanic_panda_inst.prepare_data_random_forest(df_test_publ)
    print '\n TEST DATA:----------------------------------------------- \n'
    print df_test.info()
    print '\n'
    print df_test.describe()
    print '\n'
    test_data = df_test.values

    # print np.any(np.isnan(test_data))
    # print np.all(np.isfinite(test_data))
    # print df_test.index[df_test.Parch.isnull()]
    # print df.index[df.Embarked.isnull()]


    ''' Explore data '''
    explore_data = 0
    if explore_data:

        print("Total Records for missing values: {}\n".format(titanic_panda_inst.df["Age"].count() + titanic_panda_inst.df_test['Age'].count()))

        print("Training set missing values")
        print(titanic_panda_inst.df.isnull().sum())
        print("\n")

        print("Testing set missing values")
        print(titanic_panda_inst.df_test.isnull().sum())


        print("\n=== AFTER IMPUTERS ===\n")
        print("=== Check for missing values in set ===")
        print("Total Records for missing values: {}\n".format(titanic_panda_inst.df["Age"].count() + titanic_panda_inst.df_test['Age'].count()))

        print("Training set missing values")
        print(df.isnull().sum())
        print("\n")

        print("Testing set missing values")
        print(df_test.isnull().sum())


        # Overview of data with histograms
        # df_test[df_test['Gender'] == 0].hist(bins=16, alpha=.5)
        # plt.show()

        # df['Age*Class'].dropna().hist(bins=16, alpha=.5)
        # df['FamilySize'].dropna().hist(bins=16, alpha=.5)
        # plt.show()

        # plt.hist(df[df.Survived == 0][['ChildOrAdult']].values, bins=[0, 1, 2, 3])
        # plt.hist(df[df.Survived == 1][['ChildOrAdult']].values, bins=[0, 1, 2, 3])
        # df[df.Survived == 0][['ChildOrAdult']].hist(bins=[0, 1, 2, 3], alpha=0.5)

        # Good overview of features histogram
        # df[df.Survived == 1].hist(bins=16, alpha=0.5)
        # df[df.Survived == 0].hist(bins=16, alpha=0.5)
        # plt.show()
        # plt.close()


        # Correlation Sex and Age. Violin plot
        sns.violinplot(x='Sex', y='Age', hue='Survived', data=titanic_panda_inst.df, split=True, cut=0, inner='stick', palette='Set1')
        plt.show()
        plt.close()

        # Correlation Gender and Age. Violin plot with mice
        sns.violinplot(x='Gender', y='AgeFill', hue='Survived', data=df, split=True, cut=0, inner='stick',
                       palette='Set1')
        plt.show()
        plt.close()

        # Class and Fare. Swarm plot.
        # sns.swarmplot(x='Pclass', y='Fare', hue='Survived', data=titanic_panda_inst.df, palette='dark')
        # plt.show()
        # plt.close()

        # Produce a heatmap
        family_data = df.pivot_table(values='Survived', index=['ChildOrAdult'], columns='FamilySize')
        htmp = sns.heatmap(family_data, annot=True, cmap='YlGn')
        plt.show()
        plt.close()

        # Good overview of features histogram FamilySize
        df['FamilySize'].hist(bins=10, alpha=0.5, range=(0, 10))
        # plt.show()
        plt.close()

        # Good overview of features histogram FamilySize
        for ite in np.arange(0, 2):
            df[df.Survived == ite]['AgeFill'].hist(bins=100, alpha=0.5)#, range=(0, 100))
        plt.legend(('Dead 0', 'Survived 1'), loc=1, borderaxespad=0.)
        plt.show()
        plt.close()


        # Good overview of features Embarkment based on fare price. There is split at Fare = 80
        for ite in np.arange(0, 3):
            df[(df.EmbarkedFill == ite)]['FareFill'].hist(bins=200, alpha=0.5, range=(0, 520))
            # The other features show no clear separation.
            # df[(df.EmbarkedFill == ite)]['Pclass'].hist(bins=[1, 2, 3, 4], alpha=0.5, range=(0, 6))
            # df[(df.EmbarkedFill == ite)]['Fare*Class'].hist(bins=200, alpha=0.5, range=(0, 600))
            # df[(df.EmbarkedFill == ite)]['FamilySize'].hist(bins=[0, 1, 2, 3, 4], alpha=0.5, range=(0, 5))
            # df[(df.EmbarkedFill == ite)]['Gender'].hist(bins=[0, 1, 2], alpha=0.5, range=(0, 1))
        plt.legend(('EmbarkedFill 0', 'EmbarkedFill 1', 'EmbarkedFill 2'), loc=2, borderaxespad=0.)
        # plt.show()
        plt.close()

        # Good overview of features Gender and Title
        for ite in np.arange(0, 2):
            # df[(df.EmbarkedFill == ite)]['FareFill'].hist(bins=200, alpha=0.5, range=(0, 520))
            # The other features show no clear separation.
            # df[(df.Gender == ite)]['Pclass'].hist(bins=[1, 2, 3, 4], alpha=0.5, range=(0, 6))
            df[(df.Gender == ite)]['Title'].hist(bins=5, alpha=0.5, range=(0, 5))
            # df[(df.EmbarkedFill == ite)]['Fare*Class'].hist(bins=200, alpha=0.5, range=(0, 600))
            # df[(df.EmbarkedFill == ite)]['FamilySize'].hist(bins=[0, 1, 2, 3, 4], alpha=0.5, range=(0, 5))
            # df[(df.EmbarkedFill == ite)]['Gender'].hist(bins=[0, 1, 2], alpha=0.5, range=(0, 1))
        plt.legend(('Female 0', 'Male 1'), loc=2, borderaxespad=0.)
        # plt.show()
        plt.close()



        # Plot correlation between two features
        # 'x' denote dead and 'o' denote survived.
        ax1 = plt.subplot(1, 1, 1)
        ChildOrAdult_value = 0
        # correlated_column_1 = 'FareFill'
        # correlated_column_2 = 'EmbarkedFill'
        correlated_column_1 = 'Gender'
        correlated_column_2 = 'Title'
        df_two_Correlation_survived = df[(df.Survived == 1) & (df.ChildOrAdult == ChildOrAdult_value)][[correlated_column_1, correlated_column_2]]
        ax1.plot(df_two_Correlation_survived.values[0::, 1], df_two_Correlation_survived.values[0::, 0], 'o')
        df_two_Correlation_dead = df[df.Survived == 0 & (df.ChildOrAdult == ChildOrAdult_value)][[correlated_column_1, correlated_column_2]]
        ax1.plot(df_two_Correlation_dead.values[0::, 1], df_two_Correlation_dead.values[0::, 0], 'x')
        plt.axis('tight')
        # plt.show()
        # plt.close()
        # Obs. The majority of adult men died. Hence a model is not easily able to predict if a man survived and would possible estimate a man as dead
        # Consequence: The real structure in the data is shown in the women and children that survived and assuming all men dead.


    ''' Random Forest '''
    # Fit the training data to the survived labels and create the decision trees
    x_train = train_data[0::, 1::]
    y_train = train_data[0::, 0]

    # Random forest classifier based on cross validation parameter dictionary
    # Create the random forest object which will include all the parameters for the fit
    forest = RandomForestClassifier(max_features='sqrt')#n_estimators=100)#, n_jobs=-1)#, max_depth=None, min_samples_split=2, random_state=0)#, max_features=np.sqrt(5))
    parameter_grid = {'max_depth': [4,5,6,7,8], 'n_estimators': [200,210,240,250],'criterion': ['gini', 'entropy']}
    cross_validation = StratifiedKFold(n_splits=2, random_state=None, shuffle=False)  #, n_folds=10)
    grid_search = GridSearchCV(forest, param_grid=parameter_grid, cv=cross_validation)
    grid_search = grid_search.fit(x_train, y_train)
    output = grid_search.predict(test_data)
    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))

    # Random forest (rf) classifier for feature selection
    forest_feature_selection = RandomForestClassifier(max_features='sqrt')#n_estimators=100)#, n_jobs=-1)#, max_depth=None, min_samples_split=2, random_state=0)#, max_features=np.sqrt(5))
    forest_feature_selection = forest_feature_selection.fit(x_train, y_train)
    score = forest_feature_selection.score(x_train, y_train)
    print '\nSCORE random forest train data:---------------------------------------------------'
    print score
    # print titanic_panda_inst.compute_score_crossval(forest_feature_selection, x_train, y_train)
    # Take the same decision trees and run it on the test data
    # output = forest_feature_selection.predict(test_data)

    # Explore prediction data
    # print np.histogram(output, bins=[0, 1, 2])
    # plt.hist(output, bins='auto')
    # plt.show()
    # plt.close()
    # print np.argmax(np.histogram(output, bins=[0, 1, 2])[0])


    # Evaluate variable importance with no cross validation
    importances = forest_feature_selection.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest_feature_selection.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    print '\nFeatures:'
    print np.reshape(np.append(np.array(list(df_test)), np.arange(0, len(list(df_test)))), (len(list(df_test)), 2), 'F')#, 2, len(list(df_test)))

    print 'Feature ranking:'
    for f in range(x_train.shape[1]):
        print '%d. feature %d (%f)' % (f + 1, indices[f], importances[indices[f]])

    # Select most important features
    feature_selection_model = SelectFromModel(forest_feature_selection, prefit=True)
    x_train_new = feature_selection_model.transform(x_train)
    print x_train_new.shape
    test_data_new = feature_selection_model.transform(test_data)
    print test_data_new.shape
    # We get that four features are selected

    forest = forest.fit(x_train_new, y_train)
    score = forest.score(x_train_new, y_train)
    print '\nSCORE random forest train data (feature select):---------------------------------------------------'
    print score
    print titanic_panda_inst.compute_score_crossval(forest, x_train_new, y_train)




    feature_ranking_plot = 0
    if feature_ranking_plot:
        plt.figure()
        plt.title('Feature importances')
        plt.bar(range(x_train.shape[1]), importances[indices], color='r', yerr=std[indices], align='center')
        plt.xticks(range(x_train.shape[1]), indices)
        plt.xlim([-1, x_train.shape[1]])
        plt.show()
        plt.close()



    ''' Logistic Regression '''
    logreg = LogisticRegression(max_iter=100, random_state=42)
    logreg.fit(x_train, y_train)
    y_pred = logreg.predict(test_data)
    # output = y_pred
    score = logreg.score(x_train, y_train)
    print '\nSCORE Logistic regression training data:---------------------------------------------------'
    print score


    ''' Gaussian naive Bayes'''
    gaussian = GaussianNB()
    gaussian.fit(x_train, y_train)
    Y_pred = gaussian.predict(test_data)
    score = gaussian.score(x_train, y_train)
    print '\nSCORE Gaussian naive Bayes training data:---------------------------------------------------'
    print score


    ''' SVC with RBF kernel '''
    rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=1).fit(x_train, y_train)
    score = rbf_svc.score(x_train, y_train)
    print ''.join(['\nSVC with RBF kernel', ' SCORE test data:---------------------------------------------------'])
    print score



    ''' Isolation Forest '''
    # As data may contain outliers.
    # One efficient way of performing outlier detection in high-dimensional datasets is to use random forests.
    # The ensemble.IsolationForest 'isolates' observations by randomly selecting a feature and
    # then randomly selecting a split value between the maximum and minimum values of the selected feature.
    #
    # Return the anomaly score of each sample using the IsolationForest algorithm.
    isolation_forest_plot = 0
    if isolation_forest_plot:
        correlated_column_1 = 'Pclass'
        correlated_column_2 = 'ChildOrAdult'
        # correlated_column_1 = 'FamilySize'
        # correlated_column_2 = 'AgeFill'
        # correlated_column_1 = 'FareFill'
        # correlated_column_2 = 'Age*Class'
        h = .02  # step size in the mesh
        # h = 50.0


        X, y, xx, yy = titanic_panda_inst.feature_2D_plot(df, correlated_column_1, correlated_column_2, h)
        X_test = df_test[[correlated_column_1, correlated_column_2]].values

        rng = np.random.RandomState(42)
        # Classifier (clf). With two features (for 2D plotting)
        clf = IsolationForest(max_samples=100, random_state=rng)
        clf.fit(X)
        # With all features
        clf_all_features = IsolationForest(max_samples=100, random_state=rng)
        clf_all_features.fit(x_train)

        # Predict if a particular sample is an outlier using all features for higher dimensional data set.
        y_pred_train = clf_all_features.predict(x_train)
        y_pred_test = clf_all_features.predict(test_data)
        # print 'IsolationForest predict if is_inlier or Not (+1 or -1) with all features training data (+1 is good):---------------------------------------------------'
        # print y_pred_train

        # Exclude suggested outlier samples for improvement of prediction power/score
        outlierMapOut_train = np.array(map(lambda x: x == 1, y_pred_train))
        x_train_modified = x_train[outlierMapOut_train, ]
        y_train_modified = y_train[outlierMapOut_train, ]
        # print np.shape(y_train_modified)
        # print x_train_modified

        # Create the random forest object which will include all the parameters for the fit
        forest = RandomForestClassifier(n_estimators=100)  # , n_jobs=-1)#, max_depth=None, min_samples_split=2, random_state=0)#, max_features=np.sqrt(5))

        # Fit the training data to the survived labels and create the decision trees
        forest = forest.fit(x_train_modified, y_train_modified)

        # Take the same decision trees and run it on the test data
        output = forest.predict(test_data)
        score = forest.score(x_train_modified, y_train_modified)
        print 'SCORE random forest with isolation forest train data:---------------------------------------------------'
        print score

        # plot the line, the samples, and the nearest vectors to the plane
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.title("IsolationForest")
        plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
        plt.colorbar(orientation='vertical')
        b1 = plt.scatter(X[:, 0], X[:, 1], c='white')
        b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green')
        plt.axis('tight')
        plt.legend([b1, b2],
                   ["training observations",
                    "new regular observations"],
                   loc="upper left")

        plt.show()
        plt.close()


    # Decision boundary in two dimensions
    decision_boundary_plot = 0
    if decision_boundary_plot:

        correlated_column_1 = 'Pclass'
        correlated_column_2 = 'ChildOrAdult'
        # correlated_column_1 = 'FamilySize'
        # correlated_column_2 = 'AgeFill'
        # correlated_column_1 = 'FareFill'
        # correlated_column_2 = 'Age*Class'
        h = .02  # step size in the mesh
        # h = 50.0

        X, y, xx, yy = titanic_panda_inst.feature_2D_plot(df, correlated_column_1, correlated_column_2, h)
        # we create an instance of SVM and fit out data. We do not scale our
        # data since we want to plot the support vectors
        C = 1.0  # SVM regularization parameter
        svc = svm.SVC(kernel='linear', C=C).fit(X, y)
        rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
        poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
        lin_svc = svm.LinearSVC(C=C).fit(X, y)

        # title for the plots
        titles = ['SVC with linear kernel',
                  'LinearSVC (linear kernel)',
                  'SVC with RBF kernel',
                  'SVC with polynomial (degree 3) kernel']


        # Print the scores
        for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
            score = clf.score(X, y)
            print ''.join([titles[i], 'SCORE train data:---------------------------------------------------'])
            print score


        for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            plt.subplot(2, 2, i + 1)
            plt.subplots_adjust(wspace=0.4, hspace=0.4)

            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

            # Plot also the training points
            plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
            plt.xlabel(correlated_column_1)
            plt.ylabel(correlated_column_2)
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            plt.xticks(())
            plt.yticks(())
            plt.title(titles[i])

        plt.show()
        plt.close()

    '''
    # OOB Errors for Random Forests
    '''
    OOB_error_plot = 0
    if OOB_error_plot:
        random_state = 123

        # NOTE: Setting the `warm_start` construction parameter to `True` disables
        # support for parallelized ensembles but is necessary for tracking the OOB
        # error trajectory during training.
        ensemble_clfs = [
            ("RandomForestClassifier, max_features='sqrt'",
             RandomForestClassifier(warm_start=True, oob_score=True,
                                    max_features="sqrt",
                                    random_state=random_state)),
            ("RandomForestClassifier, max_features='log2'",
             RandomForestClassifier(warm_start=True, max_features='log2',
                                    oob_score=True,
                                    random_state=random_state)),
            ("RandomForestClassifier, max_features=None",
             RandomForestClassifier(warm_start=True, max_features=None,
                                    oob_score=True,
                                    random_state=random_state))
        ]

        # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
        error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

        # Range of `n_estimators` values to explore.
        min_estimators = 15
        max_estimators = 175

        for label, clf in ensemble_clfs:
            for i in range(min_estimators, max_estimators + 1):
                clf.set_params(n_estimators=i)
                clf.fit(x_train, y_train)

                # Record the OOB error for each `n_estimators=i` setting.
                oob_error = 1 - clf.oob_score_
                error_rate[label].append((i, oob_error))

        # Generate the "OOB error rate" vs. "n_estimators" plot.
        for label, clf_err in error_rate.items():
            xs, ys = zip(*clf_err)
            plt.plot(xs, ys, label=label)

        plt.xlim(min_estimators, max_estimators)
        plt.xlabel("n_estimators")
        plt.ylabel("OOB error rate")
        plt.legend(loc="upper right")
        plt.show()
        plt.close()


    ''' Correlation Coefficient using Logistic Regression '''
    coefficients_df = pd.DataFrame(df.columns.delete(0))
    # coefficients_df = pd.DataFrame(df)
    coefficients_df.columns = ['Features']
    coefficients_df['Coefficient estimates'] = pd.Series(logreg.coef_[0])
    print '\n'
    print coefficients_df





    ''' Submission '''
    # Submission requires a csv file with PassengerId and Survived columns.
    dfBestScore = pd.read_csv('/home/user/Documents/Kaggle/Titanic/submission/submissionTitanic.csv', header=0)

    # We do not expect all to be equal since the learned model differs from time to time.
    # print (dfBestScore.values[0::, 1::].ravel() == output.astype(int))
    # print np.array_equal(dfBestScore.values[0::, 1::].ravel(), output.astype(int))  # But they are almost never all equal
    savePath = '/home/user/Documents/Kaggle/Titanic/submission/'
    submission = pd.DataFrame({'PassengerId': passengerId_df_test, 'Survived': output.astype(int)})
    submission.to_csv(''.join([savePath, 'submissionTitanic.csv']), index=False)

    # Using numpy instead of Pandas for submit
    # submitArray = np.array([passengerId_df_test.values, output]).T
    # np.savetxt(''.join([savePath, 'submissionTitanic.csv']), submitArray, delimiter=',')
    # plt.plot(np.array(passengerId_df_test), np.array(output), 'o')
    # plt.show()

if __name__ == '__main__':
    main()