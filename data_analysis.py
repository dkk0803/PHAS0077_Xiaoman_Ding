import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# calculate the diabetic ratio of all the samples, male samples and female samples respectively (data source of Fig 2 in report)
def calculate_diabetic_ratio():
    # split male and female from original dataset
    male = train.loc[train['Gender']=='M']
    female = train.loc[train['Gender']=='F']

    d = 0
    for index, row in train.iterrows():
        if row['diabetic'] == 1:
            d += 1
    print('People with diabetic: ', d/train.shape[0])

    d = 0
    for index, row in male.iterrows():
        if row['diabetic'] == 1:
            d += 1
    print('Male with diabetic: ', d/male.shape[0])

    d = 0
    for index, row in female.iterrows():
        if row['diabetic'] == 1:
            d += 1
    print('Female with diabetic: ', d/female.shape[0])
    print()

# plot the number of diabetic and nondiabetic samples for every age (Fig 3 in report)
def plot_diabetic_count():
    ages = []       # store ages that will be plot in the graph
    ratio = []      # store the diabetic ratio for every age in ages
    dia = []        # store diabetic num for every age
    nondia = []     # store nondiabetic num for every age

    for i in range(19,83):
        specific_age = train.loc[train['Age']==i]
        dia_num = specific_age.loc[specific_age['diabetic'] == 1]
        dia.append(dia_num.shape[0])
        nondia.append(specific_age.shape[0] - dia_num.shape[0])
        ages.append(i)

        if specific_age.shape[0] > 10:
            ratio.append(dia_num.shape[0]/specific_age.shape[0])
        else:
            if len(ratio) > 1:
                ratio.append((ratio[len(ratio)-1] + ratio[len(ratio)-2])/2)
            else:
                ratio.append(0)

            
    plt.bar(ages, nondia, alpha = 0.8, label = 'Non-diabetic')
    plt.bar(ages, dia,  label = 'Diabetic')
    # plt.plot(ages, ratio, label = 'Diabetic ratio')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.title('Age vs. Diabetic & Non-Diabetic Sample Count')
    plt.legend()
    plt.show()

# calculate the missing ratio for every feature in the dataset (Fig 4 in report)
def calculate_missing_ratio():
    data = pd.concat([train,test],axis=0)
    print(data.isnull().sum()/len(data))

    nan_data = (data.isnull().sum() / len(data)) * 100
    nan_data = nan_data.drop(nan_data[nan_data == 0].index).sort_values(ascending=False)
    missing_data = pd.DataFrame({'Missing Ratio' :nan_data})
    missing_data.head(20)

    f, ax = plt.subplots()
    plt.xticks(rotation='90')
    sns.barplot(x=nan_data.index, y=nan_data)
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Percentage of missing values', fontsize=14)
    plt.title('Percentage of missing data by feature', fontsize=14)
    plt.show()

# plot box plot for several features with abnormal values (Fig 5 in report)
def plot_box():
    # box plot
    box_plot = train[['*AST', '*ALT', '*ALP']]
    box_plot.plot.box()
    plt.title('Box plot of several features')
    plt.grid(linestyle="--", alpha=0.3)
    plt.show()

if __name__ == '__main__':
    # read file
    data_path = 'data/'
    train = pd.read_csv(data_path+'train.csv',encoding='gb2312')
    test = pd.read_csv(data_path+'test.csv',encoding='gb2312')

    # print the statistics information of training set
    print(train.describe())

    # add a 'diabetic' label to training set, depending on whether one's blood glucose is over 7.0 mmol/L
    is_diabetic = []
    for index, row in train.iterrows():
        if row['Glu'] >= 7.0:
            is_diabetic.append(1)
        else:
            is_diabetic.append(0)
    train['diabetic'] = is_diabetic

    calculate_diabetic_ratio()
    # plot_diabetic_count()
    calculate_missing_ratio()
    plot_box()
