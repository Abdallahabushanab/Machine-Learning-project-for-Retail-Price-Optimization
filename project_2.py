#  librires 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
sns.set_theme(style="darkgrid")

## Display all rows and columns of a dataframe instead of a truncated version
from IPython.display import display
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Dealing with colors in python
from colorama import Fore

info = pd.read_csv('Desktop\ProjectPro\project_2_pricing\Cafe+-+DateInfo.csv')
specification = pd.read_csv('Desktop\ProjectPro\project_2_pricing\Cafe+-+Sell+Meta+Data.csv')
data = pd.read_csv('Desktop\ProjectPro\project_2_pricing\Cafe+-+Transaction+-+Store.csv')


orginal_data = data.copy()
orginal_info = info.copy()
orginal_specification = specification.copy()

'''
We have to understand what is sell category 

prices Optimizations [analysis customers, prices, objectives, profit, revenue]

predictive behavior of  potential buiers to different type of prodcuts and services 

where this is required?
retail, banking, hotels,airlines, and local.

who need prices?
analyst or business owners

strategies to price products
1) cost plus pricing .
2) competition based pricing. 
3) perecived value based prices(lauxry).
4) demand based pricing.
5) price elasticity. our projects.

what should I price my products?
sales, footfall, revenue, profit

How I can price my products with the sales data I have?
sales data
property - price elasticity
elastic - pattern change based on their price
inelastic

elastic:

responsiveness 
demand and price 'nothing else has change'
desire 
% change of quantity for % increase in price
cafe- burger 
what is the optimal prices to set for their items in order to gain maximum profit.    
'''

info['HOLIDAY'] = info['HOLIDAY'].fillna('Not Holiday') # Dealing with NAN Values
# info['HOLIDAY'] = info['HOLIDAY'].replace(np.nan, 'Not Holiday') # or by this



def understand_our_data(data):
    print('Our Data Info','\n')
    print(data.info(),'\n')
    print('Describe our Numeric data','\n')
    print(data.describe(),'\n')
    print('Describe our Objectiv data','\n')
    print(data.describe(include=['O']),'\n')
    print('Objects columns','\n')
    print(data.dtypes == 'object','\n')
    print('Sorted type of columns','\n')
    print(data.dtypes.sort_values(),'\n')
    print('Number of null values','\n')
    print(data.isna().sum().sort_values(),'\n')
    print('Shape of our Data','\n')
    print(data.shape,'\n')
    print('Number of unique vales','\n')
    print(data.nunique().sort_values(),'\n')
    
understand_our_data(data)
understand_our_data(info)

info.HOLIDAY.value_counts()
info.YEAR.value_counts()
info.CALENDAR_DATE.describe()
data[data.isnull().any(axis=1)]

'''
We have 1349 days 

Info Columns
SELL_ID: a categorical variable, identifier of the combination of items that is
 contained in the product.

SELL_CATEGORY: “0” identifies single products; 
the category “2” identifies the combo ones.

ITEM_ID: a categorical variable, identifier of the item that is contained
 in the product.

ITEM_NAME: a categorical variable, identifying the name of the item

Data columns

Important: It’s supposed the PRICE for that product in that day will not vary.

In details: CALENDAR_DATE: a date/time variable, having the time always set to 00:00 AM.

PRICE: a numeric variable, associated with the price of the product identified by the SELL_ID.

QUANTITY: a numeric variable, associated with the quantity of the product sold, identified by the SELL_ID.

SELL_ID: a categorical variable, identifier of the product sold.

SELL_CATEGORY: a categorical variable, category of the product sold.
'''
# Visualaze our data to see how they behavirs 
unique_values_1 = data.columns
for i in unique_values_1:
    print('for column {} this is unique values'.format(i))
    print('unique values number:{}'.format(len(data[i].value_counts())))
    plt.hist(data[i])
    plt.title(i)
    plt.show()
 
sns.pairplot(data)

# for other dataset
unique_values_2 = info.columns    
for i in unique_values_2:
    print('for column {} this is unique values'.format(i))
    print('unique values number:{}'.format(len(info[i].value_counts())))
    plt.hist(info[i])
    plt.title(i)
    plt.show()

sns.pairplot(info)

# Now we will put two dataset togther but first we must see what is the common columns

'''


'''
print(info['CALENDAR_DATE'].max()); print(info['CALENDAR_DATE'].min())
print(data['CALENDAR_DATE'].max()); print(data['CALENDAR_DATE'].min())

pd.concat([specification['SELL_ID'], pd.get_dummies(specification['ITEM_NAME'])],
          axis=1).groupby(['SELL_ID']).sum()

#Here we saw each sell id what they take 
data_1 = pd.merge(specification.drop('ITEM_ID', axis=1),
                  data.drop('SELL_CATEGORY', axis=1), on = 'SELL_ID')

by_quantity = data_1.groupby(['SELL_ID', 'SELL_CATEGORY', 'ITEM_NAME',
                              'CALENDAR_DATE', 'PRICE']).QUANTITY.sum()
by_quantity =by_quantity.reset_index()

# combined data togther
combined_data = pd.merge(info, by_quantity, on = 'CALENDAR_DATE' )
print(combined_data['CALENDAR_DATE'].max()); print(combined_data['CALENDAR_DATE'].min())

# Check our data
info[info['CALENDAR_DATE'] == '1/1/12']
by_quantity[by_quantity['CALENDAR_DATE'] == '01/01/12']

# Chechk our data
combined_data[combined_data.isnull().any(axis=1)]

# bussiness as usual
bau_data = combined_data[(combined_data['HOLIDAY']=='Not Holiday') & (combined_data['IS_SCHOOLBREAK']==0) & (combined_data['IS_WEEKEND']==0)]


plt.hist(bau_data.ITEM_NAME) # Data exploration
plt.hist(bau_data.PRICE) # Data exploration
plt.scatter(combined_data['PRICE'], combined_data['QUANTITY'])
plt.scatter(bau_data['PRICE'], bau_data['QUANTITY'])

# check relation between quantity and price
sns.pairplot(combined_data[['PRICE', 'QUANTITY','ITEM_NAME']], hue ='ITEM_NAME',
             plot_kws={'alpha': 0.1})
sns.pairplot(bau_data[['PRICE', 'QUANTITY','ITEM_NAME']], hue ='ITEM_NAME',
             plot_kws={'alpha': 0.1})


# Burger showes that high quantity and high prices let's check
burger = combined_data[combined_data['ITEM_NAME'] == 'BURGER'] 
burger.head()
burger.describe()
burger.shape
sns.scatterplot(x='PRICE', y='QUANTITY', data=burger)
sns.scatterplot(x='PRICE', y='QUANTITY', data=burger, hue='SELL_ID',
                legend=True, alpha=0.1)
np.unique(burger['SELL_ID'])


# Sell id 1070 with burger see how it works
burger_1070 = combined_data[(combined_data['ITEM_NAME'] == 'BURGER') &
                            (combined_data['SELL_ID'] == 1070)]
burger_1070.head()
burger_1070.describe()
burger_1070.shape
sns.scatterplot(x='PRICE', y='QUANTITY', data=burger_1070,alpha=0.1)

# modeling
import statsmodels.api as sm
from statsmodels.formula.api import ols

# here we check the model for burger 
def ols_model(burger_1070):
    burger_model = ols('QUANTITY ~ PRICE', data=burger_1070).fit()
    print(burger_model.summary())
    fig = plt.figure(figsize=(12,8)) 
    fig = sm.graphics.plot_partregress_grid(burger_model, fig=fig) # but the model is not fit
    return burger_model

# let's try burger in bussines normal days
burger = bau_data[bau_data['ITEM_NAME'] == 'BURGER'] 
burger.head()
burger.describe()
burger.shape
sns.scatterplot(x='PRICE', y='QUANTITY', data=burger)
sns.scatterplot(x='PRICE', y='QUANTITY', data=burger, hue='SELL_ID',
                legend=True, alpha=0.1)
np.unique(burger['SELL_ID'])

# Bau for 1070
def burger_bau(bau_data, item ,sell_id):
    burger_1070 = bau_data[(bau_data['ITEM_NAME'] == item) &
                            (bau_data['SELL_ID'] == sell_id)]
    burger_1070.head()
    burger_1070.describe()
    burger_1070.shape
    sns.scatterplot(x='PRICE', y='QUANTITY', data=bau_data,alpha=0.1)
    return burger_1070

# but model in bau
ols_model(burger_1070)

# the model become better to get the data
# Now let's calculate with it outdoor
bau_data_1 = combined_data[(combined_data['HOLIDAY']=='Not Holiday') &
                         (combined_data['IS_SCHOOLBREAK']==0) &
                         (combined_data['IS_WEEKEND']==0) &
                         (combined_data['IS_OUTDOOR']==1)]
# here we are not holiday not schoobreak, is not weekend but it is outdoor
# the model give better results
burger_1070 = burger_bau(bau_data=bau_data_1,item ='BURGER',sell_id=1070)
burger_model = ols_model(burger_1070)

# check our data
fig = plt.figure(figsize=(12,8)) 
fig = sm.graphics.plot_regress_exog(burger_model, 'PRICE' ,fig=fig)

# burger 2051
burger_2051 = burger_bau(bau_data=combined_data,item ='BURGER',sell_id=2051)
burger_model = ols_model(burger_2051)

# Other than burger
def others(data, item):
    burger_1070 = data[data['ITEM_NAME'] == item]
    print(burger_1070.head())
    burger_1070.describe()
    burger_1070.shape
    sns.scatterplot(x='PRICE', y='QUANTITY', data=bau_data,alpha=0.1)
    return burger_1070

# for coke
coke = others(data=combined_data,item ='COKE')
coke_model = ols_model(coke)

# for coffe
coffe = others(data=combined_data,item ='COFFEE')
coffe_model = ols_model(coffe)

# for LEMONADE
lemonade = others(data=combined_data,item ='LEMONADE')
lemonade_model = ols_model(lemonade)

# find coefecient 
elasticities = {}
def create_model_and_find_elasticity(data):
    model = ols("QUANTITY ~ PRICE", data).fit()
    price_elasticity = model.params[1]
    print("Price elasticity of the product: " + str(price_elasticity))
    print(model.summary())
    fig = plt.figure(figsize=(12,8))
    fig = sm.graphics.plot_partregress_grid(model, fig=fig)
    return price_elasticity, model

# burger elastics paramaters 
price_elasticity, model_burger_1070 = create_model_and_find_elasticity(burger_1070)
elasticities['burger_1070'] = price_elasticity

burger2051_data = bau_data_1[(bau_data_1['ITEM_NAME'] == "BURGER") &
                             (bau_data_1['SELL_ID'] == 2051)]
elasticities['burger_2051'], model_burger_2051 = create_model_and_find_elasticity(burger2051_data)

burger2052_data = bau_data_1[(bau_data_1['ITEM_NAME'] == "BURGER") &
                             (bau_data_1['SELL_ID'] == 2052)]
elasticities['burger_2052'], model_burger_2052 = create_model_and_find_elasticity(burger2052_data)

burger2053_data = bau_data_1[(bau_data_1['ITEM_NAME'] == "BURGER") &
                             (bau_data_1['SELL_ID'] == 2053)]
elasticities['burger_2053'], model_burger_2053 = create_model_and_find_elasticity(burger2053_data)

# COKE elastics paramaters 
coke_data_2053 = bau_data_1[(bau_data_1['ITEM_NAME'] == "COKE") &
                            (bau_data_1['SELL_ID'] == 2053)]
elasticities['coke_2053'], model_coke_2053 = create_model_and_find_elasticity(coke_data_2053)

coke_data_2051 = bau_data_1[(bau_data_1['ITEM_NAME'] == "COKE") &
                            (bau_data_1['SELL_ID'] == 2051)]
elasticities['coke_2051'], model_coke_2051 = create_model_and_find_elasticity(coke_data_2051)

# Lemonade elastics paramaters 
lemonade_data_2052 = bau_data_1[(bau_data_1['ITEM_NAME'] == "LEMONADE") &
                                (bau_data_1['SELL_ID'] == 2052)]
elasticities['lemonade_2052'], model_lemonade_2052 = create_model_and_find_elasticity(lemonade_data_2052)

# coffee elastics paramaters 
coffee_data_2053 = bau_data_1[(bau_data_1['ITEM_NAME'] == "COFFEE") &
                              (bau_data_1['SELL_ID'] == 2053)]
elasticities['coffee_2053'], model_coffee_2053 = create_model_and_find_elasticity(coffee_data_2053)

# find optimal prices for maximum profit
coke_data = coke_data_2053
coke_data.PRICE.min()
coke_data.PRICE.max()

buying_price_coke = 9

'''
𝑐𝑜𝑘𝑒𝑑𝑎𝑡𝑎.𝑃𝑅𝑂𝐹𝐼𝑇=(𝑐𝑜𝑘𝑒𝑑𝑎𝑡𝑎.𝑃𝑅𝐼𝐶𝐸−𝑏𝑢𝑦𝑖𝑛𝑔𝑝𝑟𝑖𝑐𝑒𝑐𝑜𝑘𝑒)∗𝑐𝑜𝑘𝑒𝑑𝑎𝑡𝑎.𝑄𝑈𝐴𝑁𝑇𝐼𝑇𝑌

'''
def find_optimal_price(data,model, buying_price):
    start_price = data.PRICE.min() - 1
    ind_price = data.PRICE.max() + 10
    test = pd.DataFrame(columns=['PRICE','QUANTITY'])
    test['PRICE'] = np.arange(start_price,ind_price,0.01)
    test['QUANTITY'] = model.predict(test['PRICE'])
    test['PROFIT'] = (test['PRICE'] - buying_price )*test['QUANTITY']
    plt.plot(test['PRICE'],test['QUANTITY'], color='Green')
    plt.plot(test['PRICE'],test['PROFIT'], color = 'Red')
    plt.legend()
    plt.show()
    ind = np.where(test['PROFIT'] ==test['PROFIT'].max())[0][0]
    values_at_max_profit = test.iloc[[ind]]
    return values_at_max_profit

optimal_price = {}
buying_price = 9

optimal_price['burger_1070'] = find_optimal_price(burger_1070, model_burger_1070, buying_price)
optimal_price['burger_2051'] = find_optimal_price(burger2051_data, model_burger_2051, buying_price)
optimal_price['burger_2052'] = find_optimal_price(burger2052_data, model_burger_2052, buying_price)
optimal_price['burger_2053'] = find_optimal_price(burger2053_data, model_burger_2053, buying_price)
optimal_price['coke_2051'] = find_optimal_price(coke_data_2051, model_coke_2051, buying_price)
optimal_price['coke_2053'] = find_optimal_price(coke_data_2053, model_coke_2053, buying_price)
optimal_price['lemonade_2052'] = find_optimal_price(lemonade_data_2052, model_lemonade_2052, buying_price)
optimal_price['coffee_2053'] = find_optimal_price(coffee_data_2053, model_coffee_2053, buying_price)

print(optimal_price)

coke_data_2051.PRICE.describe()

'''
Conclusion¶
This is the price the cafe should set on it's item to earn maximum profit based on it's previous sales data. It is important to note that this is on a normal day. On 'other' days such as a holiday, or an event taking place have a different impact on customer buying behaviours and pattern. Usually an increase in consumption is seen on such days. These must be treated separately. Similarly, it is important to remove any external effects other than price that will affect the purchase behaviours of customers including the datapoints when the item was on discount.

Once, the new prices are put up, it is important to continuously monitor the sales and profit. If this method of pricing is a part of a rpoduct, a dashboard can be created for the purpose of monitoring these items and calculating the lift in the profit.

'''



