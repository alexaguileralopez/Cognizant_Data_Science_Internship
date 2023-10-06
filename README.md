# COGNIZANT DATA SCIENCE INTERSHIP

Gala Groceries is a technology-led grocery store chain based in the USA. They rely heavily on new technologies, such as IoT to give them a competitive edge over other grocery stores.

The company approached Cognizant to help them with a supply chain issue. Groceries are highly perishable items. If you overstock, you are wasting money on excessive storage and waste, but if you understock, then you risk losing customers. They want to know how to better stock the items that they sell. 

## TASK 1 - EXPLORATORY DATA ANALYSIS

The client has provided us with a sample of their sales data. In this task, that dataset will be analysed to get some basic information out of a sample of their data. 

The code for this task is [here](<TASK 1/eda.ipynb>)

The results lead to the following conclusion:

- Fruit is the category that sells the most, followed by vegetables. The categories with the least sales are spices and herbs, and pets.
- There are four types of memberships (standard, basic, gold,premium). But the non-members are the most frequent buyers.
- Cash is the most used payment type.
- The most common hour to purchase is 11 am.

Based on the EDA, we suggest to the client to provide us with more data. The sales data gives us trends on sales, but we cannot get information on product stock. Also, the current sample only covers 7 days.

## TASK 2 - DATA MODELLING

Based on the previous analysis, the client wants to focus on the following statement:

"Can we accurately predict the stock levels of products based on sales data and sensor data on an hourly basis in order to more intelligently procure products from our suppliers?"

We receive extra information, we now have 3 datasets:

![datasets](<TASK 2/datasets.png>)

Now we can create a strategic plan to approach the problem. The plan is in [strategicplan](<TASK 2/Presentación1.pptx>)


## TASK 3 - MODELLING

Now that we have 3 datasets, we clean them, merge them, and perform feature engineering to get a more accurate result when deploying our model. The categories, days of the week, months, and hours are transformed into numeric values to give solidity to our results.

The model is built using the sklearn package. In this case, we are using a Random Forest Regressor. The model loads the engineered data into a model class, that uses the Random Forest Algorithm to predict stock percentage with all the features developed. Using k-fold cross-validation, we get a Mean Absolute Error of 0.24. 

The code for this task is under [modelling](<TASK 3/modeling.ipynb>)

Using Matplotlib, we can get a graph that shows the importance of the features when predicting the stock percentage:
![results](<TASK 3/output_1.png>)

These results are better visualised in a pie chart: 

![results](<TASK 3/Imagen 1.png>)

Here, we observe that the most important feature is the unit price, followed by the temperature at which the product is kept. Therefore, there is a correlation between the stock percentage and its temperature in the warehouse, which could not have been predicted at a glance.

## TASK 4 - MACHINE LEARNING PRODUCTION

In this task, we are preparing a python module that implements the algorithm on the data. The [folder](<TASK 4>) contains 3 python files that will be used for this task. OOP was used to create a model class for simplicity. 

- [data_loader](<TASK 4/data_loader.py>) contains the preprocessing of the data for our example datasets. This file is used to make an example.
- [model](<TASK 4/modelling.py>) contains the Regression Model class, which can be modified in order to try different algorithms. 
- [module.py](<TASK 4/module.py>) contains the module to be deployed. The following bit of code will be run.  

    if __name__ == '__main__':

    # load_data was created for example usage
    #df = load_data()
    df = load_csv() 
    X,y = create_target_and_predictors(data= df)
    train_algorithm_with_cross_validation(X,y)





