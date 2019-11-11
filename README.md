# Active-Stock-Market-Management
NASDAQ Technical and fundamental analysis, and prediction of beating the index over the coming year


# Data
#### Data from WRDS and analysis process in another file.
## Variables includes: 

### Fondamentals: 
    # Yield quarter
    # marge de profit
    # P/E
    # Total assets
    # Total liabilities
    # B
    # P/B
    # Nb shares
    # % shares issued
    # % shares repurchased

### Technical:
    # Price variation
    # Momentum 3 months
    # Momentum 6 months
    # Median quarter
    # Beat median

# Models
    # Random forest
    # Gradient Boosting
    # Linear Support Vector Classification
 The prodiction can be quickly impove by using another imputer (mean imputations were used) and with a model tunnning.


# Results
 ## Models are better to prediction the 'non-beating index'stocks than the one that will beat the index.
 ## On test set (2017) the non beating median precision were 67%. This result was also found on the S&P500 and the DOW. 
 ## The use of such model can be in a context of FNB, where the model can suggest sotcks that won't perform compare to the index, en then have a better portfolio than the Index.  
