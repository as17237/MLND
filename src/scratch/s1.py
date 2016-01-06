from sklearn import datasets

if __name__ == '__main__':
    boston = datasets.load_boston()
    housing_prices = boston.target
    housing_features = boston.data
    print("Size of data (number of houses) is %s" % len(housing_features))
    print("Number of features? is %s" % len(housing_features[0]))


    # print type(housing_prices)
