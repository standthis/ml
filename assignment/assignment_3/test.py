import sys
from difflib import get_close_matches
classdiff = ['k-Nearest Neighbours', 'Naive Bayes', 'Logistic Regression',
        'Linear SVM', 'Decision Tree', 'Random Forest', 'Multilayer Perceptron']
stuff = get_close_matches(sys.argv[1], classdiff, n=1, cutoff=0.1)
print(stuff)
