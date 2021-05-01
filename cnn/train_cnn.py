from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import pickle
from main import *
from net import *
import os

os.system('clear')

parser = argparse.ArgumentParser(description='Train a convolutional neural network.')
parser.add_argument('save_path', metavar = 'Save Path', help='name of file to save parameters in.')

args = parser.parse_args()
save_path = args.save_path

cost = train(save_path = save_path)
print(cost)

params = pickle.load(open(save_path, 'rb'))
[f1, f2, w3, w4, b1, b2, b3, b4] = params

# Plot cost 
# plt.plot(cost, 'r')
# plt.xlabel('# Iterations')
# plt.ylabel('Cost')
# plt.legend('Loss', loc='upper right')
# plt.show()

data = pd.read_pickle('dataset.pkl')
train, test = train_test_split(data, test_size=0.2, random_state=42, shuffle=True) # Test 20%
test = test.reset_index().drop('index', axis=1)
print(test.head())
ord_enc = OrdinalEncoder()
test['accent_code'] = ord_enc.fit_transform(test[['accent']]).astype(int)
test.drop('sex filename accent'.split(), axis=1, inplace=True)
y = []
X = []
for i, row in test.iterrows():
    X.append(row['mfccs'])
    y.append(row['accent_code'])
X = np.array(X)
y = np.array(y).reshape(len(y), 1)
X = X.reshape((len(X), X.shape[1] * X.shape[2]))
print(X.shape, y.shape)

# Get test data
# m =10000
# X = extract_data('t10k-images-idx3-ubyte.gz', m, 28)
# y_dash = extract_labels('t10k-labels-idx1-ubyte.gz', m).reshape(m,1)
# Normalize the data
X-= int(np.mean(X)) # subtract mean
X/= int(np.std(X)) # divide by standard deviation
test_data = np.hstack((X,y))

X = test_data[:,0:-1]
X = X.reshape(len(test_data), 1, 20, 44)
y = test_data[:,-1]

corr = 0
digit_count = [0 for i in range(8)]
digit_correct = [0 for i in range(8)]

print()
print("Computing accuracy over test set:")

t = tqdm(range(len(X)), leave=True)

for i in t:
    x = X[i]
    pred, prob = predict(x, f1, f2, w3, w4, b1, b2, b3, b4)
    digit_count[int(y[i])]+=1
    if pred==y[i]:
        corr+=1
        digit_correct[pred]+=1

    t.set_description("Acc:%0.2f%%" % (float(corr/(i+1))*100))
    
print("Overall Accuracy: %.2f" % (float(corr/len(test_data)*100)))
x = np.arange(8)
digit_recall = [x/y for x,y in zip(digit_correct, digit_count)]
plt.xlabel('Accents')
plt.ylabel('Recall')
plt.title("Recall on Test Set")
plt.bar(x,digit_recall)
plt.show()