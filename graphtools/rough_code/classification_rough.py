# For generating a dictionary structure
#%%
'''arr = {}
for label in labels:
    arr[label] = {'mean_FA':[] ,'mean strl': [], 'num strl': []}
    for i in range(3):
        key = list(arr[label].keys())[i]
        for edge in range(i,(i+1)*7056):
            if edge not in labels:
                #print (edge, f_score(data_edges, edge, 'NEOFAC_A'))
                arr[label][key].append(f_score(data_edges,edge, label))'''
#%%
#%%
SVM_acc = {}
RF_acc = {}
for i in range(5):# different labels
    print(labels[i], ':', big5[i])
    SVM_acc[labels[i]] = {}
    RF_acc[labels[i]] = {}
    for j in range(3):
        print(edge_names[j])
        f_fscores = fscores[i][j] #for the label and the type of edge
        SVM_acc[labels[i]][edge_names[j]] = {}
        RF_acc[labels[i]][edge_names[j]] = {}
        #stats.iqr(f_fscores)
        for per in [5,10,15,20]:
            val = np.percentile(f_fscores, 100-per)
            index = np.where(f_fscores >= val)
            #print(f'Number of indexes where the values are in the last {per} percentile:', len(index[0]))

            Y = np.array(data[labels[i]] >= data[labels[i]].median()).astype(int)
            X = np.reshape(whole, (whole.shape[0], 3,whole.shape[1]//3))[:, j,index[0]]

            X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.80, test_size=0.2)
            clf = make_pipeline(Normalizer(), SVC(gamma='auto'))
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = sum(y_pred==y_test)/len(y_test)
            #print(acc,'SVM', label)
            SVM_acc[labels[i]][edge_names[j]][per] = acc

            clf = RandomForestClassifier(max_depth=2, random_state=0)
            clf.fit(X_train, y_train)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = sum(y_pred==y_test)/len(y_test)
            RF_acc[labels[i]][edge_names[j]][per] = acc
            #print(acc, 'RF', label)