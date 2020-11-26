### searching for the best hyperparams
 """
 References:
 https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
 https://stackoverflow.com/questions/49823192/autoencoder-gridsearch-hyperparameter-tuning-keras
 https://towardsdatascience.com/autoencoders-for-the-compression-of-stock-market-data-28e8c1a2da3e
 """

 SIZES = [X.shape[1],RED_DIM]
 model = KerasClassifier(build_fn=create_ae,verbose=False)

 batch_size = [1,4,16,32,64]
 epochs = [10, 25, 50,100,200]
 init = ['uniform', 'normal']
 activation = ['linear','relu','sigmoid']
 optimizer = ['SGD', 'Adam','rmsprop','adadelta']
 param_grid = dict(batch_size=batch_size,
                   epochs=epochs,
                   init=init,
                   activation=activation,
                   optimizer=optimizer,
                  )

 search = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3,verbose=True)
 search_results = search.fit(X, X)
 score = search_results.best_score_
 params = search_results.best_params_

 print("\nScore: %f" % (score))
 print("Selected hyperparameters: %s" % (params))
 
### fit model with selected params

 ae, encoder, decoder = create_ae(params['init'],params['activation'],params['optimizer'],ret_comp=True)

 ### train the autoencoder

 X_tr, X_te = train_test_split(X,test_size=0.1,random_state=42)
 history = ae.fit(X_tr, X_tr,
                 epochs=params['epochs'],
                 batch_size=params['batch_size'],
                 shuffle=True,
                 validation_data=(X_te, X_te))
