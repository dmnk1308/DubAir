url_listing = "http://data.insideairbnb.com/ireland/leinster/dublin/2021-11-07/data/listings.csv.gz"
listings = pd.read_csv(url_listing)
# remove extreme prices
price = listings["price"]
price = price.str.replace("$","")
price = price.str.replace(",","")
price = price.astype(float)
filter = price < 500
listings = listings[filter]
X_train, _ = train_test_split(listings, random_state = 123, test_size = 0.2)


for i in ods:
    for j in fds:
        
        mae_min = []
        mse_min = []
        r2_max = []

        kf = KFold(n_splits=5, shuffle=True, random_state=1234)
        for train_index, test_index in kf.split(X_train):
            print("TRAIN:", train_index, "TEST:", test_index)
            # X_train_kf, X_val_kf = X_train.iloc[train_index, :], X_train.iloc[test_index, :]
            # y_train_kf, y_val_kf = y_train.iloc[train_index], y_train.iloc[test_index]

            X_train_kf, X_test, X_val_kf, y_train_kf, y_test, y_val_kf = load_data_cv(for_dendro = False, train_idx = train_index, val_idx = test_index)
            bin_col = [col for col in X_train_kf if np.isin(X_train_kf[col].unique(), [0, 1]).all()]
            num_col = [col for col in X_train_kf if ~np.isin(X_train_kf[col].unique(), [0, 1]).all()]
            col_names = bin_col + num_col
            train_size = int(X_train_kf.shape[0] * 0.9)
            batch_size = int(X_train_kf.shape[0] * 0.1)

            data_train = tf.data.Dataset.from_tensor_slices({"features": X_train_kf, "price": y_train_kf})
            data_train = data_train.shuffle(6000, seed = 13)
            train_dataset = data_train.take(len(X_train_kf))
            train_dataset = train_dataset.map(transform)
            train_dataset = train_dataset.batch(batch_size)

            data_test = tf.data.Dataset.from_tensor_slices({"features": X_val_kf, "price": y_val_kf})
            test_dataset = data_test.take(len(X_val_kf))
            test_dataset = test_dataset.map(transform)
            test_dataset = test_dataset.batch(batch_size)

            feature_columns = []

            for col in col_names:
                feature_columns.append(tf.feature_column.numeric_column(col))

            model = tabnet.TabNetRegression(feature_columns, num_regressors=1,
                                            output_dim=i, feature_dim=i+j, num_groups=1,
                                            num_decision_steps=2)
                                            
            lr = 0.01
            optimizer = tf.keras.optimizers.Adam(lr)
            model.compile(optimizer, loss=['mse', "mae"] , metrics=[R_squared, "mse", "mae"])

            hist_model = model.fit(train_dataset, epochs=50, 
                                validation_data=test_dataset, verbose=1)

            mae_min.append(np.min(hist_model.history["val_mae"]))
            mse_min.append(np.min(hist_model.history["val_mse"]))
            r2_max.append(np.max(hist_model.history["val_R_squared"]))


        mae_min_m = np.mean(mae_min)
        mse_min_m = np.mean(mse_min)
        r2_max_m = np.mean(r2_max)

        overview_mae.at[i, j] = mae_min_m
        overview_mse.at[i, j] = mse_min_m
        overview_r2.at[i, j] = r2_max_m