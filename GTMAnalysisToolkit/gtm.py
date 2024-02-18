import math
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm, multivariate_normal
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class GTM:
    def __init__(self, shape_of_map=[30, 30], shape_of_rbf_centers=[10, 10],
                 variance_of_rbfs=4, lambda_in_em_algorithm=0.001,
                 number_of_iterations=200, display_flag=1, sparse_flag=False):
        """
        Initializes the GTM model with the specified parameters.

        Parameters
        ----------
        shape_of_map : list, optional
            The dimensions of the map grid (default is [30, 30]).
        shape_of_rbf_centers : list, optional
            The dimensions for the arrangement of RBF (Radial Basis Function) centers (default is [10, 10]).
        variance_of_rbfs : float, optional
            The variance of the RBFs, controlling their spread (default is 4).
        lambda_in_em_algorithm : float, optional
            The regularization parameter used in the EM (Expectation Maximization) algorithm (default is 0.001).
        number_of_iterations : int, optional
            The number of iterations for the EM algorithm to run (default is 200).
        display_flag : int, optional
            Flag to control the display of intermediate results (default is 1, displaying results).
        sparse_flag : bool, optional
            Flag to indicate whether a sparse representation is used (default is False).
        """
        self.shape_of_map = shape_of_map
        self.shape_of_rbf_centers = shape_of_rbf_centers
        self.variance_of_rbfs = variance_of_rbfs
        self.lambda_in_em_algorithm = lambda_in_em_algorithm
        self.number_of_iterations = number_of_iterations
        self.display_flag = display_flag
        self.sparse_flag = sparse_flag

    def calculate_grids(self, num_x, num_y):
        """
        Calculate grid coordinates on the GTM map. This method generates a meshgrid
        representing the GTM's discrete lattice or map space, which is later used for
        visualization or mapping data points.

        Parameters
        ----------
        num_x : int
            The number of grid points along the x-axis.
        num_y : int
            The number of grid points along the y-axis.

        Returns
        -------
        grids : numpy.ndarray
            A 2D array of shape (num_x*num_y, 2), where each row represents the normalized
            coordinates of a grid point on the map.
        """
        # Create a meshgrid of x and y coordinates with specified dimensions
        grids_x, grids_y = np.meshgrid(np.arange(0.0, num_x), np.arange(0.0, num_y))

        # Flatten the meshgrid arrays and concatenate them to form a 2D array of grid points
        grids = np.c_[np.ndarray.flatten(grids_x)[:, np.newaxis],
        np.ndarray.flatten(grids_y)[:, np.newaxis]]

        # Find the maximum values along each axis to use for normalization
        max_grids = grids.max(axis=0)

        # Normalize the grid coordinates to be between -1 and 1
        grids[:, 0] = 2 * (grids[:, 0] - max_grids[0] / 2) / max_grids[0]
        grids[:, 1] = 2 * (grids[:, 1] - max_grids[1] / 2) / max_grids[1]

        return grids

    def fit(self, input_dataset):
        """
        Train the GTM map using the Expectation-Maximization algorithm. This method
        initializes the model parameters using PCA and then iteratively refines them
        to best represent the input dataset in a lower-dimensional space.

        Parameters
        ----------
        input_dataset : numpy.array or pandas.DataFrame
            The training dataset for the GTM. The dataset must be autoscaled, meaning
            that it should have zero mean and unit variance before being passed to this method.
        """
        input_dataset = np.array(input_dataset)  # Ensure the input dataset is a NumPy array
        self.success_flag = True  # Flag to indicate if the training was successful

        # Ensure map and RBF center shapes are integers
        self.shape_of_map = [int(self.shape_of_map[0]), int(self.shape_of_map[1])]
        self.shape_of_rbf_centers = [int(self.shape_of_rbf_centers[0]), int(self.shape_of_rbf_centers[1])]

        # Generate the RBF grid coordinates
        self.rbf_grids = self.calculate_grids(self.shape_of_rbf_centers[0], self.shape_of_rbf_centers[1])

        # Generate the map grid coordinates
        self.map_grids = self.calculate_grids(self.shape_of_map[0], self.shape_of_map[1])

        # Calculate the matrix of distances between map and RBF grid points, then use it to compute phi
        distance_between_map_and_rbf_grids = cdist(self.map_grids, self.rbf_grids, 'sqeuclidean')
        self.phi_of_map_rbf_grids = np.exp(-distance_between_map_and_rbf_grids / (2.0 * self.variance_of_rbfs))

        # Initialize model parameters using PCA
        pca_model = PCA(n_components=3)
        pca_model.fit_transform(input_dataset)
        if np.linalg.matrix_rank(self.phi_of_map_rbf_grids) < min(self.phi_of_map_rbf_grids.shape):
            self.success_flag = False  # Check for sufficient rank to proceed
            return

        # Initial weights and beta parameter estimation
        self.W = np.linalg.pinv(self.phi_of_map_rbf_grids).dot(self.map_grids.dot(pca_model.components_[0:2, :]))
        self.beta = min(pca_model.explained_variance_[2], 1 / (
                (cdist(self.phi_of_map_rbf_grids.dot(self.W), self.phi_of_map_rbf_grids.dot(self.W))
                 + np.diag(np.ones(np.prod(self.shape_of_map)) * 10 ** 100)).min(axis=0).mean() / 2))
        self.bias = input_dataset.mean(axis=0)  # Calculate the bias as the mean of the input dataset

        # Initialize mixing coefficients uniformly
        self.mixing_coefficients = np.ones(np.prod(self.shape_of_map)) / np.prod(self.shape_of_map)

        # Begin the EM algorithm
        phi_of_map_rbf_grids_with_one = np.c_[self.phi_of_map_rbf_grids, np.ones((np.prod(self.shape_of_map), 1))]
        for iteration in range(self.number_of_iterations):
            # E-step: compute responsibilities
            responsibilities = self.responsibility(input_dataset)

            # M-step: update model parameters
            phi_t_G_phi_etc = phi_of_map_rbf_grids_with_one.T.dot(
                np.diag(responsibilities.sum(axis=0)).dot(phi_of_map_rbf_grids_with_one)
            ) + self.lambda_in_em_algorithm / self.beta * np.identity(phi_of_map_rbf_grids_with_one.shape[1])

            # Check condition number to prevent numerical instability
            if 1 / np.linalg.cond(phi_t_G_phi_etc) < 10 ** -15:
                self.success_flag = False
                break

            # Update weights and bias
            self.W_with_one = np.linalg.inv(phi_t_G_phi_etc).dot(
                phi_of_map_rbf_grids_with_one.T.dot(responsibilities.T.dot(input_dataset)))
            self.beta = input_dataset.size / (
                    responsibilities * cdist(input_dataset,
                                             phi_of_map_rbf_grids_with_one.dot(self.W_with_one)) ** 2).sum()

            self.W = self.W_with_one[:-1, :]  # Update weights excluding the last element (bias)
            self.bias = self.W_with_one[-1, :]  # Update bias

            # If sparse representation is enabled, update mixing coefficients
            if self.sparse_flag:
                self.mixing_coefficients = sum(responsibilities) / input_dataset.shape[0]

            # Optionally display the likelihood (or other metric) to monitor convergence
            if self.display_flag:
                print("{0}/{1} ... likelihood: {2}".format(iteration + 1,
                                                           self.number_of_iterations,
                                                           self.likelihood_value))

    def calculate_distance_between_phi_w_and_input_distances(self, input_dataset):
        """
        Calculate the squared Euclidean distance between the input dataset and its
        transformation by the GTM model. This distance measures how well the GTM model
        is able to represent the high-dimensional input data in the lower-dimensional
        GTM space.

        Parameters
        ----------
        input_dataset : numpy.array
            The input dataset for the GTM. This should be the same dataset or a subset
            of the dataset used to train the GTM model.

        Returns
        -------
        distance : numpy.array
            A 2D array of squared Euclidean distances between each point in the input dataset
            and its corresponding point in the GTM model's transformed space. Each row
            corresponds to a data point in the input dataset, and each column corresponds
            to a point in the GTM space.
        """
        # Transform the input dataset using the GTM model. This involves multiplying the
        # RBF activation matrix (phi_of_map_rbf_grids) by the weight matrix (W), and then
        # adding the bias. The operation expands the bias to match the dimensions required
        # for addition with the transformed data points.
        transformed_input = self.phi_of_map_rbf_grids.dot(self.W) + np.ones((np.prod(self.shape_of_map), 1)).dot(
            np.reshape(self.bias, (1, len(self.bias)))
        )

        # Calculate the squared Euclidean distance between the original input dataset and
        # the transformed dataset. This distance quantifies the discrepancy between the
        # original high-dimensional data and its representation in the GTM model's space.
        distance = cdist(input_dataset, transformed_input, 'sqeuclidean')

        return distance

    def responsibility(self, input_dataset):
        """
        Calculates the responsibilities, which indicate the probability distribution of
        each data point over the map grid points, and computes the likelihood of the
        input dataset given the current model parameters. Responsibilities are used in
        the E-step of the EM algorithm to update model parameters.

        Parameters
        ----------
        input_dataset : numpy.array or pandas.DataFrame
            The training dataset for the GTM. The input dataset must be autoscaled,
            meaning that each feature should be centered to have mean zero and scaled
            to have unit variance.

        Returns
        -------
        responsibilities : numpy.array
            A 2D array where each element (i, j) represents the responsibility of the
            jth grid point for the ith data point in the input dataset.

        Note: The method originally intended to return both responsibilities and the
        likelihood value. However, the current implementation only returns the
        responsibilities. The likelihood computation is performed but not returned.
        The likelihood value is instead updated as an attribute of the class.
        """
        input_dataset = np.array(input_dataset)  # Ensure input is a numpy array
        # Calculate the squared Euclidean distance between transformed dataset and input
        distance = self.calculate_distance_between_phi_w_and_input_distances(input_dataset)
        # Compute RBF activations for responsibilities, scaled by the mixing coefficients
        rbf_for_responsibility = np.exp(-self.beta / 2.0 * distance) * self.mixing_coefficients
        # Sum of RBF activations for normalization
        sum_of_rbf_for_responsibility = rbf_for_responsibility.sum(axis=1)
        # Handle cases where the sum is zero to avoid division by zero
        zero_sample_index = np.where(sum_of_rbf_for_responsibility == 0)[0]
        if len(zero_sample_index) > 0:
            sum_of_rbf_for_responsibility[zero_sample_index] = 1
            rbf_for_responsibility[zero_sample_index, :] = 1 / rbf_for_responsibility.shape[1]
        # Normalize RBF activations to compute responsibilities
        responsibilities = rbf_for_responsibility / np.reshape(sum_of_rbf_for_responsibility,
                                                               (rbf_for_responsibility.shape[0], 1))
        # Compute the likelihood of the input dataset
        self.likelihood_value = (np.log((self.beta / 2.0 / np.pi) ** (input_dataset.shape[1] / 2.0) /
                                        (np.exp(-self.beta / 2.0 * distance) * self.mixing_coefficients).sum(
                                            axis=1))).sum()

        return responsibilities

    def likelihood(self, input_dataset):
        """
        Calculate the log-likelihood of the input dataset under the current GTM model.
        This function quantifies how well the GTM model represents the input data. The
        log-likelihood is computed based on the distance between the input dataset and
        its transformation by the model, factoring in the model's parameters such as the
        precision parameter (beta) and the mixing coefficients.

        Parameters
        ----------
        input_dataset : numpy.array or pandas.DataFrame
            The input dataset for which the likelihood is to be calculated. This dataset
            should be the same as or a subset of the dataset used to train the GTM model.
            It must be autoscaled, meaning it should have zero mean and unit variance
            prior to being passed to this function.

        Returns
        -------
        likelihood : float
            The log-likelihood of the input dataset given the current state of the GTM
            model. This scalar value serves as an indicator of how well the model fits
            the data. Higher values indicate a better fit.
        """
        # Convert the input dataset to a NumPy array, if it's not already one.
        input_dataset = np.array(input_dataset)

        # Calculate the distance between the input dataset and its transformation by the model.
        distance = self.calculate_distance_between_phi_w_and_input_distances(input_dataset)

        # Compute the log-likelihood of the input dataset. This involves calculating a
        # term for each data point that combines the model's precision parameter (beta),
        # the distance computed above, and the mixing coefficients. The log-likelihood
        # is the sum of these terms across all data points.
        return (np.log((self.beta / 2.0 / np.pi) ** (input_dataset.shape[1] / 2.0) *
                       (np.exp(-self.beta / 2.0 * distance) * self.mixing_coefficients).sum(axis=1))).sum()

    def mlr(self, X, y):
        """
        Train a Multiple Linear Regression (MLR) model using the given independent variables (X)
        and dependent variable (y). The model normalizes the input data, computes regression
        coefficients to predict `y` from `X`, and calculates the model's variance. This function
        assumes that both `X` and `y` are not autoscaled before being passed in.

        Parameters
        ----------
        X : numpy.array or pandas.DataFrame
            The independent variables. Each row corresponds to a single observation,
            and each column corresponds to a different variable.
        y : numpy.array or pandas.DataFrame
            The dependent variable. Each row corresponds to a single observation. It is
            expected to be a single column if passed as a DataFrame.

        Returns
        -------
        None: This method does not return a value but updates the object's state with the
              regression coefficients and model variance.
        """
        # Convert X and y to NumPy arrays, if they're not already.
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)  # Ensure y is a column vector.

        # Autoscaling (feature scaling and mean centering) of X and y.
        self.Xmean = X.mean(axis=0)
        self.Xstd = X.std(axis=0, ddof=1)  # ddof=1 for sample standard deviation.
        autoscaled_X = (X - self.Xmean) / self.Xstd
        self.y_mean = y.mean(axis=0)
        self.ystd = y.std(axis=0, ddof=1)  # ddof=1 for sample standard deviation.
        autoscaled_y = (y - self.y_mean) / self.ystd

        # Calculate regression coefficients by solving the normal equation.
        self.regression_coefficients = np.linalg.inv(autoscaled_X.T @ autoscaled_X) @ (autoscaled_X.T @ autoscaled_y)

        # Calculate predicted y values in the autoscaled space, then rescale back to the original space.
        calculated_y = (autoscaled_X @ self.regression_coefficients) * self.ystd + self.y_mean

        # Calculate model variance as the sum of squared residuals divided by the number of observations.
        self.sigma = np.sum((y - calculated_y) ** 2) / len(y)

    def mlr_predict(self, X):
        """
        Predict the output values (y) for new input data (X) using the trained Multiple Linear Regression (MLR) model.
        This method applies the regression coefficients obtained during training to the new data,
        after appropriately scaling the inputs based on the scaling parameters (mean and standard deviation)
        calculated from the training data. The predictions are then rescaled back to the original scale of the output data.

        Parameters
        ----------
        X : numpy.array or pandas.DataFrame
            The new input data for which predictions are to be made. This data should not be autoscaled
            as the method handles scaling internally using the parameters obtained from the training data.

        Returns
        -------
        predicted_y : numpy.array
            The predicted output values corresponding to the input data X. These predictions are rescaled
            to the original scale of the output data as observed during the training phase.
        """
        # Ensure X is a numpy array
        X = np.array(X)

        # Autoscale the input data using the mean and standard deviation from the training data
        autoscaled_X = (X - self.Xmean) / self.Xstd

        # Apply the regression coefficients to the autoscaled input data
        # and rescale the predictions to the original scale of the output data
        predicted_y = (autoscaled_X.dot(self.regression_coefficients) * self.ystd + self.y_mean)

        return predicted_y

    def inverse_gtm_mlr(self, target_y_value):
        """
        Predict the input values (X) from a target output value (y) using the trained GTM and MLR models.
        This method calculates the mean and mode of the estimated input values that are likely to produce
        the target output value. It also provides responsibilities (probabilities) associated with each grid
        point on the GTM map, indicating how likely each grid point is to be the origin of the target output value.

        Parameters
        ----------
        target_y_value : scalar
            The target y-value for which the corresponding input values are to be estimated.

        Returns
        -------
        estimated_x_mean : numpy.array
            The mean of the estimated input values that could produce the target y-value.
        estimated_x_mode : numpy.array
            The mode of the estimated input values that could produce the target y-value. This represents
            the most likely input value to produce the target output.
        responsibilities_inverse : numpy.array
            A vector of probabilities associated with each grid point on the GTM map, indicating the likelihood
            that the target y-value could be produced from each grid point. This can be used to discuss the
            assigned grids on the GTM map and evaluate the applicability domain of the prediction.
        """
        # Calculate the mean (myu_i) of the transformed dataset using GTM model parameters
        myu_i = self.phi_of_map_rbf_grids.dot(self.W) + np.ones(
            (np.prod(self.shape_of_map), 1)).dot(np.reshape(self.bias, (1, len(self.bias))))

        # Initialize the covariance matrix (sigma_i) and its inverse for the input space
        sigma_i = np.diag(np.ones(len(self.regression_coefficients))) / self.beta
        inverse_sigma_i = np.diag(np.ones(len(self.regression_coefficients))) * self.beta

        # Calculate delta_i, used for adjusting the mean estimation based on the target y-value
        delta_i = np.linalg.inv(inverse_sigma_i
                                + self.regression_coefficients.dot(self.regression_coefficients.T) / self.sigma)

        # Calculate means of the probability distribution for input values corresponding to the target y-value
        pxy_means = np.empty(myu_i.shape)
        for i in range(pxy_means.shape[0]):
            pxy_means[i, :] = np.ndarray.flatten(
                delta_i.dot(
                    self.regression_coefficients / self.sigma * target_y_value
                    + inverse_sigma_i.dot(np.reshape(myu_i[i, :], [myu_i.shape[1], 1]))
                ))

        # Calculate the mean and variance of y given z (latent variable) for the GTM model
        pyz_means = myu_i.dot(self.regression_coefficients)
        pyz_var = self.sigma + self.regression_coefficients.T.dot(
            sigma_i.dot(self.regression_coefficients))

        # Calculate the probability density function of y given z for each grid point
        pyzs = np.empty(len(pyz_means))
        for i in range(len(pyz_means)):
            pyzs[i] = norm.pdf(target_y_value, pyz_means[i], pyz_var ** (1 / 2))

        # Calculate responsibilities (inverse) indicating the likelihood of each grid point
        responsibilities_inverse = pyzs / pyzs.sum()

        # Estimate the mean and mode of the input values corresponding to the target y-value
        estimated_x_mean = responsibilities_inverse.dot(pxy_means)
        estimated_x_mode = pxy_means[np.argmax(responsibilities_inverse), :]

        return estimated_x_mean, estimated_x_mode, responsibilities_inverse

    def gtmr_predict(self, input_variables, numbers_of_input_variables, numbers_of_output_variables):
        """
        Predicts values of variables for both forward analysis (regression) and inverse analysis using GTM.
        Forward analysis predicts output variables based on input variables, while inverse analysis
        predicts input variables based on output variables. This method automatically adjusts its behavior
        based on the provided indices for input and output variables.

        Parameters
        ----------
        input_variables : numpy.array or pandas.DataFrame
            An m x n matrix of variables, where m is the number of samples and n is the number of variables.
            For forward analysis, these are input (X) variables; for inverse analysis, these are output (Y) variables.
            Variables should be autoscaled before being passed to this function.
        numbers_of_input_variables : list or numpy.array
            Indices specifying which columns in `input_variables` are considered as input variables.
            Determines the mode of analysis (forward or inverse).
        numbers_of_output_variables : list or numpy.array
            Indices specifying which variables are considered as output in the prediction.
            Determines the target of the prediction based on the mode of analysis.

        Returns
        -------
        estimated_y_mean : numpy.array
            An m x k matrix of the estimated output variables using a weighted mean approach,
            where k is the number of output variables.
        estimated_y_mode : numpy.array
            An m x k matrix of the estimated output variables using the mode of the weights,
            offering an alternative prediction measure.
        responsibilities : numpy.array
            An m x l matrix of weights (responsibilities) for each sample, where l is the number of latent variables.
        px : numpy.array
            An m x l matrix of the probability density of the input variables across the latent space.

        Notes
        -----
        This method requires that the GTM model has been successfully trained (indicated by `self.success_flag`).
        If the model has not been trained or the training was not successful, this method will return zeros for predictions.
        """

        # Ensure input_variables is a numpy array and has the correct shape for processing
        input_variables = np.array(input_variables)
        if input_variables.ndim == 0:
            input_variables = np.reshape(input_variables, (1, 1))
        elif input_variables.ndim == 1:
            input_variables = np.reshape(input_variables, (1, input_variables.shape[0]))

        if self.success_flag:
            # Calculate the means of the GTM map using the model's weights and bias
            means = self.phi_of_map_rbf_grids.dot(self.W) + np.ones((np.prod(self.shape_of_map), 1)).dot(
                np.reshape(self.bias, (1, len(self.bias))))
            # Extract the means for the specified input and output variables
            input_means = means[:, numbers_of_input_variables]
            output_means = means[:, numbers_of_output_variables]
            # Define the covariance matrix for the input variables
            input_covariances = np.diag(np.ones(len(numbers_of_input_variables))) / self.beta
            # Calculate the probability density for each input variable
            px = np.empty([input_variables.shape[0], input_means.shape[0]])
            for sample_number in range(input_means.shape[0]):
                px[:, sample_number] = multivariate_normal.pdf(input_variables, mean=input_means[sample_number, :],
                                                               cov=input_covariances)
            # Compute responsibilities based on the probability densities
            responsibilities = px.T / px.T.sum(axis=0)
            responsibilities = responsibilities.T
            # Predict the output variables using a weighted mean of the output means
            estimated_y_mean = responsibilities.dot(output_means)
            # Predict the output variables using the mode of the output means
            estimated_y_mode = output_means[np.argmax(responsibilities, axis=1), :]
        else:
            # If the model is not successfully trained, return zeros
            estimated_y_mean = np.zeros(input_variables.shape[0])
            estimated_y_mode = np.zeros(input_variables.shape[0])
            px = np.empty([input_variables.shape[0], np.prod(self.shape_of_map)])
            responsibilities = np.empty([input_variables.shape[0], np.prod(self.shape_of_map)])

        return estimated_y_mean, estimated_y_mode, responsibilities, px

    def gtmr_cv_opt(self, dataset, numbers_of_output_variables, candidates_of_shape_of_map,
                    candidates_of_shape_of_rbf_centers, candidates_of_variance_of_rbfs,
                    candidates_of_lambda_in_em_algorithm, fold_number, number_of_iterations):
        """
        Optimize GTMR model hyperparameters using grid search with cross-validation.

        Parameters
        ----------
        dataset : numpy.array or pandas.DataFrame
            The dataset used for the cross-validation, where rows correspond to samples and columns to variables.
        numbers_of_output_variables : list or numpy.array
            Indices of the columns in the dataset that are considered as output variables for the GTMR model.
        candidates_of_shape_of_map : list
            Candidate values for the shape of the GTM map (both dimensions).
        candidates_of_shape_of_rbf_centers : list
            Candidate values for the shape of the RBF centers grid (both dimensions).
        candidates_of_variance_of_rbfs : list
            Candidate values for the variance of the RBFS.
        candidates_of_lambda_in_em_algorithm : list
            Candidate values for the regularization parameter lambda in the EM algorithm.
        fold_number : int
            The number of folds to use in cross-validation.
        number_of_iterations : int
            The number of iterations to run for each model during the fitting process.

        Notes
        -----
        This method performs a comprehensive search over the specified ranges of hyperparameters, evaluating
        each combination using cross-validation to determine the set of hyperparameters that maximizes the
        model's performance as measured by the R-squared value.
        """

        # Initial setup
        self.display_flag = 0  # Disable display of fitting progress
        self.number_of_iterations = number_of_iterations  # Set the number of iterations for each model fitting
        dataset = np.array(dataset)  # Ensure the dataset is a numpy array
        numbers_of_output_variables = np.array(numbers_of_output_variables)  # Ensure this is a numpy array
        # Identify input variables by excluding output variable indices from the full variable set
        numbers_of_input_variables = np.delete(np.arange(dataset.shape[1]), numbers_of_output_variables)

        # Prepare cross-validation indices
        # Calculate the number of samples per fold and handle the case where the dataset size isn't a perfect multiple of fold_number
        min_number = math.floor(dataset.shape[0] / fold_number)
        mod_number = dataset.shape[0] - min_number * fold_number
        index = np.matlib.repmat(np.arange(1, fold_number + 1), 1, min_number).ravel()
        if mod_number != 0:
            index = np.r_[index, np.arange(1, mod_number + 1)]
        fold_index_in_cv = np.random.permutation(index)  # Randomly shuffle the fold indices

        # Initialize variables for grid search
        y = np.ravel(dataset[:, numbers_of_output_variables])  # Flatten the output variables for comparison
        parameters_and_r2_cv = []  # List to store hyperparameters and their corresponding R-squared values

        # Calculate the total number of configurations to evaluate
        all_calculation_numbers = len(candidates_of_shape_of_map) * len(candidates_of_shape_of_rbf_centers) * \
                                  len(candidates_of_variance_of_rbfs) * len(candidates_of_lambda_in_em_algorithm)
        calculation_number = 0  # Counter for the number of configurations evaluated

        # Grid search over hyperparameter space
        for shape_of_map_grid in candidates_of_shape_of_map:
            for shape_of_rbf_centers_grid in candidates_of_shape_of_rbf_centers:
                for variance_of_rbfs_grid in candidates_of_variance_of_rbfs:
                    for lambda_in_em_algorithm_grid in candidates_of_lambda_in_em_algorithm:
                        calculation_number += 1
                        estimated_y_in_cv = np.zeros([dataset.shape[0], len(numbers_of_output_variables)])
                        success_flag_cv = True  # Flag to track if all folds were successfully evaluated

                        # Evaluate current hyperparameter set using cross-validation
                        for fold_number_in_cv in np.arange(1, fold_number + 1):
                            # Split dataset into training and testing sets for the current fold
                            dataset_train_in_cv = dataset[fold_index_in_cv != fold_number_in_cv, :]
                            dataset_test_in_cv = dataset[fold_index_in_cv == fold_number_in_cv, :]
                            # Set the model's hyperparameters for the current configuration
                            self.shape_of_map = [shape_of_map_grid, shape_of_map_grid]
                            self.shape_of_rbf_centers = [shape_of_rbf_centers_grid, shape_of_rbf_centers_grid]
                            self.variance_of_rbfs = variance_of_rbfs_grid
                            self.lambda_in_em_algorithm = lambda_in_em_algorithm_grid
                            self.fit(dataset_train_in_cv)  # Fit the model to the training set

                            if self.success_flag:
                                # Predict using the fitted model and update the estimated values for the test set
                                estimated_y_mean, estimated_y_mode, responsibilities, px = self.gtmr_predict(
                                    dataset_test_in_cv[:, numbers_of_input_variables], numbers_of_input_variables,
                                    numbers_of_output_variables)
                                estimated_y_in_cv[fold_index_in_cv == fold_number_in_cv, :] = estimated_y_mode
                            else:
                                success_flag_cv = False
                                break  # Exit the loop if model fitting was unsuccessful

                        # Calculate R-squared value for the current hyperparameter set if all folds were successful
                        if success_flag_cv:
                            y_pred = np.ravel(estimated_y_in_cv)
                            r2_cv = float(1 - sum((y - y_pred) ** 2) / sum((y - y.mean()) ** 2))
                        else:
                            r2_cv = -10 ** 10  # Assign a large negative value to indicate failure

                        # Store the hyperparameters and their corresponding R-squared value
                        parameters_and_r2_cv.append(
                            [shape_of_map_grid, shape_of_rbf_centers_grid, variance_of_rbfs_grid,
                             lambda_in_em_algorithm_grid, r2_cv])
                        print([calculation_number, all_calculation_numbers, r2_cv])  # Optional: Print progress

        # Select the best hyperparameter set based on R-squared values
        parameters_and_r2_cv = np.array(parameters_and_r2_cv)
        optimized_hyperparameter_number = np.where(parameters_and_r2_cv[:, 4] == np.max(parameters_and_r2_cv[:, 4]))[0][
            0]
        # Update the model with the optimized hyperparameters
        self.shape_of_map = [int(parameters_and_r2_cv[optimized_hyperparameter_number, 0]),
                             int(parameters_and_r2_cv[optimized_hyperparameter_number, 0])]
        self.shape_of_rbf_centers = [int(parameters_and_r2_cv[optimized_hyperparameter_number, 1]),
                                     int(parameters_and_r2_cv[optimized_hyperparameter_number, 1])]
        self.variance_of_rbfs = parameters_and_r2_cv[optimized_hyperparameter_number, 2]
        self.lambda_in_em_algorithm = parameters_and_r2_cv[optimized_hyperparameter_number, 3]
