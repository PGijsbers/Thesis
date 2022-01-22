import numpy
import pandas
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def make_dataset(points = 1500, splits=[0.5, 0.2, 0.5, 0.8, 0.5], lower_class=[0, 1, 1, 1, 1]) -> numpy.ndarray:
    """ Creates a binary classification problem with one continuous dependent variable and one discrete dependent variable.

    For each discrete value, the classes are perfectly separable along the continuous axis.

    `points`: the number of observations in the dataset
    `splits`: for each value in the discrete axis, the ratio which should belong to "lower class"
    `lower_class`: the label of the "lower class" for each split

    returns a x (points, 2) and y (points,) arrays
    """
    dataset = numpy.zeros((points, 3))
    points_per_split = int(points / len(splits))
    upper_class = [1 if lower == 0 else 0 for lower in lower_class]
    
    for level, split in enumerate(splits):
        level_points = numpy.random.normal(loc=1, scale=0.2, size=points_per_split)
        sorted_points = list(sorted(level_points))
        
        start = int(level * points_per_split)
        end = int((level + 1) * points_per_split)
        split_point = start + int(len(sorted_points) * split)
        
        dataset[start:end, 0] = sorted_points
        dataset[start:end, 1] = level
        dataset[start:split_point, 2] = lower_class[level]
        dataset[split_point:end, 2] = upper_class[level]
    return dataset[:, :2],dataset[:,-1] 

def plot_decision_surface(estimator, X, y, ax, granularity=100j, with_contour=True):
    x_min, x_max = 0, 2
    y_min, y_max = -0.2, 4.2

    XX, YY = numpy.mgrid[x_min:x_max:granularity, y_min:y_max:granularity]
    # decision function, proba or predict
#     if hasattr(estimator, "decision_function"):
#         Z = estimator.decision_function(numpy.c_[XX.ravel(), YY.ravel()])
#         counter_kwargs = dict(levels=[-1, 0, 1], linestyles=['--', '-', '--'], colors=['k'] * 3)
#         colormesh_kwargs = dict(vmin=-1, vmax=1)
#         print("svm:", Z[:5])
#     el
    if hasattr(estimator, "predict"):
        Z = estimator.predict(numpy.c_[XX.ravel(), YY.ravel()])
        contour_kwargs = dict(levels=[0.5], linestyles=['-'], colors=['k'])
        colormesh_kwargs = dict(vmin=0, vmax=1)

    Z = Z.reshape(XX.shape)
    if with_contour:
        ax.contour(XX, YY, Z, **contour_kwargs)
    ax.pcolormesh(XX, YY, Z, cmap=plt.cm.bwr, alpha=0.1, **colormesh_kwargs)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    #ax.set_xticks()
    ax.set_xticks([0.0, 1.0, 2.0])
    ax.set_yticks([0, 1, 2, 3, 4])
    return ax
    

# Adapted from the ML engineering course
def plot_svm_surface(gamma, X, y, ax, title="t", dual_coef=None):
    """
    Visualizes the SVM model given the various outputs. It plots:
    * All the data point, color coded by class: blue or red
    * The support vectors, indicated by circling the points with a black border. 
      If the dual coefficients are known (only for kernel SVMs) if paints support vectors with high coefficients darker
    * The decision function as a blue-to-red gradient. It is white where the decision function is near 0.
    * The decision boundary as a full line, and the SVM margins (-1 and +1 values) as a dashed line
    
    Parameters:
    pipeline -- a scikit-learn pipeline with an SVC as the last step
    X -- The training data
    y -- The correct labels
    title -- The plot title
    dual_coef -- The dual coefficients of all the support vectors (not relevant for LinearSVM)
    show -- whether to plot the figure already or not
    """
    svm = SVC(gamma=gamma)
    svm.fit(X, y)
    support_vector_indices = numpy.where((2 * y - 1) * svm.decision_function(X) <= 1)[0]
    support_vectors = X[support_vector_indices]
    # plot the line, the points, and the nearest vectors to the plane
    #plt.figure(fignum, figsize=(5, 5))
    ax.set_title(title, fontsize=30)
    ax.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.bwr, marker='.')
    if dual_coef is not None:
        ax.scatter(support_vectors[:, 0], support_vectors[:, 1], c=dual_coef[0, :],
                    s=70, edgecolors='k', zorder=10, marker='.', cmap=plt.cm.bwr)
    else:
        ax.scatter(support_vectors[:, 0], support_vectors[:, 1], facecolors='none',
                    s=70, edgecolors='k', zorder=10, marker='.', cmap=plt.cm.bwr)
    
    return plot_decision_surface(svm, X, y, ax)

# Adapted from https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html#sphx-glr-auto-examples-tree-plot-iris-dtc-py

def plot_estimator_surface(estimator, X, y, ax, title, granularity=100j, with_contour=True):
    # Parameters
    plot_colors = "br"
    plot_step = 0.02
    ax.set_title(title, fontsize=30)

    # Plot the training points
    for i, color in zip(range(len(set(y))), plot_colors):
        idx = numpy.where(y == i)
        ax.scatter(
            X[idx, 0],
            X[idx, 1],
            c=color,
            #label=iris.target_names[i],
            cmap=plt.cm.bwr,
            s=15,
        )
        
    # Train
    clf = estimator.fit(X, y)
    plot_decision_surface(clf, X, y, ax, granularity, with_contour)
    return clf