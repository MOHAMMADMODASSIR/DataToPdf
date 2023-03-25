from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
import base64
from flask import make_response
import pdfkit

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if not file:
        return 'No file uploaded.', 400

    # Check if file is a csv file
    if file.filename[-3:] != 'csv':
        return 'Invalid file type. Only csv files are allowed.', 400

    # Read csv file
    df = pd.read_csv(file)

    # Get data for specific column
    column_name = request.form['column']
    if not column_name:
        return 'Invalid column name.', 400
    if column_name not in df.columns:
        return 'Column not found.', 400
    data = df[column_name]

    # Choose chart type
    chart_type = request.form['chart']
    if not chart_type:
        return 'Invalid chart type.', 400
    if chart_type == 'piechart':
        counts = df['model_target'].value_counts()
        # Plot pie chart
        fig, ax = plt.subplots()
        ax.pie(counts, labels=counts.index,autopct='%1.1f%%')
        ax.set_title(column_name)
    elif chart_type == 'histogram':
        # Plot histogram
        fig, ax = plt.subplots()
        ax.hist(data)
        ax.set_title(column_name)
    elif chart_type == 'barchart':
        # Create a bar chart
        counts = df['model_target'].value_counts()
        fig, ax = plt.subplots()
        ax.bar(counts.index, counts.values)
        # ax.bar(data,height=20)
        ax.set_title(column_name)
    
    # Choose ML Model name    
    model_description=""
    specific=""
    model_name = request.form['modelname'] 
    if not model_name:
        return 'Invalid chart type.', 400
    if model_name == 'Linear Regression':
        model_description+="Logistic regression is a statistical method used to model the relationship between a binary dependent variable (i.e., a variable that takes on only two possible values, such as 0 or 1) and one or more independent variables. Unlike linear regression, which assumes a linear relationship between the dependent variable and the independent variables, logistic regression uses a logistic function to model the relationship. The logistic function produces an S-shaped curve that maps the independent variables to a probability of the dependent variable being 1. The goal of logistic regression is to find the coefficients of the independent variables that maximize the likelihood of the observed data given the model. Logistic regression is widely used in many fields, including healthcare, marketing, and social sciences, to model and predict binary outcomes."
        specific+='''Linear regression is a type of supervised learning algorithm that is commonly used for predicting a continuous output variable based on one or more input variables. However, it can also be used for classification tasks, such as cat classification.

To perform cat classification using linear regression, you would need to first prepare your dataset. You would need to gather a set of labeled images of cats, where the labels indicate whether each image is of a cat or not. You would then need to preprocess the images and extract relevant features, such as the color and texture of the fur, the shape of the ears and eyes, etc.

Once you have prepared your dataset and extracted the features, you can use linear regression to train a model that maps the input features to the binary output variable (cat or not cat). In this case, you would need to use a variant of linear regression called logistic regression, which is specifically designed for binary classification tasks.

During the training phase, the model learns a set of weights that define the linear relationship between the input features and the output variable. The weights are optimized to minimize the difference between the predicted output and the true labels in the training dataset.

Once the model is trained, you can use it to classify new images of cats by extracting the features from the image and applying the learned weights to make a prediction. The output of the model will be a probability value between 0 and 1, indicating the likelihood that the input image is of a cat. You can use a threshold value to convert the probability into a binary output (cat or not cat).

Overall, cat classification using linear regression involves the following steps:

Prepare a labeled dataset of cat images
Preprocess the images and extract relevant features
Train a logistic regression model to map the features to the binary output variable
Use the trained model to classify new images of cats'''
    elif model_name == 'Logistic Regression':
        model_description+="Logistic regression is a statistical method used to model the relationship between a binary dependent variable (i.e., a variable that takes on only two possible values, such as 0 or 1) and one or more independent variables. Unlike linear regression, which assumes a linear relationship between the dependent variable and the independent variables, logistic regression uses a logistic function to model the relationship. The logistic function produces an S-shaped curve that maps the independent variables to a probability of the dependent variable being 1. The goal of logistic regression is to find the coefficients of the independent variables that maximize the likelihood of the observed data given the model. Logistic regression is widely used in many fields, including healthcare, marketing, and social sciences, to model and predict binary outcomes."
        specific+='''Cat classification using logistic regression is a binary classification problem where the goal is to predict whether an input image contains a cat or not. Logistic regression is a popular algorithm for solving binary classification problems.

In this problem, we can represent each input image as a feature vector of pixel values, where each pixel in the image corresponds to a feature. We can then train a logistic regression model on a dataset of labeled images to learn a decision boundary that separates images containing cats from those that do not.

The logistic regression model learns a set of weights for each feature that are used to compute a weighted sum of the features. This sum is then passed through the logistic function, which maps the output to a probability score between 0 and 1. If the score is above a certain threshold (e.g., 0.5), the model predicts that the input image contains a cat; otherwise, it predicts that it does not.

To train the logistic regression model, we need a dataset of labeled images. We can use various techniques to preprocess the images, such as resizing and normalization. We can then split the dataset into training and validation sets and train the model on the training set using an optimization algorithm such as gradient descent. We can evaluate the performance of the model on the validation set and tune the hyperparameters of the model (such as learning rate, regularization strength, etc.) to improve its performance.

Once the model is trained and validated, we can use it to predict whether new input images contain cats or not. We can apply the same preprocessing steps to the new images and pass them through the trained model to obtain the predicted probability score. If the score is above the threshold, we can predict that the input image contains a cat.'''
    elif model_name == 'Decision Tree':
        model_description+="A decision tree is a machine learning algorithm for classification and regression tasks. It has a tree-like structure with root, internal, and leaf nodes. Internal nodes represent tests on features, while leaf nodes represent class labels or numerical values. Decision trees are constructed with a training dataset and rules that split the data at each internal node. They are used for decision-making and predictive modeling in various fields due to their interpretability and ease of use."
    elif model_name == 'SVM Algorithm':
        model_description+="SVM (Support Vector Machine) is a supervised learning algorithm used for classification and regression tasks. It separates data points into different classes using a hyperplane with maximum margin. SVM finds the optimal hyperplane that maximizes the distance between the classes. It is widely used in various fields, including computer vision, natural language processing, and bioinformatics"
    elif model_name == 'Naive Bayes Algorithm':
        model_description+="Naive Bayes is a probabilistic machine learning algorithm used for classification tasks. It calculates the probability of an instance belonging to a certain class based on the probabilities of its features. Naive Bayes assumes that features are independent of each other, hence the 'naive' assumption. It is widely used in various fields, including spam filtering, sentiment analysis, and document classification."
    elif model_name == 'KNN Algorithm':
        model_description+="KNN (K-Nearest Neighbors) is a machine learning algorithm used for classification and regression tasks. It works by finding the k closest data points in the training dataset to a new instance and predicting its label or value based on the majority vote or average of those k neighbors. KNN does not make any assumptions about the underlying distribution of the data, making it a non-parametric algorithm. It is widely used in various fields, including image and text classification, recommendation systems, and anomaly detection."
    elif model_name == 'K-means':
        model_description+="K-means is an unsupervised machine learning algorithm used for clustering tasks. It aims to group similar data points into k clusters based on their features. K-means works by iteratively assigning data points to their nearest cluster centroid, then updating the centroid based on the mean of the data points in the cluster. The algorithm converges when the assignment of data points to clusters no longer changes. K-means is widely used in various fields, including image segmentation, customer segmentation, and anomaly detection"
    elif model_name == 'Random Forest Algorithm':
        model_description+="Random forest is a machine learning algorithm used for classification and regression tasks. It is an ensemble of decision trees where each tree is trained on a random subset of the features and a random subset of the training dataset. Random forest makes predictions by aggregating the predictions of all the decision trees. It is widely used in various fields, including finance, healthcare, and natural language processing, due to its high accuracy and robustness to overfitting."
    elif model_name == 'Dimensionality Reduction Algorithms':
        model_description+="Dimensionality reduction is a process of reducing the number of variables or features in a dataset while preserving the most important information. It is often used to simplify the data, make it more manageable, and speed up machine learning algorithms.There are several dimensionality reduction algorithms available, some of the most popular ones are:Principal Component Analysis (PCA): It is a linear technique that identifies the most important features in a dataset and reduces them to a lower dimensional space by projecting them onto a new coordinate system.t-distributed Stochastic Neighbor Embedding (t-SNE): It is a non-linear technique that is mainly used for visualization of high-dimensional datasets. It creates a low-dimensional representation of the data that preserves local structure and reveals clusters and patterns."
    elif model_name == 'Gradient Boosting Algorithm':
        model_description+="Gradient Boosting is a machine learning algorithm that can be used for both regression and classification problems. It is an ensemble learning algorithm that combines the predictions of multiple weak learning models to create a stronger model.The Gradient Boosting algorithm is an iterative algorithm that builds models in a stage-wise manner. At each stage, a new weak model is trained on the residual errors of the previous model. The residual errors are the differences between the predicted values of the previous model and the true values.Gradient Boosting can use various weak models, such as decision trees or regression models. In each iteration, the algorithm computes the negative gradient of the loss function with respect to the predicted values of the previous model, and uses this gradient to train the next weak model. The predicted values of the weak model are then added to the predictions of the previous model to create a new set of predictions."
    
    # Convert plot to base64-encoded image
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()

    # Save plot to pdf
    pdf_buffer = BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        pdf.savefig(fig, bbox_inches='tight')

    # Convert pdf to base64-encoded file
    pdf_base64 = base64.b64encode(pdf_buffer.getvalue()).decode()

    # Render template with image and pdf
    return render_template('result.html', name=chart_type, image=image_base64, modelname=model_name, description=model_description, specific=specific, pdf=pdf_base64)


#Store the result in pdf

@app.route('/download')
def download():
    # Get parameters
    chart_type = request.args.get('name')
    image_base64 = request.args.get('image')
    pdf_base64 = request.args.get('pdf')

    # Generate HTML string
    html = render_template('result.html', name=chart_type, image=image_base64, pdf=pdf_base64)

    # Create PDF from HTML
    pdf = pdfkit.from_string(html, False)

    # Create response object
    response = make_response(pdf)

    # Set headers
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=result.pdf'

    return response




if __name__ == '__main__':
    app.run(debug=True)