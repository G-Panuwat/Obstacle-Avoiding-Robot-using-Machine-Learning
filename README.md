Obstacle Avoiding Robot using Machine Learning
🎯 Purpose
This repository contains a robust machine learning pipeline designed to be embedded in a robot for real-time obstacle avoidance. The primary goal is to allow a robot to safely and autonomously navigate its environment by detecting and avoiding physical barriers like walls.

Beyond basic navigation, this system is highly suitable for integration into more complex robotics projects, such as an autonomous AI companion rover, by providing the foundational spatial awareness needed to maneuver smoothly through dynamic spaces.

🚀 Features
The codebase provides an end-to-end data processing and model training workflow:

Intelligent Feature Engineering: Transforms raw sensor data into highly predictive spatial metrics, including open-space approximations (cross-products), relative openness (ratios), and lateral steering bias (differences).

Class Imbalance Handling: Implements Synthetic Minority Over-sampling Technique (SMOTE) to ensure the robot learns how to turn left and right just as effectively as it learns to drive forward.

Dimensionality Reduction: Utilizes Principal Component Analysis (PCA) to evaluate performance in a reduced feature space.

Bayesian Optimization: Automates the hyperparameter tuning process using skopt (BayesSearchCV) to find the most accurate model configurations.

Model Comparison: Trains and evaluates both Random Forest and K-Nearest Neighbors (KNN) classifiers across multiple data processing variants (Original vs. PCA vs. SMOTE).

📊 Dataset Requirements
The pipeline requires a local CSV file named rover_navigation_dataset.csv.

The dataset should contain the following columns:

sensor_left: Distance reading from the left sensor.

sensor_front: Distance reading from the front sensor.

sensor_right: Distance reading from the right sensor.

action: The target directional label where 0 = forward, 1 = left, and 2 = right.

(Note: The data cleaning step automatically removes any rows with a 0 sensor reading to prevent ratio calculation errors and filter out potential sensor malfunctions.)

🛠️ Setup & Installation
Ensure you have Python 3.x installed. You will need the standard scientific computing stack, specialized machine learning libraries used in the notebook, and Pygame for robotic control/simulation integrations.

You can install all required dependencies using pip:

Bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn scikit-optimize scipy pygame
Library Breakdown:

numpy / pandas: Numerical operations and data manipulation.

matplotlib / seaborn: Exploratory data analysis and visualization.

scikit-learn / scikit-optimize: Preprocessing pipelines, model building, and Bayesian hyperparameter search.

imbalanced-learn: SMOTE oversampling.

pygame: Integrated for local simulation environments, keyboard overrides, or graphical robot feedback displays.

💻 Usage
Clone this repository to your local machine.

Place your collected rover_navigation_dataset.csv in the root directory.

Run all cells in the Jupyter Notebook to execute the pipeline.

At the end of the notebook, a summary table will be generated comparing the Accuracy, Precision, Recall, F1 Score, and AUC of all 8 model variants.

Export the best-performing model (e.g., via joblib or pickle) to embed it into your robot's main execution loop.
