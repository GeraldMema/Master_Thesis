# Master Thesis
During the last decades, in the area of machine learning and data mining, the development of
ensemble methods has gained significant attention from the scientific community. Machine
learning ensemble methods combine multiple learning algorithms to obtain better predictive performance than could be obtained from any of the constituent learning algorithms alone. One of the
most challenging tasks in ensemble models is to create classifiers that are accurate and diverse. This
optimization problem can be addressed by using evolutionary learning algorithms. In this thesis, we
have developed an Evolutionary Ensemble Classification (EVENC) algorithm which approaches the
ensemble construction by evolving a population of accurate and diverse classifiers. The EVENC was
evaluated on over 100 classification datasets and compared with the most popular ensemble models,
such as Random Forest, Gradient Boosting, and XGBoost. The experiments show that our model
outperforms competing models in some datasets

# Execution
python ec_des.py

# project parameters
config.yml

Requirements
numpy~=1.19.0
sklearn~=0.0
scikit-learn~=0.23.1
pandas~=1.0.4
PyYAML~=5.3.1
scipy~=1.4.1
matplotlib~=3.2.1
xgboost~=1.1.1
plmb~=1.0.1


