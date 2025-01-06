# SmartPlan: Enhancing Outpatient Appointment Scheduling  

**SmartPlan** is a data-driven solution to address uncertainties in outpatient appointment scheduling by predicting no-show probabilities and service time variability. This project leverages machine learning models and a large-scale dataset to optimise resource utilisation and improve patient satisfaction.  

---

## Description  📜 

This study focuses on addressing a critical challenge in healthcare: improving the reliability and efficiency of outpatient appointment scheduling. Using a dataset from primary and specialised care for American veterans, the project applies the **CRISP-DM** methodology to predict:  
- **No-show probabilities**  
- **Service time variability**  

Four machine learning models were evaluated for classification and regression tasks:  
- Random Forest (RF)  
- Extreme Gradient Boosting (XGBoost)  
- Adaptive Boosting (AdaBoost)  
- Artificial Neural Networks (ANN)  

The dataset, described by Feyman et al. (2021) ([article link](https://www.sciencedirect.com/science/article/pii/S2352340921004182)), includes anonymised data from over 51 million consultations between January 2014 and August 2023. The data stems from the Veterans Health Administration (VHA) and Community Medical Center (CC) Facility Wait Time Study, which emerged after the 2014 Veterans Choice Program (VCP) reforms.  
- **Dataset link:** [Mendeley Data Repository](https://data.mendeley.com/datasets/rmk89k4rhb/16)  

---

## Installation  ⚙️ 

To get started, clone the repository and follow these steps:  

1. Clone the repository to your local machine.  
2. Download the dataset from the [Mendeley Data Repository](https://data.mendeley.com/datasets/rmk89k4rhb/16).  
3. Place the downloaded CSV file in the `data` directory and rename it to `consult_waits_2024_03_25.csv`.  
4. You can also modify the value of the `path` variable in the `data_exploration.ipynb` file to specify a custom path for the dataset.  

   - **Data Exploration:**  
     - `data_exploration.ipynb` contains the exploration of no-show and service time predictions.  
     - Place `utils_exploration_optimised.py` and `type_stop_code.pdf` in the same directory. These files optimise image sizes and provide supplementary data for primary and secondary care types.  
   - **Prediction Models:**  
     - `Prediction_No_Show.ipynb` covers no-show prediction models and their evaluation. Place `utils_ns.py` in the same directory.  
     - `Prediction_Service_Times_.ipynb` and `Prediction_Service_Times_complement.ipynb` address service time predictions and their evaluation. Place `utils_st.py` in the same directory.  

---

##  Usage  🚀

Outpatient clinics can leverage the findings of this study to:  
- Improve predictive accuracy of no-shows and service times.
- as inspiration for custom designs
- improve their current practices by exploiting the results obtained, such as the key characteristics identified
  

---

##  Contributing  🤝

We welcome contributions to SmartPlan! To contribute:  
1. Fork the repository.  
2. Create a new branch (`feature/your-feature-name`).  
3. Commit your changes and push them to your forked repository.  
4. Open a pull request, detailing the changes and their purpose.  

---




 
