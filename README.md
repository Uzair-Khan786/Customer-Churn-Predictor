**📉 Customer Churn Prediction**

Predicting customer churn with thoughtful feature selection, feature engineering, exploratory data analysis, and machine learning.

A machine learning project built using the Telecom Customer Churn dataset from Kaggle, focused on identifying customers who are likely to leave a telecom service.

The goal wasn't simply to maximize a metric—it was to build a model with meaningful features, strong generalization, and useful churn detection performance.

🚀**Project Highlights**
🧹 Heavy feature selection to remove redundant and low-value variables
🧠 Feature engineering to create business-relevant churn indicators
📊 In-depth EDA to understand customer behavior and churn patterns
🤖 Multiple machine learning models trained and evaluated
⚙️ Hyperparameter tuning performed on the two strongest models
🔍 Compared tuned models against their default counterparts
🎯 Selected the best default model because tuning produced only marginal improvement while showing signs of slight underfitting
📈 Achieved 0.868 ROC-AUC and 0.704 PR-AUC
⚖️ Training and testing performance are almost identical, indicating strong generalization
📂 **Dataset**

Dataset: Telecom Customer Churn
Source: Kaggle

The dataset contains customer-level information related to:

Customer tenure
Contract and payment information
Monthly charges
Internet and streaming services
Technical support
Senior citizen status
Churn behavior

The target variable is:

Churn

🧹 **Feature Selection**

Instead of feeding every available feature into the models, I performed extensive feature selection to reduce redundancy and retain the information that actually contributes to prediction.

❌ **Removed Features**

The following features were removed:

gender
Dependents
Partner
PhoneService
customerID
DeviceProtection


These variables were considered unnecessary for the final modeling pipeline because they provided limited additional predictive value.

🔗 **Removed Highly Correlated Features**

I also removed features that were providing redundant information.

*Total_charges*

Total_charges was highly correlated with Monthly_charges, so it was removed to reduce redundancy.

*Streaming_Movies*

Streaming_Movies and Streaming_TV were carrying highly similar information. Since both were not providing meaningful new information to the model, Streaming_Movies was removed.

🎯 **Result**

The final feature set was deliberately kept more compact and informative rather than blindly retaining every available column.

*Principle: More features ≠ better model.*

🛠️ **Feature Engineering**

One of the most important parts of the project was transforming raw variables into features that could represent customer churn risk more meaningfully.

1. tenure_group

Customer tenure was transformed into groups to capture different stages of the customer lifecycle.

This helps the model distinguish between:

New customers
Early-stage customers
Long-term customers

This is particularly useful because churn behavior often changes substantially with customer tenure.

2. high_risk

A custom high-risk indicator was created using two churn-related conditions:

df['high_risk'] = (
    (df['tenure'] <= 6) &
    (df['TechSupport'] == 'No')
).astype(int)

💡 **Why?**

Customers with very low tenure and no technical support can represent a particularly vulnerable segment.

The feature converts this interaction into a simple signal:

1 → High-risk combination
0 → Otherwise


This allows the model to capture an interaction that may otherwise be harder to learn from the individual variables alone.

3. Senior_Pressure

Another engineered feature combines monthly charges with senior-citizen status:

Senior_Pressure = Monthly_charges × SeniorCitizen


This feature captures the idea that higher monthly charges may have a different implication for senior customers than for non-senior customers.

🔎**Exploratory Data Analysis**

Extensive EDA was performed before final model selection.

The analysis focused on understanding:

📌 Churn distribution
📌 Customer tenure
📌 Monthly charges
📌 Contract behavior
📌 Service usage
📌 Technical support
📌 Customer segments
📌 Relationships between features and churn
📌 Potentially redundant variables
📌 Patterns that could be converted into engineered features

The EDA was not treated as a purely visual step—it directly influenced feature selection and feature engineering.

🤖 **Modeling**

Multiple machine learning models were trained and evaluated.

After comparing the models, the two best-performing models were selected for hyperparameter tuning.

⚙️ **Hyperparameter Tuning**

The two strongest models were tuned to determine whether their performance could be improved further.

However, an interesting result emerged:

The best tuned model showed slight underfitting and improved only very marginally over the corresponding default model.

Rather than automatically choosing the tuned model simply because it had been optimized, I compared the results and selected the best-performing default model.

🧠 *Why keep the default model?*

Because the objective is not:

"Have the most complicated model."

It is:

"Have a model that generalizes well and performs reliably on unseen data."

The default model offered essentially the same—and slightly better—generalization performance without introducing unnecessary complexity.

**📊 Final Model Performance**
**🏆 Selected Model
Metric	Score
🏋️ Training Score	0.825
🧪 Testing Score	0.824
🎯 Precision	0.67
🔍 Recall	0.72
⚖️ F1 Score	0.69
📈 PR-AUC	0.704
🚀 ROC-AUC	0.868
Classification Report
Precision : 0.67
Recall    : 0.72
F1 Score  : 0.69**

**📈 Understanding the Results
Training vs Testing
Training : 0.825
Testing  : 0.824**


The extremely small difference between training and testing scores is encouraging.

Gap = 0.825 - 0.824
    = 0.001


A 0.001 gap suggests that the model is not heavily overfitting to the training data.

🎯 Recall — 0.72

The model correctly identifies approximately 72% of actual churners.

For a churn prediction problem, recall is particularly important because missing a customer who is about to churn can represent a lost retention opportunity.

⚖️ F1 Score — 0.69

The F1 score provides a balance between precision and recall.

With:

Precision = 0.67
Recall    = 0.72
F1        = 0.69


the model maintains a reasonable trade-off between finding churners and avoiding excessive false positives.

📈 PR-AUC — 0.704

A PR-AUC of 0.704 indicates useful performance in identifying the positive/churn class while accounting for the precision-recall trade-off.

This metric is especially informative for churn problems where the classes may not be perfectly balanced.

🚀 ROC-AUC — 0.868

The ROC-AUC of 0.868 demonstrates strong ranking/discrimination ability.

In practical terms, the model is quite effective at distinguishing customers who are more likely to churn from those who are less likely to churn.

🧪**Model Selection Philosophy**

One of the key lessons from this project was:

Hyperparameter tuning ≠ guaranteed improvement


The tuning process produced a model that was slightly underfitting and only marginally better than the default configuration.

Therefore, instead of selecting the tuned model automatically, I evaluated:

Training performance
Testing performance
Generalization gap
Precision
Recall
F1 score
PR-AUC
ROC-AUC

The final decision favored the best default model.

The simplest model that generalizes well is often more valuable than a tuned model that offers negligible improvement.

🔄 **Project Workflow**
                ┌──────────────────────┐
                │   Kaggle Telecom     │
                │       Dataset        │
                └──────────┬───────────┘
                           ↓
                ┌──────────────────────┐
                │   Data Cleaning &    │
                │   Preprocessing      │
                └──────────┬───────────┘
                           ↓
                ┌──────────────────────┐
                │   Feature Selection  │
                │ Remove redundancy &  │
                │ low-value features   │
                └──────────┬───────────┘
                           ↓
                ┌──────────────────────┐
                │ Feature Engineering  │
                │ tenure_group          │
                │ high_risk             │
                │ Senior_Pressure       │
                └──────────┬───────────┘
                           ↓
                ┌──────────────────────┐
                │       EDA            │
                │ Behavioral & churn   │
                │ pattern analysis     │
                └──────────┬───────────┘
                           ↓
                ┌──────────────────────┐
                │  Model Training      │
                └──────────┬───────────┘
                           ↓
                ┌──────────────────────┐
                │ Model Comparison     │
                └──────────┬───────────┘
                           ↓
                ┌──────────────────────┐
                │ Hyperparameter       │
                │ Tuning (Top 2)       │
                └──────────┬───────────┘
                           ↓
                ┌──────────────────────┐
                │ Final Model Selection│
                │ Best Default Model   │
                └──────────┬───────────┘
                           ↓
                ┌──────────────────────┐
                │ Churn Prediction     │
                └──────────────────────┘

💡 **Key Takeaways**
1. Feature quality matters more than feature quantity

Removing redundant features helped create a cleaner modeling problem.

2. Domain-inspired features can add value

Features such as high_risk and Senior_Pressure attempt to encode meaningful customer behavior rather than relying entirely on raw variables.

3. Tuning should be validated, not blindly trusted

The tuned model did not provide a meaningful enough improvement to justify replacing the default model.

4. Generalization matters

The final model achieved:

Train → 82.5%
Test  → 82.4%


with only a 0.1 percentage-point difference.

5. Churn prediction is not just about accuracy

Precision, recall, F1, PR-AUC, and ROC-AUC provide a much more complete picture of model performance.

📌 **Final Results at a Glance**
╔══════════════════════════════════════╗
║       CUSTOMER CHURN MODEL           ║
╠══════════════════════════════════════╣
║ Training Score       →   0.825       ║
║ Testing Score        →   0.824       ║
║ Precision            →   0.670       ║
║ Recall               →   0.720       ║
║ F1 Score             →   0.690       ║
║ PR-AUC               →   0.704       ║
║ ROC-AUC              →   0.868       ║
╚══════════════════════════════════════╝

🏁 **Conclusion**

This project demonstrates an end-to-end approach to customer churn prediction, from aggressive feature selection and domain-driven feature engineering to EDA, model comparison, hyperparameter tuning, and final model selection.

The most important takeaway wasn't simply achieving a high score—it was understanding why the model performed the way it did and choosing a model based on generalization and practical performance rather than complexity.

Build thoughtfully. Validate rigorously. Keep what actually works.

⭐ If you found this project interesting

Feel free to explore the notebook, experiment with different models, improve the feature engineering, or build a deployment layer around the final churn prediction model.

**Developed by Uzair - Computer Science Student and Data Science Enthusiast.**
