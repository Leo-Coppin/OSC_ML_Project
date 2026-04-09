import shap
import matplotlib.pyplot as plt
from Gradient_Boosting_model import *

model_rdkit, pred_rdkit, results_rdkit = gradient_boosting(
    X_train_RDKit, Y_train, X_test_RDKit, Y_test
)

explainer = shap.TreeExplainer(model_rdkit)

shap_values = explainer(X_test_RDKit)

# print(type(shap_values))
# print(shap_values.values.shape)

target_names = ['Voc', 'Jsc', 'FF', 'PCE', 'delta_HOMO', 'delta_LUMO']
target_idx = 3   # PCE

pce_shap = shap.Explanation(
    values=shap_values.values[:, :, target_idx],
    base_values=shap_values.base_values[:, target_idx],
    data=X_test_RDKit.values,
    feature_names=X_test_RDKit.columns.tolist()
)

plt.title("PCE")
shap.summary_plot(pce_shap.values, X_test_RDKit)
plt.title("PCE")
shap.plots.bar(pce_shap)
plt.show()

plt.title("PCE")
shap.plots.waterfall(pce_shap[0])
plt.show()