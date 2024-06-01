import pandas as pd
import matplotlib.pyplot as plt
import math
from copy import deepcopy
import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, precision_recall_curve



print('1. Відкрити та зчитати файл з даними.')
data = pd.read_csv('KM_12_1.csv')

print('2. Визначити збалансованість набору даних. Вивести кількість об’єктів кожного класу.')
class_counts = data['GT'].value_counts()
print(class_counts) # Кількість об'єктів кожного класу однакова, ми можемо вважати набір даних збалансованим.


print("""Для зчитаного набору даних виконати наступні дії: 
a. Обчислити всі метрики (Accuracy, Precision, Recall, F-Scores, 
Matthews Correlation Coefficient, Balanced Accuracy, Youden’s J 
statistics, Area Under Curve for Precision-Recall Curve, Area 
Under Curve for Receiver Operation Curve) для кожної моделі при
різних значеннях порогу класифікатора (крок зміни порогу
0,3).""")

# Імена стовпців
actual_col = 'GT'  
model1_col = 'Model_1'   
model2_col = 'Model_2'   

# Функція для обчислення метрик
def compute_metrics(y_true, y_scores, threshold):
    metrics = {
        'threshold': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'mcc': [],
        'balanced_accuracy': [],
        'youden_j': [],
        'roc_auc': [],
        'pr_auc': []
    }
    
    y_pred = (y_scores >= threshold).astype(int)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    youden_j = recall + (1 - (1 - precision)) - 1  # Youden's J = Sensitivity + Specificity - 1
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    
    metrics['threshold'].append(threshold)
    metrics['accuracy'].append(accuracy)
    metrics['precision'].append(precision)
    metrics['recall'].append(recall)
    metrics['f1_score'].append(f1)
    metrics['mcc'].append(mcc)
    metrics['balanced_accuracy'].append(balanced_acc)
    metrics['youden_j'].append(youden_j)
    metrics['roc_auc'].append(roc_auc)
    metrics['pr_auc'].append(pr_auc)
    
    return pd.DataFrame(metrics)

# Визначення порогу
threshold = 0.3

# Обчислення метрик для моделі №1
metrics_model1 = compute_metrics(data[actual_col], data[model1_col], threshold)

# Обчислення метрик для моделі №2
metrics_model2 = compute_metrics(data[actual_col], data[model2_col], threshold)

# Виведення результатів
print("Metrics for Model 1:")
print(metrics_model1)
print("\nMetrics for Model 2:")
print(metrics_model2)

print(""" Збудувати на одному графіку в одній координатній системі
(величина порогу; значення метрики) графіки усіх обчислених
метрик, відмітивши певним чином максимальне значення кожної
з них. """)

# Функція для обчислення метрик
def compute_metrics(y_true, y_scores, thresholds):
    metrics = {
        'threshold': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'mcc': [],
        'balanced_accuracy': [],
        'youden_j': [],
        'roc_auc': [],
        'pr_auc': []
    }
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        youden_j = recall + (1 - (1 - precision)) - 1  # Youden's J = Sensitivity + Specificity - 1
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = roc_auc_score(y_true, y_scores)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = average_precision_score(y_true, y_scores)
        
        metrics['threshold'].append(threshold)
        metrics['accuracy'].append(accuracy)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1_score'].append(f1)
        metrics['mcc'].append(mcc)
        metrics['balanced_accuracy'].append(balanced_acc)
        metrics['youden_j'].append(youden_j)
        metrics['roc_auc'].append(roc_auc)
        metrics['pr_auc'].append(pr_auc)
    
    return pd.DataFrame(metrics)

# Визначення порогів
thresholds = np.arange(0, 1.1, 0.1)

# Обчислення метрик для моделі №1
metrics_model1 = compute_metrics(data[actual_col], data[model1_col], thresholds)

# Обчислення метрик для моделі №2
metrics_model2 = compute_metrics(data[actual_col], data[model2_col], thresholds)

# Побудова графіків
def plot_metrics(metrics, model_name):
    plt.figure(figsize=(14, 10))
    
    for metric in metrics.columns[1:]:
        plt.plot(metrics['threshold'], metrics[metric], label=f'{metric}')
        
        # Пошук максимального значення
        max_idx = metrics[metric].idxmax()
        max_threshold = metrics.loc[max_idx, 'threshold']
        max_value = metrics.loc[max_idx, metric]
        
        # Позначення максимального значення
        plt.scatter(max_threshold, max_value, s=100, label=f'Max {metric}: {max_value:.3f} at threshold {max_threshold:.1f}', zorder=5)
    
    plt.xlabel('Поріг')
    plt.ylabel('Значення метрик')
    plt.title(f'Значення метрик для різних порогів ({model_name})')
    plt.legend()
    plt.grid(True)
    plt.show()

# Побудова графіків для моделі №1
plot_metrics(metrics_model1, "Модель 1")

# Побудова графіків для моделі №2
plot_metrics(metrics_model2, "Модель 2")

print(""". Збудувати в координатах (значення оцінки класифікаторів; 
кількість об’єктів кожного класу) окремі для кожного класу
графіки кількості об’єктів та відмітити вертикальними лініями
оптимальні пороги відсічення для кожної метрики""")



# Функція для обчислення метрик
def compute_metrics(y_true, y_scores, thresholds):
    metrics = {
        'threshold': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'mcc': [],
        'balanced_accuracy': [],
        'youden_j': []
    }
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        youden_j = recall + (1 - (1 - precision)) - 1  # Youden's J = Sensitivity + Specificity - 1
        
        metrics['threshold'].append(threshold)
        metrics['accuracy'].append(accuracy)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1_score'].append(f1)
        metrics['mcc'].append(mcc)
        metrics['balanced_accuracy'].append(balanced_acc)
        metrics['youden_j'].append(youden_j)
    
    return pd.DataFrame(metrics)

# Визначення порогів
thresholds = np.arange(0, 1.1, 0.1)

# Обчислення метрик для моделі №1
metrics_model1 = compute_metrics(data[actual_col], data[model1_col], thresholds)

# Обчислення метрик для моделі №2
metrics_model2 = compute_metrics(data[actual_col], data[model2_col], thresholds)

# Функція для побудови гістограм з позначенням оптимальних порогів
def plot_histograms_with_thresholds(data, actual_col, score_col, optimal_thresholds):
    class_0_scores = data[data[actual_col] == 0][score_col]
    class_1_scores = data[data[actual_col] == 1][score_col]
    
    plt.figure(figsize=(14, 10))
    
    plt.hist(class_0_scores, bins=30, alpha=0.5, label='Class 0', color='blue')
    plt.hist(class_1_scores, bins=30, alpha=0.5, label='Class 1', color='red')
    
    for metric, threshold in optimal_thresholds.items():
        plt.axvline(x=threshold, linestyle='--', label=f'{metric}: {threshold:.2f}', linewidth=2)
    
    plt.xlabel('Score')
    plt.ylabel('Number of Objects')
    plt.title(f'Distribution of Scores for {score_col}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Знаходження оптимальних порогів для кожної метрики
optimal_thresholds_model1 = {
    'accuracy': thresholds[metrics_model1['accuracy'].idxmax()],
    'precision': thresholds[metrics_model1['precision'].idxmax()],
    'recall': thresholds[metrics_model1['recall'].idxmax()],
    'f1_score': thresholds[metrics_model1['f1_score'].idxmax()],
    'mcc': thresholds[metrics_model1['mcc'].idxmax()],
    'balanced_accuracy': thresholds[metrics_model1['balanced_accuracy'].idxmax()],
    'youden_j': thresholds[metrics_model1['youden_j'].idxmax()]
}

optimal_thresholds_model2 = {
    'accuracy': thresholds[metrics_model2['accuracy'].idxmax()],
    'precision': thresholds[metrics_model2['precision'].idxmax()],
    'recall': thresholds[metrics_model2['recall'].idxmax()],
    'f1_score': thresholds[metrics_model2['f1_score'].idxmax()],
    'mcc': thresholds[metrics_model2['mcc'].idxmax()],
    'balanced_accuracy': thresholds[metrics_model2['balanced_accuracy'].idxmax()],
    'youden_j': thresholds[metrics_model2['youden_j'].idxmax()]
}

# Побудова гістограм для моделі №1
plot_histograms_with_thresholds(data, actual_col, model1_col, optimal_thresholds_model1)

# Побудова гістограм для моделі №2
plot_histograms_with_thresholds(data, actual_col, model2_col, optimal_thresholds_model2)

print("""Збудувати для кожного класифікатору PR-криву та ROC-криву, 
показавши графічно на них значення оптимального порогу""")

# Обчислення метрик для різних порогів
thresholds = np.arange(0, 1.1, 0.1)
def compute_metrics(y_true, y_scores, thresholds):
    metrics = {
        'threshold': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'mcc': [],
        'balanced_accuracy': [],
        'youden_j': []
    }
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        youden_j = recall + (1 - (1 - precision)) - 1  # Youden's J = Sensitivity + Specificity - 1
        
        metrics['threshold'].append(threshold)
        metrics['accuracy'].append(accuracy)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1_score'].append(f1)
        metrics['mcc'].append(mcc)
        metrics['balanced_accuracy'].append(balanced_acc)
        metrics['youden_j'].append(youden_j)
    
    return pd.DataFrame(metrics)

# Обчислення метрик для моделі №1 і №2
metrics_model1 = compute_metrics(data[actual_col], data[model1_col], thresholds)
metrics_model2 = compute_metrics(data[actual_col], data[model2_col], thresholds)

# Знаходження оптимальних порогів для кожної метрики
optimal_thresholds_model1 = {
    'accuracy': thresholds[metrics_model1['accuracy'].idxmax()],
    'precision': thresholds[metrics_model1['precision'].idxmax()],
    'recall': thresholds[metrics_model1['recall'].idxmax()],
    'f1_score': thresholds[metrics_model1['f1_score'].idxmax()],
    'mcc': thresholds[metrics_model1['mcc'].idxmax()],
    'balanced_accuracy': thresholds[metrics_model1['balanced_accuracy'].idxmax()],
    'youden_j': thresholds[metrics_model1['youden_j'].idxmax()]
}

optimal_thresholds_model2 = {
    'accuracy': thresholds[metrics_model2['accuracy'].idxmax()],
    'precision': thresholds[metrics_model2['precision'].idxmax()],
    'recall': thresholds[metrics_model2['recall'].idxmax()],
    'f1_score': thresholds[metrics_model2['f1_score'].idxmax()],
    'mcc': thresholds[metrics_model2['mcc'].idxmax()],
    'balanced_accuracy': thresholds[metrics_model2['balanced_accuracy'].idxmax()],
    'youden_j': thresholds[metrics_model2['youden_j'].idxmax()]
}

# Функція для побудови PR-кривої та ROC-кривої
def plot_curves(y_true, y_scores, optimal_thresholds, model_name):
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    
    optimal_pr = {metric: precision[np.argmax(pr_thresholds >= threshold)] for metric, threshold in optimal_thresholds.items()}
    optimal_recall = {metric: recall[np.argmax(pr_thresholds >= threshold)] for metric, threshold in optimal_thresholds.items()}
    optimal_fpr = {metric: fpr[np.argmax(roc_thresholds >= threshold)] for metric, threshold in optimal_thresholds.items()}
    optimal_tpr = {metric: tpr[np.argmax(roc_thresholds >= threshold)] for metric, threshold in optimal_thresholds.items()}
    
    plt.figure(figsize=(14, 7))

    # PR-крива
    plt.subplot(1, 2, 1)
    plt.plot(recall, precision, label='PR Curve')
    for metric, threshold in optimal_thresholds.items():
        plt.scatter(optimal_recall[metric], optimal_pr[metric], label=f'{metric}: {threshold:.2f}', zorder=5)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve ({model_name})')
    plt.legend()
    plt.grid(True)

    # ROC-крива
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, label='ROC Curve')
    for metric, threshold in optimal_thresholds.items():
        plt.scatter(optimal_fpr[metric], optimal_tpr[metric], label=f'{metric}: {threshold:.2f}', zorder=5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({model_name})')
    plt.legend()
    plt.grid(True)
    
    plt.show()

# Побудова PR-кривої та ROC-кривої для моделі №1
plot_curves(data[actual_col], data[model1_col], optimal_thresholds_model1, "Model 1")

# Побудова PR-кривої та ROC-кривої для моделі №2
plot_curves(data[actual_col], data[model2_col], optimal_thresholds_model2, "Model 2")

print("""Створити новий набір даних, прибравши з початкового набору 
(50 + 10К)% об’єктів класу 1, вибраних випадковим чином. Параметр К
представляє собою залишок від ділення місяця народження студента на
чотири та має визначатися в програмі на основі дати народження
студента, яка задана в програмі у вигляді текстової змінної формату
‘DD-MM’.""")
def calculate_K(birthdate):
    day, month = birthdate.split('-')
    K = (int(month) % 4)
    return K

birthdate = '23-08'
K = calculate_K(birthdate)
print("K =", K)

class_1_count = data['GT'].sum()
percent_remove = 50 + 10 * K
objects_remove = int(class_1_count * percent_remove / 100)
class_1_indices = data[data['GT'] == 1].index
objects_remove_indices = np.random.choice(class_1_indices, size=objects_remove, replace=False)
new_data = data.drop(objects_remove_indices)

print('6. Вивести відсоток видалених об’єктів класу 1 та кількість елементів кожного класу після видалення.')

print("Кількість об'єктів класу 1 в початковому наборі даних:", class_1_count)
print("Кількість об'єктів класу 0 в початковому наборі даних:", len(data) - class_1_count)
print(f"Відсоток видалених об'єктів класу 1: {percent_remove}%")

new_class_1_count = new_data['GT'].sum()
print("Кількість об'єктів класу 1 після видалення:", new_class_1_count)
print("Кількість об'єктів класу 0 після видалення:", len(new_data) - new_class_1_count)

new_data.to_csv (r'my_data.csv', index= False )