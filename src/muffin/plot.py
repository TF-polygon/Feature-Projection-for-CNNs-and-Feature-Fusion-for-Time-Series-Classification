from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def get_data(path1, path2, path3, path4):
    data1 = pd.read_csv(path1)
    data2 = pd.read_csv(path2)
    data3 = pd.read_csv(path3)
    data4 = pd.read_csv(path4)

    return data1, data2, data3, data4

def visualize_elbow_method(p_e24, p_e48, p_e72, p_e96, p_g24, p_g48, p_g72, p_g96, p_c24, p_c48, p_c72, p_c96, p_j24, p_j48, p_j72, p_j96, metric='inertia'):    
    xticks = [2, 4, 6, 8, 10, 12, 14]
    xaxis = [0, 2, 4, 6, 8, 10, 12]
    k = list(range(2, 16))

    eurusd24, eurusd48, eurusd72, eurusd96 = get_data(p_e24, p_e48, p_e72, p_e96)
    gbpusd24, gbpusd48, gbpusd72, gbpusd96 = get_data(p_g24, p_g48, p_g72, p_g96)
    usdcad24, usdcad48, usdcad72, usdcad96 = get_data(p_c24, p_c48, p_c72, p_c96)
    usdjpy24, usdjpy48, usdjpy72, usdjpy96 = get_data(p_j24, p_j48, p_j72, p_j96)

    currencies = [
        eurusd24, eurusd48, eurusd72, eurusd96, 
        gbpusd24, gbpusd48, gbpusd72, gbpusd96, 
        usdcad24, usdcad48, usdcad72, usdcad96, 
        usdjpy24, usdjpy48, usdjpy72, usdjpy96
    ]
    
    data_list = []
    pairs = ['EUR/USD', 'GBP/USD', 'USD/CAD', 'USD/JPY']

    if metric == 'inertia':
        data_list = [c['Inertia'] for c in currencies]

    elif metric == 'chi':
        data_list = [c['CHI'] for c in currencies]

    elif metric == 'dbi':
        data_list = [c['DBI'] for c in currencies]

    elif metric == 'sc':
        data_list = [c['SC'] for c in currencies]

    plt.figure(figsize=(14, 11))

    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.title(pairs[i], fontsize=18)
        for j in range(4):
            plt.plot(data_list[(i * 4) + j], label=f'$W={24 * (j + 1)}')
        
        plt.xlabel('$k$', fontsize=15)
        if (i + 1) % 2 == 0:
            if metric == 'inertia':
                plt.ylabel('Inertia', fontsize=15)
            elif metric == 'sc':
                plt.ylabel('Silhouette Coefficient Score (SC)', fontsize=15)
            elif metric == 'chi':
                plt.ylabel('Calinski and Harabasz Index(CHI)', fontsize=15)
            elif metric == 'dbi':
                plt.ylabel('Davies Bouldin Index (DBI)', fontsize=15)

        plt.xticks(xaxis, xticks, fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(loc='best', fontsize=15)
        plt.grid()
    
    plt.tight_layout()
    plt.show()

def visualize_distribution_of_cluster(npz):
    data = np.load(npz)
    scaled_data = data['scaled_data']
    n_clusters = data['n_clusters']
    labels = data['labels']
    centers = data['centers']

    plt.figure(figsize=(100, 40))

    for i in range(n_clusters):
        cluster_indices = np.where(labels == i)[0]
        cluster_data = scaled_data[cluster_indices]

        num_to_sample = max(750, len(cluster_data))

        sample_indices = np.random.choice(
            len(cluster_data),
            size=num_to_sample,
            replace=False
        )

        sample_data = cluster_data[sample_indices]

        plt.subplot(1, n_clusters, i + 1)
        
        plt.plot(np.squeeze(sample_data, -1).T, c='black', alpha=0.015)
        plt.plot(np.squeeze(centers, -1)[i], c='red', linewidth=20)

        plt.xticks(fontsize=100)
        plt.yticks(fontsize=100)
        plt.xlabel('$k$')
        plt.title(f"Cluster {i}", fontsize=120)

    plt.show()

def visualize_confusion_matrix(labels_path, preds_path):
    y_true = np.load(labels_path)
    y_prob = np.load(preds_path)

    conf_matrix = confusion_matrix(y_true, y_prob)

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label', fontsize=15)
    plt.ylabel('True Label', fontsize=15)


def visualize_learning_curve(data_path):
    df = pd.read_csv(data_path)
    
    train_acc = df['Train Accuracy']
    train_loss = df['Train Loss']

    valid_acc = df['Valid Accuracy']
    valid_loss = df['Valid Loss']

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_acc, label='Training', linewidth=2)#, color='black', linestyle='-')
    plt.plot(valid_acc, label='Validation', linewidth=2)#, color='black', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')

    plt.subplot(1, 2, 2)
    plt.plot(train_loss, label='Training', linewidth=2)#, color='black', linestyle='-')
    plt.plot(valid_loss, label='Validation', linewidth=2)#, color='black', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')

if __name__ == '__main__':
    pass