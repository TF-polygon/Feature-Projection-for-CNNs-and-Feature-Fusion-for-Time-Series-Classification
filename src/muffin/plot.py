from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

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
            plt.plot(data_list[(i * 4) + j], label=f'$W$={24 * (j + 1)}')
        
        plt.xlabel('$k$', fontsize=15)
        if i % 2 == 0:
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
        plt.subplots_adjust(hspace=0.3)
        plt.legend(loc='best', fontsize=15)
        plt.grid()
    
    # plt.tight_layout()
    plt.show()

def visualize_cluster_distribution(npz):
    data = np.load(npz)
    scaled_data = data['scaled_data']
    n_clusters = int(data['n_clusters'])
    labels = data['labels']
    centers = data['centers']

    width = 10
    if int(n_clusters) == 3:
        width = 15
    elif int(n_clusters) == 4:
        width = 20

    plt.figure(figsize=(width, 4))

    for i in range(n_clusters):
        cluster_indices = np.where(labels == i)[0]
        cluster_data = scaled_data[cluster_indices]
        cluster_center = centers[i].flatten() 

        distances = np.linalg.norm(cluster_data - cluster_center, axis=1)

        num_to_sample = min(15, len(cluster_data))
        closest_indices = np.argsort(distances)[:num_to_sample]
        sample_data = cluster_data[closest_indices]

        plt.subplot(1, n_clusters, i + 1)

        if sample_data.size > 0:
            plt.plot(sample_data.T, c='dimgray', alpha=0.3, linewidth=4)        
            plt.plot(cluster_center.T, c='red', linewidth=2, label='Center')
            
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlabel('Time Step')
        plt.title(f"Cluster {i}", fontsize=12)

    plt.show()
    
def visualize_tSNE(npz):
    data = np.load(npz)
    scaled_data = data['scaled_data']
    cluster_labels = data['labels']
    n_series = scaled_data.shape[0]
    X_tsne_input = scaled_data.reshape(n_series, -1)

    tsne = TSNE(n_components=2, perplexity=30, random_state=123)
    X_tsne = tsne.fit_transform(X_tsne_input)

    tsne_df = pd.DataFrame(X_tsne, columns=['TSNE Component 1', 'TSNE Component 2'])
    tsne_df['Cluster'] = cluster_labels
    tsne_df['Cluster'] = tsne_df['Cluster'].astype('category')

    plt.figure(figsize=(17, 12))
    sns.scatterplot(
        x='TSNE Component 1',
        y='TSNE Component 2',
        hue='Cluster',
        palette='binary',
        data=tsne_df,
        legend=False,
        alpha=0.5,
        s=450
    )
    # plt.title('t-SNE Visualization of Time Series Clusters')
    plt.xlabel('TSNE Component 1', fontsize=22)
    plt.xticks(fontsize=18)
    plt.ylabel('TSNE Component 2', fontsize=22)
    plt.yticks(fontsize=18)
    plt.grid(False)
    # plt.legend(fontsize=25)
    plt.show()    

def visualize_pca(npz):
    data = np.load(npz)
    scaled_data = data['scaled_data']
    cluster_labels = data['labels']
    
    n_series = scaled_data.shape[0]
    X_pca_input = scaled_data.reshape(n_series, -1)

    pca = PCA(n_components=2, random_state=123)
    X_pca = pca.fit_transform(X_pca_input)

    pca_df = pd.DataFrame(X_pca, columns=['PCA Component 1', 'PCA Component 2'])
    pca_df['Cluster'] = cluster_labels
    pca_df['Cluster'] = pca_df['Cluster'].astype('category')

    plt.figure(figsize=(17, 12))
    sns.scatterplot(
        x='PCA Component 1',
        y='PCA Component 2',
        hue='Cluster',
        palette='hls',
        data=pca_df,
        legend=True,
        alpha=0.6,
        s=450,
        edgecolor='w',
        linewidth=0.5
    )

    var_exp = pca.explained_variance_ratio_ * 100
    plt.xlabel(f'PCA Component 1 ({var_exp[0]:.2f}%)', fontsize=22)
    plt.ylabel(f'PCA Component 2 ({var_exp[1]:.2f}%)', fontsize=22)
    
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.show()

def visualize_confusion_matrix(labels_path, preds_path):
    y_true = np.load(labels_path)
    y_prob = np.load(preds_path)

    conf_matrix = confusion_matrix(y_true, y_prob)

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label', fontsize=15)
    plt.ylabel('True Label', fontsize=15)


def visualize_learning_curve(train_path, valid_path):
    train = pd.read_csv(train_path)
    valid = pd.read_csv(valid_path)
    
    train_acc = train['Train Accuracy']
    train_loss = train['Train Loss']

    valid_acc = valid['Val Accuracy']
    valid_loss = valid['Val Loss']

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