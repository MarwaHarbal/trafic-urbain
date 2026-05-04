import matplotlib.pyplot as plt
import os

def plot_learning_curve(rewards_history, window=100, save_path=None):
    plt.figure(figsize=(10, 6))
    
    # Calcul de la moyenne glissante
    moving_avg = []
    for i in range(len(rewards_history)):
        start = max(0, i - window)
        moving_avg.append(sum(rewards_history[start:i+1]) / (i - start + 1))
        
    plt.plot(rewards_history, alpha=0.3, color='blue', label='Récompense par épisode')
    plt.plot(moving_avg, color='red', label=f'Moyenne glissante (fenêtre={window})')
    plt.title('Courbe d\'apprentissage du Q-Learning')
    plt.xlabel('Épisode')
    plt.ylabel('Récompense cumulée')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()

def plot_comparison(q_rewards, base_rewards, window=50, title="Comparaison", save_path=None):
    plt.figure(figsize=(10, 6))
    
    def get_ma(data):
        ma = []
        for i in range(len(data)):
            start = max(0, i - window)
            ma.append(sum(data[start:i+1]) / (i - start + 1))
        return ma
        
    plt.plot(get_ma(q_rewards), color='blue', label='Q-Learning')
    plt.plot(get_ma(base_rewards), color='orange', label='Baseline (Feu Fixe)')
    
    plt.title(title)
    plt.xlabel('Pas de temps')
    plt.ylabel('Récompense (Moyenne glissante)')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()
