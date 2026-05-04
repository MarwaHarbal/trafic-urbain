import os
import numpy as np
from src.env.traffic_env import TrafficEnv
from src.agent.q_learning import QLearningAgent
from src.utils.visualization import plot_learning_curve

def train():
    # Création de l'environnement avec un trafic équilibré par défaut
    env = TrafficEnv()
    agent = QLearningAgent()
    
    episodes = 2000
    steps_per_episode = 100
    
    rewards_history = []
    
    print("Début de l'entraînement...")
    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0
        
        for step in range(steps_per_episode):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state)
            
            state = next_state
            ep_reward += reward
            
        agent.decay_epsilon()
        rewards_history.append(ep_reward)
        
        if (ep + 1) % 100 == 0:
            print(f"Épisode {ep+1}/{episodes} - Récompense cumulée: {ep_reward} - Epsilon: {agent.epsilon:.3f}")
            
    # Sauvegarde des résultats
    os.makedirs("results", exist_ok=True)
    plot_learning_curve(rewards_history, save_path="results/learning_curve.png")
    
    # Sauvegarde du modèle (Table Q)
    os.makedirs("models", exist_ok=True)
    np.save("models/q_table.npy", agent.q_table)
    print("\nEntraînement terminé. Modèle sauvegardé dans models/q_table.npy")
    print("Courbe d'apprentissage sauvegardée dans results/learning_curve.png")

if __name__ == "__main__":
    train()
