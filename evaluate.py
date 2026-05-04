import time
import numpy as np
from src.env.traffic_env import TrafficEnv
from src.agent.q_learning import QLearningAgent
from src.agent.baseline import FixedDurationAgent
from src.utils.visualization import plot_comparison

def evaluate_scenario(name, lambda_rates, steps=200, render=False):
    print(f"\n{'='*50}")
    print(f"--- Évaluation du scénario : {name} ---")
    print(f"{'='*50}")
    
    env = TrafficEnv(lambda_rates=lambda_rates)
    
    # --- Agent Q-Learning ---
    q_agent = QLearningAgent()
    try:
        q_agent.q_table = np.load("models/q_table.npy")
    except FileNotFoundError:
        print("Erreur: Table Q non trouvée. Veuillez lancer train.py d'abord.")
        return
        
    state = env.reset()
    q_rewards = []
    print("\n[1] Exécution de l'Agent Q-Learning...")
    for _ in range(steps):
        if render:
            env.render()
            time.sleep(0.3)
        action = q_agent.act(state, evaluate=True)
        next_state, reward, done, _ = env.step(action)
        q_rewards.append(reward)
        state = next_state
        
    # --- Agent Baseline ---
    base_agent = FixedDurationAgent(cycle_length=10)
    state = env.reset()
    base_rewards = []
    print("\n[2] Exécution de l'Agent Baseline (Feu Fixe)...")
    for _ in range(steps):
        action = base_agent.act(state)
        next_state, reward, done, _ = env.step(action)
        base_rewards.append(reward)
        state = next_state
        
    # --- Comparaison et Affichage ---
    plot_comparison(q_rewards, base_rewards, window=20, 
                    title=f"Comparaison Agent vs Baseline ({name})",
                    save_path=f"results/comparison_{name.lower().replace(' ', '_')}.png")
                    
    print("\n--- RÉSULTATS ---")
    print(f"Récompense totale Q-Learning : {sum(q_rewards)}")
    print(f"Récompense totale Baseline   : {sum(base_rewards)}")
    print(f"Graphique sauvegardé dans results/comparison_{name.lower().replace(' ', '_')}.png\n")

if __name__ == "__main__":
    # Scénario 1: Trafic équilibré
    balanced_rates = {'N': 0.2, 'S': 0.2, 'E': 0.2, 'W': 0.2}
    evaluate_scenario("Equilibre", balanced_rates, steps=200, render=False)
    
    # Scénario 2: Trafic asymétrique
    asym_rates = {'N': 0.4, 'S': 0.4, 'E': 0.1, 'W': 0.1}
    evaluate_scenario("Asymetrique", asym_rates, steps=200, render=False)
    
    print("\nAstuce: Pour voir l'animation temps réel, passez render=True dans l'appel à evaluate_scenario() en bas du fichier evaluate.py.")
