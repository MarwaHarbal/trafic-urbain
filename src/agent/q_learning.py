import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_space_sizes=(5, 5, 5, 5, 2), n_actions=2, 
                 alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01):
        """
        Initialise l'agent Q-Learning tabulé from scratch.
        """
        # Table Q initialisée à zéro pour toutes les paires (état, action)
        self.q_table = np.zeros(state_space_sizes + (n_actions,))
        self.alpha = alpha       # Taux d'apprentissage
        self.gamma = gamma       # Facteur d'actualisation
        self.epsilon = epsilon   # Taux d'exploration initial
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.n_actions = n_actions
        
    def act(self, state, evaluate=False):
        """
        Politique epsilon-greedy.
        """
        # Exploration
        if not evaluate and random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.n_actions - 1)
            
        # Exploitation
        q_values = self.q_table[state]
        return np.argmax(q_values)
        
    def learn(self, state, action, reward, next_state):
        """
        Mise à jour de la table Q selon l'équation de Bellman.
        """
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        
        self.q_table[state][action] += self.alpha * td_error
        
    def decay_epsilon(self):
        """
        Réduit progressivement le taux d'exploration.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
